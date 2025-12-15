"""Warp FEM solver for 2D wave equation (manufactured solution).

Solves: ∂²u/∂t² = c²∇²u  on [0,1]²

Uses EXPLICIT central differences with mass lumping for fast time-stepping:
    u^{n+1} = 2u^n - u^{n-1} + dt²·c²·M_lumped⁻¹·(-K·u^n)

This avoids linear solves at each time step (unlike implicit Newmark-beta).

Manufactured solution: Standing wave
    u(x,y,t) = cos(2πmx)·cos(2πny)·cos(ωt)

where ω = c·2π√(m² + n²) satisfies the dispersion relation.

Boundary conditions: Homogeneous Neumann (natural BC)
    - For the standing wave cos(2πmx) with integer m, ∂u/∂x = 0 at x=0,1
    - This is automatically satisfied by the weak form (no boundary terms needed)

Initial conditions:
    u(x,y,0) = cos(2πmx)·cos(2πny)
    ∂u/∂t(x,y,0) = 0

Default parameters:
    c = 1.0         (wave speed, normalized)
    m = 2, n = 2    (mode numbers)
    T = 1.0         (final time)

CFL condition for stability: dt ≤ h/(c·√2) where h is mesh spacing
"""

import numpy as np
from typing import Tuple

import warp as wp
import warp.fem as fem
from warp.sparse import bsr_mv


# Wave speed (normalized)
WAVE_SPEED = 1.0

# Mode numbers (must match analytical.py)
MODE_NUMBER_X = 2
MODE_NUMBER_Y = 2

# Time parameters (must match analytical.py)
DEFAULT_FINAL_TIME = 1.0
DEFAULT_NUM_OUTPUT_STEPS = 101


@wp.func
def initial_displacement_function(position: wp.vec2):
    """Initial displacement: standing wave at t=0.

    u(x,y,0) = cos(2πmx)·cos(2πny)
    """
    x = position[0]
    y = position[1]
    two_pi = 2.0 * 3.141592653589793
    return wp.cos(two_pi * 2.0 * x) * wp.cos(two_pi * 2.0 * y)


@fem.integrand
def mass_form(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    """Mass matrix bilinear form: (u, v)."""
    return u(s) * v(s)


@fem.integrand
def stiffness_form(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    """Stiffness matrix bilinear form: (∇u, ∇v)."""
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@wp.kernel
def compute_lumped_mass_inverse(
    mass_row_sums: wp.array(dtype=float),
    lumped_mass_inverse: wp.array(dtype=float),
):
    """Compute inverse of lumped (diagonal) mass matrix."""
    i = wp.tid()
    m = mass_row_sums[i]
    if m > 1e-14:
        lumped_mass_inverse[i] = 1.0 / m
    else:
        lumped_mass_inverse[i] = 0.0


@wp.kernel
def explicit_wave_update(
    u_current: wp.array(dtype=float),
    u_previous: wp.array(dtype=float),
    k_u: wp.array(dtype=float),
    lumped_mass_inverse: wp.array(dtype=float),
    dt_squared_c_squared: float,
    u_next: wp.array(dtype=float),
):
    """Explicit central difference update for wave equation.

    u^{n+1} = 2u^n - u^{n-1} + dt²·c²·M_lumped⁻¹·(-K·u^n)
    """
    i = wp.tid()
    # Central difference: u_next = 2*u_current - u_previous + dt²*c²*M⁻¹*(-K*u)
    u_next[i] = (
        2.0 * u_current[i]
        - u_previous[i]
        - dt_squared_c_squared * lumped_mass_inverse[i] * k_u[i]
    )


def compute_lumped_mass_from_consistent(mass_matrix) -> wp.array:
    """Compute lumped mass by summing rows of consistent mass matrix.

    This is a simple row-sum lumping approach.
    """
    # Get the number of rows from the matrix
    num_rows = mass_matrix.nrow

    # Create a vector of ones to sum each row
    ones = wp.ones(num_rows, dtype=float)
    row_sums = wp.zeros(num_rows, dtype=float)

    # Compute M * ones = row sums
    bsr_mv(A=mass_matrix, x=ones, y=row_sums, alpha=1.0, beta=0.0)

    return row_sums


def solve_wave_equation_2d(
    grid_resolution: int = 32,
    polynomial_degree: int = 1,
    final_time: float = DEFAULT_FINAL_TIME,
    num_output_steps: int = DEFAULT_NUM_OUTPUT_STEPS,
    quiet: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve 2D wave equation with explicit central differences using Warp FEM.

    Uses mass lumping to avoid linear solves at each time step.

    Args:
        grid_resolution: Number of cells in each dimension
        polynomial_degree: Polynomial degree for Lagrange elements
        final_time: Final simulation time T
        num_output_steps: Number of output time steps (including t=0)
        quiet: If True, suppress solver output

    Returns:
        Tuple of (solution_values, node_positions, time_values)
        - solution_values: shape (num_output_steps, num_nodes) array
        - node_positions: shape (num_nodes, 2) array of (x, y) coordinates
        - time_values: shape (num_output_steps,) array of time values
    """
    c_squared = WAVE_SPEED**2

    # Mesh spacing
    h = 1.0 / grid_resolution

    # CFL condition for explicit scheme with lumped mass FEM
    # The standard FD CFL is dt ≤ h / (c * sqrt(2))
    # But with mass lumping on bilinear quads, the effective max eigenvalue of M_L^-1*K
    # is larger, requiring a smaller timestep.
    # Use safety factor of 0.1 for stability
    cfl_dt = 0.1 * h / (WAVE_SPEED * np.sqrt(2.0))

    # Determine number of time steps needed
    num_time_steps = int(np.ceil(final_time / cfl_dt))
    dt = final_time / num_time_steps

    if not quiet:
        print(f"Grid resolution: {grid_resolution}, h = {h:.4f}")
        print(f"CFL dt = {cfl_dt:.6f}, actual dt = {dt:.6f}")
        print(f"Number of time steps: {num_time_steps}")

    # Compute which simulation steps correspond to output steps
    output_time_values = np.linspace(0.0, final_time, num_output_steps)
    output_simulation_steps = (output_time_values / dt).astype(int)
    output_simulation_steps[0] = 0
    output_simulation_steps[-1] = num_time_steps

    # Create 2D grid geometry
    geometry = fem.Grid2D(
        bounds_lo=wp.vec2(0.0, 0.0),
        bounds_hi=wp.vec2(1.0, 1.0),
        res=wp.vec2i(grid_resolution, grid_resolution),
    )

    # Create scalar Lagrange function space
    scalar_space = fem.make_polynomial_space(
        geometry,
        degree=polynomial_degree,
    )

    # Create domain for integration
    domain = fem.Cells(geometry)

    # Create trial and test functions
    trial = fem.make_trial(space=scalar_space, domain=domain)
    test = fem.make_test(space=scalar_space, domain=domain)

    # Assemble consistent mass matrix
    mass_matrix = fem.integrate(
        mass_form,
        fields={"u": trial, "v": test},
        output_dtype=float,
    )

    # Assemble stiffness matrix
    stiffness_matrix = fem.integrate(
        stiffness_form,
        fields={"u": trial, "v": test},
        output_dtype=float,
    )

    # Compute lumped mass inverse
    lumped_mass = compute_lumped_mass_from_consistent(mass_matrix)
    num_dofs = lumped_mass.shape[0]
    lumped_mass_inverse = wp.zeros(num_dofs, dtype=float)
    wp.launch(
        compute_lumped_mass_inverse,
        dim=num_dofs,
        inputs=[lumped_mass, lumped_mass_inverse],
    )

    # Initialize displacement with initial condition
    u_current = scalar_space.make_field()
    initial_u_field = fem.ImplicitField(
        domain=domain,
        func=initial_displacement_function,
    )
    fem.interpolate(initial_u_field, dest=u_current)

    # Get node positions
    node_positions = scalar_space.node_positions().numpy()

    # Initialize u_previous for first step
    # Since ∂u/∂t(0) = 0, we use: u^{-1} = u^0 - dt*v^0 + (dt²/2)*a^0
    # With v^0 = 0, this simplifies to: u^{-1} = u^0 + (dt²/2)*a^0
    # where a^0 = c²*∇²u^0 = -c²*M⁻¹*K*u^0

    u_previous = wp.zeros(num_dofs, dtype=float)
    k_u = wp.zeros(num_dofs, dtype=float)

    # Compute K*u^0
    bsr_mv(A=stiffness_matrix, x=u_current.dof_values, y=k_u, alpha=1.0, beta=0.0)

    # u^{-1} = u^0 - (dt²/2)*c²*M⁻¹*K*u^0
    u_current_np = u_current.dof_values.numpy()
    k_u_np = k_u.numpy()
    lumped_mass_inv_np = lumped_mass_inverse.numpy()
    u_previous_np = u_current_np - 0.5 * dt**2 * c_squared * lumped_mass_inv_np * k_u_np
    u_previous = wp.array(u_previous_np, dtype=float)

    # Allocate output storage
    solution_values = np.zeros((num_output_steps, num_dofs))

    # Store initial condition (t=0)
    current_output_index = 0
    solution_values[current_output_index, :] = u_current.dof_values.numpy()
    current_output_index += 1

    # Working arrays - use plain warp arrays instead of field
    u_current_arr = wp.clone(u_current.dof_values)
    u_next = wp.zeros(num_dofs, dtype=float)
    dt_squared_c_squared = dt**2 * c_squared

    # Time-stepping loop using explicit central differences
    for step in range(num_time_steps):
        # Compute K*u_current
        bsr_mv(A=stiffness_matrix, x=u_current_arr, y=k_u, alpha=1.0, beta=0.0)

        # Explicit update: u_next = 2*u_current - u_previous - dt²*c²*M⁻¹*K*u_current
        wp.launch(
            explicit_wave_update,
            dim=num_dofs,
            inputs=[
                u_current_arr,
                u_previous,
                k_u,
                lumped_mass_inverse,
                dt_squared_c_squared,
                u_next,
            ],
        )

        # Rotate arrays: previous <- current, current <- next
        u_previous, u_current_arr, u_next = u_current_arr, u_next, u_previous

        # Check if this step is an output step
        simulation_step_number = step + 1
        if current_output_index < num_output_steps and simulation_step_number == output_simulation_steps[current_output_index]:
            solution_values[current_output_index, :] = u_current_arr.numpy()
            current_output_index += 1

    return solution_values, node_positions, output_time_values


def solve(grid_resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unified solver interface for CLI.

    Args:
        grid_resolution: Number of cells in each dimension

    Returns:
        Tuple of (solution_values, node_positions, time_values)
        - solution_values: shape (num_time_steps, num_nodes) array
        - node_positions: shape (num_nodes, 2) array of (x, y) coordinates
        - time_values: shape (num_time_steps,) array of time values
    """
    wp.init()
    return solve_wave_equation_2d(grid_resolution=grid_resolution, quiet=True)
