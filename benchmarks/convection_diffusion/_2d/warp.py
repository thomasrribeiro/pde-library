"""Warp FEM solver for 2D convection-diffusion equation.

Solves: ∂u/∂t + c⃗·∇u = κ∇²u  on [0,1]²

Uses semi-Lagrangian advection combined with implicit diffusion:
    - Advection: Trace particles backward along characteristics
    - Diffusion: Implicit backward Euler for stability

Time-stepping scheme (operator splitting):
    1. Semi-Lagrangian advection: evaluate u at upstream position
    2. Implicit diffusion: solve (M + κ·dt·K)u^{n+1} = M·u_advected

This approach is unconditionally stable for any time step size.

Initial condition: Gaussian blob at (0.3, 0.3)
    u₀(x,y) = A·exp(-((x-x₀)² + (y-y₀)²) / σ²)

Velocity field: Constant c⃗ = (0.4, 0.4) (diagonal transport)

Boundary handling:
    - For advection: upstream points outside domain use zero value
    - For diffusion: homogeneous Neumann (natural) BCs - flux vanishes at boundaries

Default parameters:
    c⃗ = (0.4, 0.4)  (diagonal transport)
    κ = 0.01        (diffusivity)
    σ = 0.1         (Gaussian width)
    A = 1.0         (amplitude)
    (x₀, y₀) = (0.3, 0.3)  (initial center)
    T = 1.0         (final time - blob moves to (0.7, 0.7) while spreading)
"""

import numpy as np
from typing import Tuple

import warp as wp
import warp.fem as fem


# Velocity field (constant) - must match analytical.py
VELOCITY_X = 0.4
VELOCITY_Y = 0.4

# Diffusivity - must match analytical.py
DIFFUSIVITY = 0.01

# Gaussian initial condition parameters (must match analytical.py)
GAUSSIAN_CENTER_X = 0.3
GAUSSIAN_CENTER_Y = 0.3
GAUSSIAN_WIDTH_SIGMA = 0.1
GAUSSIAN_AMPLITUDE = 1.0

# Time parameters (must match analytical.py)
DEFAULT_FINAL_TIME = 1.0
DEFAULT_NUM_TIME_STEPS = 1000
DEFAULT_NUM_OUTPUT_STEPS = 51


@wp.func
def velocity_field(pos: wp.vec2):
    """Constant velocity field c⃗ = (cx, cy)."""
    return wp.vec2(0.4, 0.4)


@wp.func
def initial_condition_function(position: wp.vec2):
    """Initial condition: Gaussian blob.

    u₀(x,y) = A·exp(-((x-x₀)² + (y-y₀)²) / σ²)
    """
    x = position[0]
    y = position[1]

    center_x = 0.3
    center_y = 0.3
    sigma = 0.1
    amplitude = 1.0

    dx = x - center_x
    dy = y - center_y
    r_squared = dx * dx + dy * dy

    return amplitude * wp.exp(-r_squared / (sigma * sigma))


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


@fem.integrand
def diffusion_and_mass_form(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
    dt: float,
    kappa: float,
):
    """Combined mass + diffusion form: (u, v) + κ·dt·(∇u, ∇v)."""
    return mass_form(s, u, v) + kappa * dt * stiffness_form(s, u, v)


@fem.integrand
def semi_lagrangian_rhs_form(
    s: fem.Sample,
    domain: fem.Domain,
    phi: fem.Field,
    psi: fem.Field,
    dt: float,
):
    """Semi-Lagrangian advection RHS.

    Evaluates phi at the upstream position (x - c⃗·dt).
    For points that trace outside the domain, we get zero (outflow behavior).
    """
    pos = fem.position(domain, s)
    vel = velocity_field(pos)

    # Trace backward along characteristic
    upstream_pos = pos - vel * dt

    # Lookup the field value at upstream position
    upstream_sample = fem.lookup(domain, upstream_pos, s)
    upstream_phi = phi(upstream_sample)

    # Check if upstream position is outside [0,1]² - if so, use zero (inflow BC)
    x_up = upstream_pos[0]
    y_up = upstream_pos[1]
    inside = float(x_up >= 0.0 and x_up <= 1.0 and y_up >= 0.0 and y_up <= 1.0)

    return inside * upstream_phi * psi(s)


def solve_convection_diffusion_2d(
    grid_resolution: int = 32,
    polynomial_degree: int = 1,
    final_time: float = DEFAULT_FINAL_TIME,
    num_time_steps: int = DEFAULT_NUM_TIME_STEPS,
    num_output_steps: int = DEFAULT_NUM_OUTPUT_STEPS,
    quiet: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve 2D convection-diffusion equation with semi-Lagrangian advection + implicit diffusion.

    Args:
        grid_resolution: Number of cells in each dimension
        polynomial_degree: Polynomial degree for Lagrange elements
        final_time: Final simulation time T
        num_time_steps: Number of time steps for integration
        num_output_steps: Number of output time steps (including t=0)
        quiet: If True, suppress solver output

    Returns:
        Tuple of (solution_values, node_positions, time_values)
        - solution_values: shape (num_output_steps, num_nodes) array
        - node_positions: shape (num_nodes, 2) array of (x, y) coordinates
        - time_values: shape (num_output_steps,) array of time values
    """
    time_step_size = final_time / num_time_steps

    # Compute which simulation steps correspond to output steps
    output_time_values = np.linspace(0.0, final_time, num_output_steps)
    output_simulation_steps = (output_time_values / time_step_size).astype(int)
    output_simulation_steps[0] = 0
    output_simulation_steps[-1] = num_time_steps

    # Create 2D grid geometry with BVH for lookup operations
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

    # Assemble system matrix: M + κ·dt·K (constant, assembled once)
    system_matrix = fem.integrate(
        diffusion_and_mass_form,
        fields={"u": trial, "v": test},
        values={"dt": time_step_size, "kappa": DIFFUSIVITY},
        output_dtype=float,
    )

    # Initialize solution with initial condition
    u_current = scalar_space.make_field()

    # Use ImplicitField for initial condition
    initial_field = fem.ImplicitField(
        domain=domain,
        func=initial_condition_function,
    )
    fem.interpolate(initial_field, dest=u_current)

    # Get number of DOFs and extract node positions
    num_dofs = u_current.dof_values.shape[0]
    node_positions = scalar_space.node_positions().numpy()

    # Allocate output storage
    solution_values = np.zeros((num_output_steps, num_dofs))

    # Store initial condition (t=0)
    current_output_index = 0
    solution_values[current_output_index, :] = u_current.dof_values.numpy()
    current_output_index += 1

    # Create field for next time step
    u_next = scalar_space.make_field()

    # Import CG solver
    from warp.optim.linear import cg

    # Time-stepping loop
    for step in range(num_time_steps):
        # Step 1: Semi-Lagrangian advection - assemble RHS with advected values
        rhs = fem.integrate(
            semi_lagrangian_rhs_form,
            fields={"phi": u_current, "psi": test},
            values={"dt": time_step_size},
            output_dtype=float,
        )

        # Step 2: Implicit diffusion - solve (M + κ·dt·K) * u_next = rhs
        # Reset u_next to zero
        u_next.dof_values.zero_()

        cg(
            A=system_matrix,
            b=rhs,
            x=u_next.dof_values,
            tol=1e-10,
            maxiter=1000,
        )

        # Swap fields for next iteration
        u_current, u_next = u_next, u_current

        # Check if this step is an output step
        simulation_step_number = step + 1
        if current_output_index < num_output_steps and simulation_step_number == output_simulation_steps[current_output_index]:
            solution_values[current_output_index, :] = u_current.dof_values.numpy()
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
    return solve_convection_diffusion_2d(grid_resolution=grid_resolution, quiet=True)
