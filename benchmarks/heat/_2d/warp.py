"""Warp FEM solver for 2D heat/diffusion equation.

Solves: u_t = κ∇²u  on [0,1]²
with homogeneous Dirichlet BC: u = 0 on boundary
and initial condition: u₀(x,y) = sin(πx)sin(πy)

Uses backward Euler (implicit) time-stepping for unconditional stability:
    (M + κΔt·K) u^{n+1} = M·u^n

where M is the mass matrix and K is the stiffness matrix.

Default parameters:
    κ (diffusivity) = 0.01
    T (final time) = 0.1
    Δt = T/100 (100 time steps)
"""

import numpy as np
from typing import Tuple

import warp as wp
import warp.fem as fem
import warp.sparse as sparse
import warp.examples.fem.utils as fem_example_utils


# Default parameters
DEFAULT_DIFFUSIVITY = 0.01
DEFAULT_FINAL_TIME = 0.1
DEFAULT_NUM_TIME_STEPS = 100


@fem.integrand
def mass_bilinear_form(
    sample: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    """Mass matrix bilinear form: (u, v)"""
    return u(sample) * v(sample)


@fem.integrand
def stiffness_bilinear_form(
    sample: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    """Stiffness matrix bilinear form: (∇u, ∇v)"""
    return wp.dot(fem.grad(u, sample), fem.grad(v, sample))


@fem.integrand
def boundary_projector_form(
    sample: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    """Bilinear form for boundary projection: (u, v) on boundary"""
    return u(sample) * v(sample)


@wp.func
def initial_condition_function(position: wp.vec2):
    """Initial condition u₀(x,y) = sin(πx)sin(πy)"""
    x = position[0]
    y = position[1]
    pi = 3.14159265358979323846
    return wp.sin(pi * x) * wp.sin(pi * y)


def solve_heat_2d(
    grid_resolution: int = 32,
    polynomial_degree: int = 1,
    diffusivity: float = DEFAULT_DIFFUSIVITY,
    final_time: float = DEFAULT_FINAL_TIME,
    num_time_steps: int = DEFAULT_NUM_TIME_STEPS,
    quiet: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve 2D heat equation with backward Euler time-stepping.

    Solves: u_t = κ∇²u on [0,1]²
    with u = 0 on boundary and u₀(x,y) = sin(πx)sin(πy).

    Args:
        grid_resolution: Number of cells in each dimension
        polynomial_degree: Polynomial degree for Lagrange elements
        diffusivity: Thermal diffusivity κ
        final_time: Final simulation time T
        num_time_steps: Number of time steps
        quiet: If True, suppress solver output

    Returns:
        Tuple of (solution_values, node_positions)
        - solution_values: shape (N,) array of u at each node at final time
        - node_positions: shape (N, 2) array of (x, y) coordinates
    """
    time_step_size = final_time / num_time_steps

    # Create 2D grid geometry on unit square [0, 1]²
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

    # Assemble mass matrix M
    mass_matrix = fem.integrate(
        mass_bilinear_form,
        fields={"u": trial, "v": test},
        output_dtype=float,
    )

    # Assemble stiffness matrix K
    stiffness_matrix = fem.integrate(
        stiffness_bilinear_form,
        fields={"u": trial, "v": test},
        output_dtype=float,
    )

    # Build the system matrix: A = M + κΔt·K
    # For backward Euler: A u^{n+1} = M u^n
    system_matrix = sparse.bsr_axpy(
        x=stiffness_matrix,
        y=mass_matrix,
        alpha=diffusivity * time_step_size,
        beta=1.0,
    )

    # Set up boundary conditions
    boundary = fem.BoundarySides(geometry)
    boundary_trial = fem.make_trial(space=scalar_space, domain=boundary)
    boundary_test = fem.make_test(space=scalar_space, domain=boundary)

    boundary_projector = fem.integrate(
        boundary_projector_form,
        fields={"u": boundary_trial, "v": boundary_test},
        assembly="nodal",
        output_dtype=float,
    )

    # Initialize solution with initial condition
    initial_field = fem.ImplicitField(
        domain=domain,
        func=initial_condition_function,
    )

    # Project initial condition onto FEM space
    u_current = scalar_space.make_field()
    fem.interpolate(initial_field, dest=u_current)

    # Allocate arrays for time-stepping
    num_dofs = u_current.dof_values.shape[0]
    u_next = wp.zeros(num_dofs, dtype=float)
    rhs_vector = wp.zeros(num_dofs, dtype=float)

    # Time-stepping loop
    for step in range(num_time_steps):
        # Compute RHS: M·u^n using matrix-vector multiplication
        sparse.bsr_mv(
            A=mass_matrix,
            x=u_current.dof_values,
            y=rhs_vector,
            alpha=1.0,
            beta=0.0,
        )

        # Apply boundary conditions to system
        # Make a copy of system matrix for this step (BC projection modifies it)
        system_matrix_step = sparse.bsr_copy(system_matrix)

        # Project linear system to enforce homogeneous Dirichlet BC
        fem.project_linear_system(system_matrix_step, rhs_vector, boundary_projector)

        # Solve for u^{n+1}
        u_next.zero_()
        fem_example_utils.bsr_cg(
            system_matrix_step,
            b=rhs_vector,
            x=u_next,
            quiet=quiet,
        )

        # Update current solution
        wp.copy(u_next, u_current.dof_values)

    # Extract node positions
    node_positions = scalar_space.node_positions().numpy()

    return u_current.dof_values.numpy(), node_positions


def solve(grid_resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unified solver interface for CLI.

    Args:
        grid_resolution: Number of cells in each dimension

    Returns:
        Tuple of (solution_values, node_positions)
        - solution_values: shape (N,) array of u at each node at final time
        - node_positions: shape (N, 2) array of (x, y) coordinates
    """
    import warp as wp
    wp.init()
    return solve_heat_2d(grid_resolution=grid_resolution, quiet=True)
