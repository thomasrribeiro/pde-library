"""Warp FEM solver for 2D heat/diffusion equation.

Solves: u_t = κ∇²u  on [0,1]²
with homogeneous Dirichlet BC: u = 0 on boundary

Initial condition: Gaussian blob at center (0.5, 0.5)
    u₀(x,y) = A·exp(-((x-0.5)² + (y-0.5)²) / σ²)

    - Moderate width (σ = 0.1) for good numerical resolution
    - Centered heat source that diffuses outward

Uses backward Euler (implicit) time-stepping for unconditional stability:
    (M + κΔt·K) u^{n+1} = M·u^n

where M is the mass matrix and K is the stiffness matrix.

Default parameters:
    κ (diffusivity) = 0.01  (moderate diffusivity)
    σ (Gaussian width) = 0.1 (well-resolved at typical grid resolutions)
    A (amplitude) = 1.0
    T (final time) = 0.5 (enough time to see significant spreading)
    Δt = T/1000 (1000 time steps)
"""

import numpy as np
from typing import Tuple

import warp as wp
import warp.fem as fem
import warp.sparse as sparse
import warp.examples.fem.utils as fem_example_utils


# Default parameters (must match analytical.py)
DEFAULT_DIFFUSIVITY = 0.01  # Moderate diffusivity
DEFAULT_FINAL_TIME = 0.5    # Enough time to see significant spreading
DEFAULT_NUM_TIME_STEPS = 1000  # Fine time resolution for accuracy
DEFAULT_NUM_OUTPUT_STEPS = 51  # More frames for smooth animation

# Gaussian initial condition parameters (must match analytical.py)
GAUSSIAN_CENTER_X = 0.5
GAUSSIAN_CENTER_Y = 0.5
GAUSSIAN_WIDTH_SIGMA = 0.1  # Moderate width (well-resolved at typical resolutions)
GAUSSIAN_AMPLITUDE = 1.0


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
    """Initial condition: Gaussian blob at center.

    u₀(x,y) = A·exp(-((x-x₀)² + (y-y₀)²) / σ²)
    """
    x = position[0]
    y = position[1]

    # Gaussian parameters (must match module constants)
    center_x = 0.5
    center_y = 0.5
    sigma = 0.1  # Moderate width
    amplitude = 1.0

    dx = x - center_x
    dy = y - center_y
    r_squared = dx * dx + dy * dy

    return amplitude * wp.exp(-r_squared / (sigma * sigma))


def solve_heat_2d(
    grid_resolution: int = 32,
    polynomial_degree: int = 1,
    diffusivity: float = DEFAULT_DIFFUSIVITY,
    final_time: float = DEFAULT_FINAL_TIME,
    num_time_steps: int = DEFAULT_NUM_TIME_STEPS,
    num_output_steps: int = DEFAULT_NUM_OUTPUT_STEPS,
    quiet: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve 2D heat equation with backward Euler time-stepping.

    Solves: u_t = κ∇²u on [0,1]²
    with u = 0 on boundary and Gaussian IC at center.

    Args:
        grid_resolution: Number of cells in each dimension
        polynomial_degree: Polynomial degree for Lagrange elements
        diffusivity: Thermal diffusivity κ
        final_time: Final simulation time T
        num_time_steps: Number of time steps for integration
        num_output_steps: Number of output time steps (including t=0)
        quiet: If True, suppress solver output

    Returns:
        Tuple of (solution_values, node_positions, time_values)
        - solution_values: shape (num_output_steps, num_nodes) array of u at each node over time
        - node_positions: shape (num_nodes, 2) array of (x, y) coordinates
        - time_values: shape (num_output_steps,) array of time values
    """
    time_step_size = final_time / num_time_steps

    # Compute which simulation steps correspond to output steps
    output_time_values = np.linspace(0.0, final_time, num_output_steps)
    output_simulation_steps = (output_time_values / time_step_size).astype(int)
    # Ensure first step is 0 and last is num_time_steps
    output_simulation_steps[0] = 0
    output_simulation_steps[-1] = num_time_steps

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
    # Note: bsr_axpy modifies y in-place, so we pass y=None to allocate a new matrix
    # Formula: result = alpha*x + beta*y, with y=None treated as zero
    # So we first compute κΔt·K, then add M
    system_matrix = sparse.bsr_axpy(
        x=stiffness_matrix,
        y=None,
        alpha=diffusivity * time_step_size,
        beta=0.0,
    )
    # Now add M to get A = M + κΔt·K
    sparse.bsr_axpy(
        x=mass_matrix,
        y=system_matrix,
        alpha=1.0,
        beta=1.0,
    )

    # Initialize solution with initial condition
    initial_field = fem.ImplicitField(
        domain=domain,
        func=initial_condition_function,
    )

    # Project initial condition onto FEM space
    u_current = scalar_space.make_field()
    fem.interpolate(initial_field, dest=u_current)

    # Get number of DOFs and extract node positions
    num_dofs = u_current.dof_values.shape[0]

    # Extract node positions
    node_positions = scalar_space.node_positions().numpy()

    # Identify boundary nodes for applying Dirichlet BC
    # Nodes are on boundary if x=0, x=1, y=0, or y=1
    boundary_mask = (
        (np.abs(node_positions[:, 0]) < 1e-10) |
        (np.abs(node_positions[:, 0] - 1.0) < 1e-10) |
        (np.abs(node_positions[:, 1]) < 1e-10) |
        (np.abs(node_positions[:, 1] - 1.0) < 1e-10)
    )
    boundary_indices = np.where(boundary_mask)[0]
    interior_indices = np.where(~boundary_mask)[0]

    # Convert system matrix to dense for proper BC application
    # This is necessary because Warp's project_linear_system corrupts
    # the interior-interior block for time-stepping problems
    def bsr_to_dense(bsr_matrix):
        values = bsr_matrix.values.numpy()
        row_offsets = bsr_matrix.offsets.numpy()
        col_indices = bsr_matrix.columns.numpy()
        nrow = bsr_matrix.nrow
        dense = np.zeros((nrow, nrow))
        for i in range(nrow):
            for j_idx in range(row_offsets[i], row_offsets[i + 1]):
                j = col_indices[j_idx]
                dense[i, j] = values[j_idx]
        return dense

    A_dense = bsr_to_dense(system_matrix)
    M_dense = bsr_to_dense(mass_matrix)

    # Apply homogeneous Dirichlet BC to system matrix:
    # Set boundary rows to identity, zero boundary columns
    for i in boundary_indices:
        A_dense[i, :] = 0.0
        A_dense[:, i] = 0.0
        A_dense[i, i] = 1.0

    # Allocate output storage
    solution_values = np.zeros((num_output_steps, num_dofs))

    # Store initial condition (t=0)
    current_output_index = 0
    solution_values[current_output_index, :] = u_current.dof_values.numpy()
    current_output_index += 1

    # Time-stepping loop using numpy for correct BC handling
    u_current_np = u_current.dof_values.numpy()

    for step in range(num_time_steps):
        # Compute RHS: M·u^n
        rhs_np = M_dense @ u_current_np

        # Zero out boundary entries of RHS (homogeneous Dirichlet BC)
        rhs_np[boundary_indices] = 0.0

        # Solve for u^{n+1}
        u_current_np = np.linalg.solve(A_dense, rhs_np)

        # Check if this step is an output step
        simulation_step_number = step + 1
        if current_output_index < num_output_steps and simulation_step_number == output_simulation_steps[current_output_index]:
            solution_values[current_output_index, :] = u_current_np
            current_output_index += 1

    return solution_values, node_positions, output_time_values


def solve(grid_resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unified solver interface for CLI.

    Args:
        grid_resolution: Number of cells in each dimension

    Returns:
        Tuple of (solution_values, node_positions, time_values)
        - solution_values: shape (num_time_steps, num_nodes) array of u at each node over time
        - node_positions: shape (num_nodes, 2) array of (x, y) coordinates
        - time_values: shape (num_time_steps,) array of time values
    """
    import warp as wp
    wp.init()
    return solve_heat_2d(grid_resolution=grid_resolution, quiet=True)
