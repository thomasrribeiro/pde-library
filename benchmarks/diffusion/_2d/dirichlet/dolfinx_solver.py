"""DOLFINx FEM solver for 2D heat/diffusion equation.

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
"""

from typing import Tuple

import numpy as np
from mpi4py import MPI

import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from petsc4py import PETSc


# Default parameters (must match analytical.py and warp.py)
DEFAULT_DIFFUSIVITY = 0.01  # Moderate diffusivity
DEFAULT_FINAL_TIME = 0.5    # Enough time to see significant spreading
DEFAULT_NUM_TIME_STEPS = 1000  # Fine time resolution for accuracy
DEFAULT_NUM_OUTPUT_STEPS = 51  # More frames for smooth animation

# Gaussian initial condition parameters
GAUSSIAN_CENTER_X = 0.5
GAUSSIAN_CENTER_Y = 0.5
GAUSSIAN_WIDTH_SIGMA = 0.1  # Moderate width (well-resolved at typical resolutions)
GAUSSIAN_AMPLITUDE = 1.0


def gaussian_initial_condition(x: np.ndarray) -> np.ndarray:
    """Compute Gaussian initial condition at given coordinates.

    Args:
        x: Array of shape (gdim, N) containing coordinates

    Returns:
        Array of shape (N,) with initial condition values
    """
    dx = x[0] - GAUSSIAN_CENTER_X
    dy = x[1] - GAUSSIAN_CENTER_Y
    r_squared = dx**2 + dy**2
    return GAUSSIAN_AMPLITUDE * np.exp(-r_squared / (GAUSSIAN_WIDTH_SIGMA**2))


def solve_heat_2d(
    grid_resolution: int = 32,
    polynomial_degree: int = 1,
    diffusivity: float = DEFAULT_DIFFUSIVITY,
    final_time: float = DEFAULT_FINAL_TIME,
    num_time_steps: int = DEFAULT_NUM_TIME_STEPS,
    num_output_steps: int = DEFAULT_NUM_OUTPUT_STEPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve 2D heat equation with backward Euler time-stepping using DOLFINx.

    Solves: u_t = κ∇²u on [0,1]²
    with u = 0 on boundary and Gaussian IC at center.

    Args:
        grid_resolution: Number of cells in each dimension
        polynomial_degree: Polynomial degree for Lagrange elements
        diffusivity: Thermal diffusivity κ
        final_time: Final simulation time T
        num_time_steps: Number of time steps for integration
        num_output_steps: Number of output time steps (including t=0)

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
    output_simulation_steps[0] = 0
    output_simulation_steps[-1] = num_time_steps

    # Create unit square mesh with quadrilateral elements
    domain = mesh.create_unit_square(
        comm=MPI.COMM_WORLD,
        nx=grid_resolution,
        ny=grid_resolution,
        cell_type=mesh.CellType.quadrilateral,
    )

    # Create scalar Lagrange function space
    function_space = fem.functionspace(domain, ("Lagrange", polynomial_degree))

    # Define trial and test functions
    u = ufl.TrialFunction(function_space)
    v = ufl.TestFunction(function_space)

    # Bilinear forms for heat equation
    # Mass matrix: M = ∫ u·v dx
    mass_form = ufl.inner(u, v) * ufl.dx

    # Stiffness matrix: K = ∫ ∇u·∇v dx
    stiffness_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    # System matrix for backward Euler: A = M + κΔt·K
    system_form = mass_form + diffusivity * time_step_size * stiffness_form

    # Homogeneous Dirichlet BC on all boundaries
    def boundary_marker(x):
        return (
            np.isclose(x[0], 0.0) |
            np.isclose(x[0], 1.0) |
            np.isclose(x[1], 0.0) |
            np.isclose(x[1], 1.0)
        )

    boundary_dofs = fem.locate_dofs_geometrical(function_space, boundary_marker)
    zero_bc = fem.Function(function_space)
    zero_bc.x.array[:] = 0.0
    boundary_condition = fem.dirichletbc(zero_bc, boundary_dofs)

    # Assemble system matrix with BC
    system_form_compiled = fem.form(system_form)
    system_matrix = assemble_matrix(system_form_compiled, bcs=[boundary_condition])
    system_matrix.assemble()

    # Assemble mass matrix (for RHS computation)
    mass_form_compiled = fem.form(mass_form)
    mass_matrix = assemble_matrix(mass_form_compiled)
    mass_matrix.assemble()

    # Create solution and RHS vectors
    u_current = fem.Function(function_space)
    u_next = fem.Function(function_space)
    rhs_vector = create_vector(function_space)

    # Initialize with Gaussian initial condition
    u_current.interpolate(gaussian_initial_condition)

    # Set up KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(system_matrix)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    # Get node positions
    node_positions = function_space.tabulate_dof_coordinates()[:, :2]

    # Round coordinates to eliminate floating-point noise
    node_positions = np.round(node_positions, decimals=10)

    # Create sorted indices for consistent ordering with other solvers
    x_coords = node_positions[:, 0]
    y_coords = node_positions[:, 1]
    sorted_indices = np.lexsort((y_coords, x_coords))

    # Allocate output storage
    num_dofs = u_current.x.array.shape[0]
    solution_values = np.zeros((num_output_steps, num_dofs))

    # Store initial condition (t=0)
    current_output_index = 0
    solution_values[current_output_index, :] = u_current.x.array[sorted_indices]
    current_output_index += 1

    # Time-stepping loop
    for step in range(num_time_steps):
        # Compute RHS: M·u^n
        with rhs_vector.localForm() as loc:
            loc.set(0.0)
        mass_matrix.mult(u_current.x.petsc_vec, rhs_vector)

        # Apply boundary conditions to RHS
        fem.petsc.apply_lifting(rhs_vector, [system_form_compiled], [[boundary_condition]])
        rhs_vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(rhs_vector, [boundary_condition])

        # Solve for u^{n+1}
        solver.solve(rhs_vector, u_next.x.petsc_vec)
        u_next.x.scatter_forward()

        # Update for next step
        u_current.x.array[:] = u_next.x.array

        # Check if this step is an output step
        simulation_step_number = step + 1
        if current_output_index < num_output_steps and simulation_step_number == output_simulation_steps[current_output_index]:
            solution_values[current_output_index, :] = u_current.x.array[sorted_indices]
            current_output_index += 1

    # Clean up PETSc objects
    solver.destroy()

    # Reorder node positions to match solution ordering
    node_positions = node_positions[sorted_indices]

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
    return solve_heat_2d(grid_resolution=grid_resolution)
