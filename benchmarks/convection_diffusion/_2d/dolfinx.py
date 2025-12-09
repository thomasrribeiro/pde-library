"""DOLFINx FEM solver for 2D convection-diffusion equation.

Solves: ∂u/∂t + c⃗·∇u = κ∇²u  on [0,1]²

Uses SUPG (Streamline Upwind Petrov-Galerkin) stabilization for the
advection term combined with standard Galerkin for diffusion.

The SUPG weak form for convection-diffusion:
    ∫(∂u/∂t + c⃗·∇u - κ∇²u)·(v + τ·c⃗·∇v) dx = 0

Time discretization: Backward Euler for stability
    (u^{n+1} - u^n)/dt + c⃗·∇u^{n+1} = κ∇²u^{n+1}

The Péclet number Pe = |c|h/(2κ) determines whether advection or diffusion
dominates. When Pe >> 1 (advection-dominated), SUPG stabilization is crucial.

Initial condition: Gaussian blob at (0.3, 0.3)
Velocity field: Constant c⃗ = (0.4, 0.4) (diagonal transport)

Boundary conditions:
    - Homogeneous Neumann on all boundaries (natural BC for diffusion)
    - No inflow BC enforcement (Gaussian stays away from boundaries)

Default parameters:
    c⃗ = (0.4, 0.4)  (diagonal transport)
    κ = 0.01        (diffusivity)
    σ = 0.1         (Gaussian width)
    A = 1.0         (amplitude)
    (x₀, y₀) = (0.3, 0.3)  (initial center)
    T = 1.0         (final time - blob moves to (0.7, 0.7) while spreading)
"""

from typing import Tuple

import numpy as np
from mpi4py import MPI

import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from petsc4py import PETSc


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


def solve_convection_diffusion_2d(
    grid_resolution: int = 32,
    polynomial_degree: int = 1,
    final_time: float = DEFAULT_FINAL_TIME,
    num_time_steps: int = DEFAULT_NUM_TIME_STEPS,
    num_output_steps: int = DEFAULT_NUM_OUTPUT_STEPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve 2D convection-diffusion equation with SUPG stabilization using DOLFINx.

    Args:
        grid_resolution: Number of cells in each dimension
        polynomial_degree: Polynomial degree for Lagrange elements
        final_time: Final simulation time T
        num_time_steps: Number of time steps for integration
        num_output_steps: Number of output time steps (including t=0)

    Returns:
        Tuple of (solution_values, node_positions, time_values)
        - solution_values: shape (num_output_steps, num_nodes) array
        - node_positions: shape (num_nodes, 2) array of (x, y) coordinates
        - time_values: shape (num_output_steps,) array of time values
    """
    dt = final_time / num_time_steps

    # Compute which simulation steps correspond to output steps
    output_time_values = np.linspace(0.0, final_time, num_output_steps)
    output_simulation_steps = (output_time_values / dt).astype(int)
    output_simulation_steps[0] = 0
    output_simulation_steps[-1] = num_time_steps

    # Create unit square mesh
    domain = mesh.create_unit_square(
        comm=MPI.COMM_WORLD,
        nx=grid_resolution,
        ny=grid_resolution,
        cell_type=mesh.CellType.quadrilateral,
    )

    # Create scalar Lagrange function space
    V = fem.functionspace(domain, ("Lagrange", polynomial_degree))

    # Define constant velocity field
    c = ufl.as_vector([VELOCITY_X, VELOCITY_Y])

    # Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Element size (for SUPG stabilization parameter)
    h = ufl.CellDiameter(domain)

    # Velocity magnitude
    c_mag = ufl.sqrt(ufl.dot(c, c))

    # SUPG stabilization parameter
    # τ = h / (2 * |c|) is modified for convection-diffusion
    # Use the formula that accounts for both advection and diffusion
    # τ = h / (2 * |c|) * (coth(Pe) - 1/Pe) where Pe = |c|h/(2κ)
    # For simplicity, use the advection-dominated limit: τ = h / (2 * |c|)
    tau = h / (2.0 * c_mag)

    # Time discretization: backward Euler
    # (u^{n+1} - u^n)/dt + c·∇u^{n+1} - κ∇²u^{n+1} = 0
    #
    # SUPG weak form:
    # ∫ u^{n+1}·v_supg dx + dt·∫ (c·∇u^{n+1})·v_supg dx
    #   + dt·κ·∫ ∇u^{n+1}·∇v dx = ∫ u^n·v_supg dx
    #
    # Note: Diffusion term uses standard Galerkin (no SUPG modification for 2nd derivatives)

    # SUPG test function modification (only for advection terms)
    v_supg = v + tau * ufl.dot(c, ufl.grad(v))

    # Bilinear form: mass + advection (SUPG) + diffusion (standard Galerkin)
    a = (
        ufl.inner(u, v_supg) * ufl.dx
        + dt * ufl.inner(ufl.dot(c, ufl.grad(u)), v_supg) * ufl.dx
        + dt * DIFFUSIVITY * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    )

    # Create functions for current and previous solutions
    u_n = fem.Function(V)  # u at time n
    u_np1 = fem.Function(V)  # u at time n+1

    # Linear form: RHS from previous time step
    L = ufl.inner(u_n, v_supg) * ufl.dx

    # Initialize with Gaussian initial condition
    u_n.interpolate(gaussian_initial_condition)

    # No Dirichlet boundary conditions (homogeneous Neumann is natural BC)
    bcs = []

    # Compile forms
    a_compiled = fem.form(a)
    L_compiled = fem.form(L)

    # Assemble system matrix (constant for linear problem)
    A = assemble_matrix(a_compiled, bcs=bcs)
    A.assemble()

    # Create RHS vector
    b = create_vector(V)

    # Set up KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    solver.getPC().setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)

    # Get node positions
    node_positions = V.tabulate_dof_coordinates()[:, :2]
    node_positions = np.round(node_positions, decimals=10)

    # Create sorted indices for consistent ordering
    x_coords = node_positions[:, 0]
    y_coords = node_positions[:, 1]
    sorted_indices = np.lexsort((y_coords, x_coords))

    # Allocate output storage
    num_dofs = u_n.x.array.shape[0]
    solution_values = np.zeros((num_output_steps, num_dofs))

    # Store initial condition (t=0)
    current_output_index = 0
    solution_values[current_output_index, :] = u_n.x.array[sorted_indices]
    current_output_index += 1

    # Time-stepping loop
    for step in range(num_time_steps):
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        assemble_vector(b, L_compiled)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        # Solve
        solver.solve(b, u_np1.x.petsc_vec)
        u_np1.x.scatter_forward()

        # Update for next step
        u_n.x.array[:] = u_np1.x.array

        # Check if this step is an output step
        simulation_step_number = step + 1
        if current_output_index < num_output_steps and simulation_step_number == output_simulation_steps[current_output_index]:
            solution_values[current_output_index, :] = u_n.x.array[sorted_indices]
            current_output_index += 1

    # Clean up
    solver.destroy()

    # Reorder node positions
    node_positions = node_positions[sorted_indices]

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
    return solve_convection_diffusion_2d(grid_resolution=grid_resolution)
