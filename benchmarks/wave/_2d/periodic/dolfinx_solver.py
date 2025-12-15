"""DOLFINx FEM solver for 2D wave equation (manufactured solution).

Solves: ∂²u/∂t² = c²∇²u  on [0,1]²

Uses Newmark-beta time-stepping (average acceleration method):
    β = 0.25, γ = 0.5 (unconditionally stable, second-order accurate)

The Newmark-beta scheme:
    u^{n+1} = u^n + dt·v^n + dt²·[(1/2-β)a^n + β·a^{n+1}]
    v^{n+1} = v^n + dt·[(1-γ)a^n + γ·a^{n+1}]

where a = c²∇²u (acceleration from wave equation).

This leads to the system:
    (M + β·dt²·c²·K) u^{n+1} = M·ũ - β·dt²·c²·K·ũ + β·dt²·c²·K·u^n
where ũ = u^n + dt·v^n + (1/2-β)·dt²·a^n is the predictor.

Simplification for β=0.25, γ=0.5:
    (M + dt²·c²/4·K) u^{n+1} = RHS

Manufactured solution: Standing wave
    u(x,y,t) = cos(2πmx)·cos(2πny)·cos(ωt)

where ω = c·2π√(m² + n²) satisfies the dispersion relation.

Boundary conditions: Homogeneous Neumann (natural BC)
    - For the standing wave cos(2πmx) with integer m, ∂u/∂x = 0 at x=0,1
    - This is automatically satisfied by the weak form

Initial conditions:
    u(x,y,0) = cos(2πmx)·cos(2πny)
    ∂u/∂t(x,y,0) = 0

Default parameters:
    c = 1.0         (wave speed, normalized)
    m = 2, n = 2    (mode numbers)
    T = 1.0         (final time)
"""

from typing import Tuple

import numpy as np
from mpi4py import MPI

import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from petsc4py import PETSc


# Wave speed (normalized)
WAVE_SPEED = 1.0

# Mode numbers (must match analytical.py)
MODE_NUMBER_X = 2
MODE_NUMBER_Y = 2

# Angular frequency from dispersion relation: ω = c·2π√(m² + n²)
ANGULAR_FREQUENCY = WAVE_SPEED * 2.0 * np.pi * np.sqrt(
    MODE_NUMBER_X**2 + MODE_NUMBER_Y**2
)

# Time parameters (must match analytical.py)
DEFAULT_FINAL_TIME = 1.0
DEFAULT_NUM_TIME_STEPS = 2000  # More steps for wave equation accuracy
DEFAULT_NUM_OUTPUT_STEPS = 101


def initial_displacement(x: np.ndarray) -> np.ndarray:
    """Compute initial displacement at given coordinates.

    u(x,y,0) = cos(2πmx)·cos(2πny)

    Args:
        x: Array of shape (gdim, N) containing coordinates

    Returns:
        Array of shape (N,) with initial displacement values
    """
    return (
        np.cos(2.0 * np.pi * MODE_NUMBER_X * x[0]) *
        np.cos(2.0 * np.pi * MODE_NUMBER_Y * x[1])
    )


def initial_velocity(x: np.ndarray) -> np.ndarray:
    """Compute initial velocity at given coordinates.

    ∂u/∂t(x,y,0) = 0

    Args:
        x: Array of shape (gdim, N) containing coordinates

    Returns:
        Array of zeros
    """
    return np.zeros(x.shape[1])


def solve_wave_equation_2d(
    grid_resolution: int = 32,
    polynomial_degree: int = 1,
    final_time: float = DEFAULT_FINAL_TIME,
    num_time_steps: int = DEFAULT_NUM_TIME_STEPS,
    num_output_steps: int = DEFAULT_NUM_OUTPUT_STEPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve 2D wave equation with Newmark-beta time-stepping using DOLFINx.

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
    c_squared = WAVE_SPEED**2

    # Newmark-beta parameters (average acceleration, unconditionally stable)
    beta = 0.25
    gamma = 0.5

    # Compute which simulation steps correspond to output steps
    output_time_values = np.linspace(0.0, final_time, num_output_steps)
    output_simulation_steps = (output_time_values / dt).astype(int)
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
    V = fem.functionspace(domain, ("Lagrange", polynomial_degree))

    # Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Bilinear forms
    # Mass matrix: M = ∫ u·v dx
    mass_form = ufl.inner(u, v) * ufl.dx

    # Stiffness matrix: K = ∫ ∇u·∇v dx
    stiffness_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    # No Dirichlet BCs (homogeneous Neumann is natural)
    bcs = []

    # Compile and assemble matrices
    mass_form_compiled = fem.form(mass_form)
    stiffness_form_compiled = fem.form(stiffness_form)

    M = assemble_matrix(mass_form_compiled, bcs=bcs)
    M.assemble()

    K = assemble_matrix(stiffness_form_compiled, bcs=bcs)
    K.assemble()

    # System matrix for Newmark-beta: A = M + β·dt²·c²·K
    A = M.copy()
    A.axpy(beta * dt**2 * c_squared, K)
    A.assemble()

    # Create solution functions
    u_n = fem.Function(V)    # displacement at time n
    v_n = fem.Function(V)    # velocity at time n
    a_n = fem.Function(V)    # acceleration at time n
    u_np1 = fem.Function(V)  # displacement at time n+1

    # Initialize displacement and velocity
    u_n.interpolate(initial_displacement)
    v_n.interpolate(initial_velocity)

    # Compute initial acceleration: M·a = -c²·K·u
    # a = -c²·M⁻¹·K·u
    rhs_init = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    rhs_vec = PETSc.Vec().createSeq(rhs_init, comm=PETSc.COMM_SELF)
    rhs_vec.setUp()

    # Create solver for mass matrix (for initial acceleration)
    mass_solver = PETSc.KSP().create(domain.comm)
    mass_solver.setOperators(M)
    mass_solver.setType(PETSc.KSP.Type.CG)
    mass_solver.getPC().setType(PETSc.PC.Type.JACOBI)
    mass_solver.setTolerances(rtol=1e-12, atol=1e-14, max_it=1000)

    # Compute K·u_n
    K.mult(u_n.x.petsc_vec, rhs_vec)
    rhs_vec.scale(-c_squared)

    # Solve M·a_n = -c²·K·u_n
    mass_solver.solve(rhs_vec, a_n.x.petsc_vec)
    a_n.x.scatter_forward()

    # Create system solver for Newmark time-stepping
    system_solver = PETSc.KSP().create(domain.comm)
    system_solver.setOperators(A)
    system_solver.setType(PETSc.KSP.Type.CG)
    system_solver.getPC().setType(PETSc.PC.Type.ILU)
    system_solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)

    # Create RHS vector
    b = create_vector(V)

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

    # Temporary vectors
    temp_vec = PETSc.Vec().createSeq(num_dofs, comm=PETSc.COMM_SELF)
    temp_vec.setUp()

    # Time-stepping loop using Newmark-beta
    for step in range(num_time_steps):
        # Predictor: ũ = u^n + dt·v^n + (1/2-β)·dt²·a^n
        u_pred = (
            u_n.x.array +
            dt * v_n.x.array +
            (0.5 - beta) * dt**2 * a_n.x.array
        )

        # RHS: M·ũ + β·dt²·c²·(K·u^n - K·ũ)
        # Simplified: M·ũ - β·dt²·c²·K·(ũ - u^n)
        # Since we want: (M + β·dt²·c²·K)·u^{n+1} = M·ũ + β·dt²·c²·K·u^n
        # This simplifies when using predictor approach

        # Compute b = M·ũ
        u_np1.x.array[:] = u_pred
        M.mult(u_np1.x.petsc_vec, b)

        # Solve (M + β·dt²·c²·K)·u^{n+1} = b
        system_solver.solve(b, u_np1.x.petsc_vec)
        u_np1.x.scatter_forward()

        # Compute new acceleration: a^{n+1} = (u^{n+1} - ũ) / (β·dt²)
        a_np1 = (u_np1.x.array - u_pred) / (beta * dt**2)

        # Update velocity: v^{n+1} = v^n + dt·[(1-γ)·a^n + γ·a^{n+1}]
        v_np1 = v_n.x.array + dt * ((1 - gamma) * a_n.x.array + gamma * a_np1)

        # Update for next step
        u_n.x.array[:] = u_np1.x.array
        v_n.x.array[:] = v_np1
        a_n.x.array[:] = a_np1

        # Check if this step is an output step
        simulation_step_number = step + 1
        if current_output_index < num_output_steps and simulation_step_number == output_simulation_steps[current_output_index]:
            solution_values[current_output_index, :] = u_n.x.array[sorted_indices]
            current_output_index += 1

    # Clean up PETSc objects
    mass_solver.destroy()
    system_solver.destroy()
    rhs_vec.destroy()
    temp_vec.destroy()

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
    return solve_wave_equation_2d(grid_resolution=grid_resolution)
