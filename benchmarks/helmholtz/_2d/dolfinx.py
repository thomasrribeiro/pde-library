"""DOLFINx FEM solver for 2D Helmholtz plane wave equation.

Solves: -∇²u - k₀²u = 0 on [0,1]²

Mixed boundary conditions (matching Ihlenburg's book):
- Dirichlet: u = u_exact on x=0 and y=0
- Neumann: ∂u/∂n = g on x=1 and y=1

Exact solution: u(x,y) = A·cos(k₀·(cos(θ)·x + sin(θ)·y))

Reference: Ihlenburg, "Finite Element Analysis of Acoustic Scattering" (p138-139)
"""

from typing import Tuple

import numpy as np
from mpi4py import MPI

import ufl
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import LinearProblem


# Wave parameters (matching FEniCSx example from Ihlenburg's book)
WAVE_NUMBER_K0 = 4.0 * np.pi  # k₀ = 4π (~12.57)
PROPAGATION_ANGLE_THETA = np.pi / 4.0  # 45 degrees
AMPLITUDE_A = 1.0

# Derived constants
WAVE_DIRECTION_X = np.cos(PROPAGATION_ANGLE_THETA)  # cos(π/4) = √2/2
WAVE_DIRECTION_Y = np.sin(PROPAGATION_ANGLE_THETA)  # sin(π/4) = √2/2


def solve_helmholtz_2d(
    grid_resolution: int = 64,
    polynomial_degree: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve 2D Helmholtz plane wave equation with mixed BCs using DOLFINx.

    Solves: -∇²u - k₀²u = 0 on [0,1]²
    with:
        - Dirichlet BC: u = u_exact on x=0 and y=0
        - Neumann BC: ∂u/∂n = g on x=1 and y=1

    Args:
        grid_resolution: Number of cells in each dimension
        polynomial_degree: Polynomial degree for Lagrange elements

    Returns:
        Tuple of (solution_values, node_positions)
        - solution_values: shape (N,) array of u at each node
        - node_positions: shape (N, 2) array of (x, y) coordinates
    """
    # Create unit square mesh
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

    # Bilinear form for Helmholtz: a(u,v) = ∫(∇u·∇v - k₀²·u·v) dx
    bilinear_form = (
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - WAVE_NUMBER_K0**2 * ufl.inner(u, v) * ufl.dx
    )

    # Create exact solution in a higher-order space for accuracy
    function_space_exact = fem.functionspace(domain, ("Lagrange", polynomial_degree + 3))
    u_exact = fem.Function(function_space_exact, name="u_exact")

    # Interpolate exact plane wave solution (real part: cos)
    def exact_solution(x):
        phase = WAVE_NUMBER_K0 * (WAVE_DIRECTION_X * x[0] + WAVE_DIRECTION_Y * x[1])
        return AMPLITUDE_A * np.cos(phase)

    u_exact.interpolate(exact_solution)

    # Neumann BC: g = -∇u_exact · n (computed via UFL)
    # The weak form gives: ∫(∇u·∇v - k²uv)dx = ∫(∇u·n)v ds
    # We have g = ∇u_exact · n, so RHS contribution is ∫g·v ds
    x = ufl.SpatialCoordinate(domain)
    n = ufl.FacetNormal(domain)

    # Compute gradient of exact solution using UFL
    # For u = A·cos(φ), ∇u = -A·k₀·sin(φ)·(cos(θ), sin(θ))
    # But we use the interpolated u_exact directly
    g = ufl.dot(ufl.grad(u_exact), n)

    # Linear form: L(v) = ∫g·v ds (Neumann contribution on boundary)
    # Note: The FEniCSx example uses L = -inner(g, v) * ds where g = -dot(n, grad(u_exact))
    # This simplifies to L = inner(dot(grad(u_exact), n), v) * ds
    linear_form = ufl.inner(g, v) * ufl.ds

    # Dirichlet BC on x=0 and y=0
    def dirichlet_boundary(x):
        return np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))

    dirichlet_dofs = fem.locate_dofs_geometrical(function_space, dirichlet_boundary)

    # Interpolate exact solution for Dirichlet BC values
    u_bc = fem.Function(function_space)
    u_bc.interpolate(exact_solution)

    boundary_condition = fem.dirichletbc(u_bc, dirichlet_dofs)

    # Solve the linear problem
    problem = LinearProblem(
        bilinear_form,
        linear_form,
        bcs=[boundary_condition],
        petsc_options_prefix="helmholtz_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_error_if_not_converged": True,
        },
    )
    solution_function = problem.solve()

    # Extract solution values and node positions
    solution_values = solution_function.x.array.copy()

    # Get node coordinates from the function space geometry
    node_positions = function_space.tabulate_dof_coordinates()[:, :2]

    # Round coordinates to eliminate floating-point noise
    node_positions = np.round(node_positions, decimals=10)

    # Reorder to match standard grid ordering (column-major)
    x_coords = node_positions[:, 0]
    y_coords = node_positions[:, 1]
    sorted_indices = np.lexsort((y_coords, x_coords))

    solution_values = solution_values[sorted_indices]
    node_positions = node_positions[sorted_indices]

    return solution_values, node_positions


def solve(grid_resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unified solver interface for CLI.

    Args:
        grid_resolution: Number of cells in each dimension

    Returns:
        Tuple of (solution_values, node_positions)
        - solution_values: shape (N,) array of u at each node
        - node_positions: shape (N, 2) array of (x, y) coordinates
    """
    return solve_helmholtz_2d(grid_resolution=grid_resolution, polynomial_degree=1)
