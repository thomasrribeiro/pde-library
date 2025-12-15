"""DOLFINx FEM solver for 2D Poisson equation.

Solves: -∇²u = f  on [0,1]²
with homogeneous Dirichlet BC: u = 0 on boundary

Uses the manufactured solution approach where:
    f(x,y) = 2π²sin(πx)sin(πy)
    exact solution: u(x,y) = sin(πx)sin(πy)
"""

from typing import Tuple

import numpy as np
from mpi4py import MPI

import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem


def compute_manufactured_source_term(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute source term f(x,y) = 2π²sin(πx)sin(πy).

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates

    Returns:
        Array of source term values at each point
    """
    return 2.0 * np.pi**2 * np.sin(np.pi * x_coordinates) * np.sin(np.pi * y_coordinates)


def solve_poisson_2d(
    grid_resolution: int = 32,
    polynomial_degree: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve 2D Poisson equation with manufactured solution using DOLFINx.

    Solves: -∇²u = 2π²sin(πx)sin(πy) on [0,1]²
    with u = 0 on boundary.

    Args:
        grid_resolution: Number of cells in each dimension
        polynomial_degree: Polynomial degree for Lagrange elements

    Returns:
        Tuple of (solution_values, node_positions)
        - solution_values: shape (N,) array of u at each node
        - node_positions: shape (N, 2) array of (x, y) coordinates
    """
    # Create unit square mesh with triangular elements
    domain = mesh.create_unit_square(
        comm=MPI.COMM_WORLD,
        nx=grid_resolution,
        ny=grid_resolution,
        cell_type=mesh.CellType.quadrilateral,
    )

    # Create scalar Lagrange function space
    function_space = fem.functionspace(domain, ("Lagrange", polynomial_degree))

    # Define boundary condition: u = 0 on all boundaries
    # Find all boundary facets
    domain_dimension = domain.topology.dim
    domain.topology.create_connectivity(domain_dimension - 1, domain_dimension)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)

    # Get DOFs on boundary
    boundary_dofs = fem.locate_dofs_topological(
        V=function_space,
        entity_dim=domain_dimension - 1,
        entities=boundary_facets,
    )

    # Create homogeneous Dirichlet BC (u = 0)
    boundary_condition = fem.dirichletbc(
        value=np.float64(0.0),
        dofs=boundary_dofs,
        V=function_space,
    )

    # Define variational problem
    # Trial and test functions
    u = ufl.TrialFunction(function_space)
    v = ufl.TestFunction(function_space)

    # Source term using UFL spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    source_term = 2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Bilinear form: a(u,v) = ∫∇u·∇v dx
    bilinear_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    # Linear form: L(v) = ∫f·v dx
    linear_form = ufl.inner(source_term, v) * ufl.dx

    # Solve the linear problem
    problem = LinearProblem(
        bilinear_form,
        linear_form,
        bcs=[boundary_condition],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    solution_function = problem.solve()

    # Extract solution values and node positions
    solution_values = solution_function.x.array.copy()

    # Get node coordinates from the function space geometry
    node_positions = function_space.tabulate_dof_coordinates()[:, :2]

    # Round coordinates to eliminate floating-point noise (e.g., -3.47e-18 → 0.0)
    # This prevents scipy.griddata cubic interpolation from failing at boundaries
    node_positions = np.round(node_positions, decimals=10)

    # DOLFINx returns DOFs in its own ordering which doesn't match the standard
    # grid ordering (column-major, y varies first) used by Warp and analytical.
    # We need to reorder to match: sort by x first, then by y within each x.
    # This creates the column-major ordering expected by other solvers.
    x_coords = node_positions[:, 0]
    y_coords = node_positions[:, 1]

    # Sort by x (primary) and y (secondary) to get column-major ordering
    # Use lexsort which sorts by last key first, so we provide (y, x) to sort by x then y
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
    return solve_poisson_2d(grid_resolution=grid_resolution, polynomial_degree=1)
