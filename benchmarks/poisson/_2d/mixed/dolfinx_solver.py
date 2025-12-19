"""DOLFINx FEM solver for 2D Poisson equation with mixed boundary conditions.

Solves: -∇²u = f on [0,1]²

Boundary conditions:
    u(0, y) = 0           (left - Dirichlet, cold wall)
    u(1, y) = 0           (right - Dirichlet, cold wall)
    ∂u/∂y(x, 0) = 0       (bottom - Neumann, insulated)
    ∂u/∂y(x, 1) = 0       (top - Neumann, insulated)

Manufactured solution: u(x,y) = sin(πx)cos(πy)
Source term: f(x,y) = 2π²sin(πx)cos(πy)

Physical interpretation:
    Heat conduction in a plate with cold side walls (T=0) and
    insulated top/bottom edges, with internal heating.
"""

from typing import Tuple

import numpy as np
from mpi4py import MPI

import ufl
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import LinearProblem


def solve_poisson_2d_mixed_bc(
    grid_resolution: int = 32,
    polynomial_degree: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve 2D Poisson equation with mixed boundary conditions using DOLFINx.

    Solves: -∇²u = 2π²sin(πx)cos(πy) on [0,1]²
    with Dirichlet BCs (u=0) on left/right and natural Neumann BCs on top/bottom.

    Args:
        grid_resolution: Number of cells in each dimension
        polynomial_degree: Polynomial degree for Lagrange elements

    Returns:
        Tuple of (solution_values, node_positions)
        - solution_values: shape (N,) array of u at each node
        - node_positions: shape (N, 2) array of (x, y) coordinates
    """
    # Create unit square mesh with quadrilateral elements
    domain = mesh.create_unit_square(
        comm=MPI.COMM_WORLD,
        nx=grid_resolution,
        ny=grid_resolution,
        cell_type=mesh.CellType.quadrilateral,
    )

    # Create scalar Lagrange function space
    function_space = fem.functionspace(domain, ("Lagrange", polynomial_degree))

    # Define boundary conditions
    domain_dimension = domain.topology.dim
    domain.topology.create_connectivity(domain_dimension - 1, domain_dimension)

    # Locate boundary facets for left and right edges only
    # Top and bottom have natural (Neumann) BC - no explicit enforcement needed
    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    def right_boundary(x):
        return np.isclose(x[0], 1.0)

    # Get DOFs on Dirichlet boundaries
    left_dofs = fem.locate_dofs_geometrical(function_space, left_boundary)
    right_dofs = fem.locate_dofs_geometrical(function_space, right_boundary)

    # Create boundary conditions (homogeneous Dirichlet: u = 0)
    bc_left = fem.dirichletbc(
        value=default_scalar_type(0.0),
        dofs=left_dofs,
        V=function_space,
    )

    bc_right = fem.dirichletbc(
        value=default_scalar_type(0.0),
        dofs=right_dofs,
        V=function_space,
    )

    # Dirichlet BCs on left and right only
    # Top and bottom are natural (homogeneous Neumann) - automatically satisfied
    boundary_conditions = [bc_left, bc_right]

    # Define variational problem for Poisson equation
    # Trial and test functions
    u = ufl.TrialFunction(function_space)
    v = ufl.TestFunction(function_space)

    # Spatial coordinates for source term
    x = ufl.SpatialCoordinate(domain)

    # Source term: f(x,y) = 2π²sin(πx)cos(πy)
    source_term = 2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])

    # Bilinear form: a(u,v) = ∫∇u·∇v dx
    bilinear_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    # Linear form: L(v) = ∫f·v dx
    linear_form = ufl.inner(source_term, v) * ufl.dx

    # Solve the linear problem
    problem = LinearProblem(
        bilinear_form,
        linear_form,
        bcs=boundary_conditions,
        petsc_options_prefix="poisson_mixed_",
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

    # Round coordinates to eliminate floating-point noise
    node_positions = np.round(node_positions, decimals=10)

    # DOLFINx returns DOFs in its own ordering which doesn't match the standard
    # grid ordering (column-major, y varies first) used by Warp and analytical.
    # We need to reorder to match: sort by x first, then by y within each x.
    x_coords = node_positions[:, 0]
    y_coords = node_positions[:, 1]

    # Sort by x (primary) and y (secondary) to get column-major ordering
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
    return solve_poisson_2d_mixed_bc(grid_resolution=grid_resolution, polynomial_degree=1)
