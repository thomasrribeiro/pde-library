"""DOLFINx FEM solver for 3D Laplace equation with mixed boundary conditions.

Solves: ∇²u = 0 on [0,1]³

Boundary conditions:
    u(x, y, 1) = sin(πx)sin(πy)  (top/ceiling - Dirichlet, heated)
    u(0, y, z) = 0               (x=0 wall - Dirichlet, cold)
    u(1, y, z) = 0               (x=1 wall - Dirichlet, cold)
    u(x, 0, z) = 0               (y=0 wall - Dirichlet, cold)
    u(x, 1, z) = 0               (y=1 wall - Dirichlet, cold)
    ∂u/∂z(x, y, 0) = 0           (bottom/floor - Neumann, insulated)

Analytical solution: u(x,y,z) = sin(πx) · sin(πy) · cosh(√2·π·z) / cosh(√2·π)

Physical interpretation:
    Heat conduction in a room with heated ceiling (sin pattern), cold walls,
    and an insulated floor. No internal heat sources (Laplace, not Poisson).
"""

from typing import Tuple

import numpy as np
from mpi4py import MPI

import ufl
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import LinearProblem


def solve_laplace_3d_mixed_bc(
    grid_resolution: int = 32,
    polynomial_degree: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve 3D Laplace equation with mixed boundary conditions using DOLFINx.

    Solves: ∇²u = 0 on [0,1]³
    with:
        - Dirichlet BC u = sin(πx)sin(πy) on top (z=1)
        - Dirichlet BC u = 0 on 4 vertical walls
        - Neumann BC ∂u/∂n = 0 on bottom (z=0) - natural, no explicit enforcement

    Args:
        grid_resolution: Number of cells in each dimension
        polynomial_degree: Polynomial degree for Lagrange elements

    Returns:
        Tuple of (solution_values, node_positions)
        - solution_values: shape (N,) array of u at each node
        - node_positions: shape (N, 3) array of (x, y, z) coordinates
    """
    # Create unit cube mesh with hexahedral elements
    domain = mesh.create_box(
        comm=MPI.COMM_WORLD,
        points=[
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        ],
        n=[grid_resolution, grid_resolution, grid_resolution],
        cell_type=mesh.CellType.hexahedron,
    )

    # Create scalar Lagrange function space
    function_space = fem.functionspace(domain, ("Lagrange", polynomial_degree))

    # Define boundary conditions
    domain_dimension = domain.topology.dim
    domain.topology.create_connectivity(domain_dimension - 1, domain_dimension)

    # Locate boundary facets - note: bottom (z=0) is NOT included (Neumann)
    def top_boundary(x):
        return np.isclose(x[2], 1.0)

    def x_walls(x):
        return np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))

    def y_walls(x):
        return np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))

    # Get DOFs on Dirichlet boundaries (top + 4 walls, NOT bottom)
    top_dofs = fem.locate_dofs_geometrical(function_space, top_boundary)
    x_wall_dofs = fem.locate_dofs_geometrical(function_space, x_walls)
    y_wall_dofs = fem.locate_dofs_geometrical(function_space, y_walls)

    # Create boundary conditions
    # Top: u = sin(πx)sin(πy) - need to interpolate this function
    def top_boundary_value(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

    top_bc_function = fem.Function(function_space)
    top_bc_function.interpolate(top_boundary_value)

    bc_top = fem.dirichletbc(
        value=top_bc_function,
        dofs=top_dofs,
    )

    # 4 walls: homogeneous Dirichlet (u = 0)
    bc_x_walls = fem.dirichletbc(
        value=default_scalar_type(0.0),
        dofs=x_wall_dofs,
        V=function_space,
    )

    bc_y_walls = fem.dirichletbc(
        value=default_scalar_type(0.0),
        dofs=y_wall_dofs,
        V=function_space,
    )

    # Bottom (z=0) has natural Neumann BC - automatically satisfied, no explicit BC
    boundary_conditions = [bc_top, bc_x_walls, bc_y_walls]

    # Define variational problem for Laplace equation
    # Trial and test functions
    u = ufl.TrialFunction(function_space)
    v = ufl.TestFunction(function_space)

    # Bilinear form: a(u,v) = ∫∇u·∇v dx
    bilinear_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    # Linear form: L(v) = 0 (no source term for Laplace equation)
    linear_form = fem.Constant(domain, default_scalar_type(0.0)) * v * ufl.dx

    # Solve the linear problem
    problem = LinearProblem(
        bilinear_form,
        linear_form,
        bcs=boundary_conditions,
        petsc_options_prefix="laplace_3d_mixed_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    solution_function = problem.solve()

    # Extract solution values and node positions
    solution_values = solution_function.x.array.copy()

    # Get node coordinates from the function space geometry (all 3 dimensions)
    node_positions = function_space.tabulate_dof_coordinates()

    # Round coordinates to eliminate floating-point noise
    node_positions = np.round(node_positions, decimals=10)

    # DOLFINx returns DOFs in its own ordering which doesn't match the standard
    # grid ordering (column-major) used by Warp and analytical.
    # Sort by x (primary), then y (secondary), then z (tertiary) for column-major ordering
    x_coords = node_positions[:, 0]
    y_coords = node_positions[:, 1]
    z_coords = node_positions[:, 2]

    # lexsort sorts by last key first, so provide (z, y, x) to sort by x, then y, then z
    sorted_indices = np.lexsort((z_coords, y_coords, x_coords))

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
        - node_positions: shape (N, 3) array of (x, y, z) coordinates
    """
    return solve_laplace_3d_mixed_bc(grid_resolution=grid_resolution, polynomial_degree=1)
