"""DOLFINx FEM solver for 3D Laplace equation.

Solves: ∇²u = 0 on [0,1]³

Boundary conditions:
    u(x, y, 0) = 0              (bottom, z=0)
    u(x, y, 1) = sin(πx)sin(πy) (top, z=1)
    u(0, y, z) = 0              (left, x=0)
    u(1, y, z) = 0              (right, x=1)
    u(x, 0, z) = 0              (front, y=0)
    u(x, 1, z) = 0              (back, y=1)

Analytical solution: u(x,y,z) = sin(πx) · sin(πy) · sinh(πz) / sinh(π)
"""

from typing import Tuple

import numpy as np
from mpi4py import MPI

import ufl
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import LinearProblem


def solve_laplace_3d(
    grid_resolution: int = 32,
    polynomial_degree: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve 3D Laplace equation with non-homogeneous Dirichlet BCs using DOLFINx.

    Solves: ∇²u = 0 on [0,1]³
    with u(x,y,1) = sin(πx)sin(πy) on top face, u = 0 elsewhere.

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

    # Locate boundary facets for each face
    def bottom_boundary(x):
        return np.isclose(x[2], 0.0)

    def top_boundary(x):
        return np.isclose(x[2], 1.0)

    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    def right_boundary(x):
        return np.isclose(x[0], 1.0)

    def front_boundary(x):
        return np.isclose(x[1], 0.0)

    def back_boundary(x):
        return np.isclose(x[1], 1.0)

    # Get DOFs on each boundary
    bottom_dofs = fem.locate_dofs_geometrical(function_space, bottom_boundary)
    top_dofs = fem.locate_dofs_geometrical(function_space, top_boundary)
    left_dofs = fem.locate_dofs_geometrical(function_space, left_boundary)
    right_dofs = fem.locate_dofs_geometrical(function_space, right_boundary)
    front_dofs = fem.locate_dofs_geometrical(function_space, front_boundary)
    back_dofs = fem.locate_dofs_geometrical(function_space, back_boundary)

    # Create boundary conditions
    # Bottom, left, right, front, back: u = 0
    bc_bottom = fem.dirichletbc(
        value=default_scalar_type(0.0),
        dofs=bottom_dofs,
        V=function_space,
    )
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
    bc_front = fem.dirichletbc(
        value=default_scalar_type(0.0),
        dofs=front_dofs,
        V=function_space,
    )
    bc_back = fem.dirichletbc(
        value=default_scalar_type(0.0),
        dofs=back_dofs,
        V=function_space,
    )

    # Top: u = sin(πx)sin(πy) - need to interpolate this function
    def top_boundary_value(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

    top_bc_function = fem.Function(function_space)
    top_bc_function.interpolate(top_boundary_value)

    bc_top = fem.dirichletbc(
        value=top_bc_function,
        dofs=top_dofs,
    )

    boundary_conditions = [bc_bottom, bc_top, bc_left, bc_right, bc_front, bc_back]

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
        petsc_options_prefix="laplace_3d_",
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
    return solve_laplace_3d(grid_resolution=grid_resolution, polynomial_degree=1)
