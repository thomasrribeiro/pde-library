"""Warp FEM solver for 3D Laplace equation.

Solves: ∇²u = 0 on [0,1]³

Boundary conditions:
    u(x, y, 0) = 0              (bottom, z=0)
    u(x, y, 1) = sin(πx)sin(πy) (top, z=1)
    u(0, y, z) = 0              (left, x=0)
    u(1, y, z) = 0              (right, x=1)
    u(x, 0, z) = 0              (front, y=0)
    u(x, 1, z) = 0              (back, y=1)

Analytical solution: u(x,y,z) = sin(πx) · sin(πy) · sinh(kz) / sinh(k)
where k = √2·π ≈ 4.443

Based on Warp's example_diffusion_3d.py structure.
"""

import numpy as np
from typing import Tuple

import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils


@fem.integrand
def laplace_bilinear_form(
    sample: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    """Bilinear form for Laplace equation: (∇u, ∇v)"""
    return wp.dot(fem.grad(u, sample), fem.grad(v, sample))


@fem.integrand
def boundary_projector_form(
    sample: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    """Bilinear form for boundary projection: (u, v) on boundary"""
    return u(sample) * v(sample)


@fem.integrand
def boundary_value_linear_form(
    sample: fem.Sample,
    v: fem.Field,
    boundary_value_field: fem.Field,
):
    """Linear form for non-homogeneous Dirichlet BC: (g, v) on boundary"""
    return boundary_value_field(sample) * v(sample)


@wp.func
def dirichlet_boundary_value_function(position: wp.vec3):
    """Boundary value function: sin(πx)sin(πy) on top (z=1), 0 elsewhere.

    Args:
        position: Point on the boundary

    Returns:
        Prescribed boundary value at this point
    """
    x = position[0]
    y = position[1]
    z = position[2]
    pi = 3.14159265358979323846

    # Top boundary (z ≈ 1): u = sin(πx)sin(πy)
    # All other boundaries: u = 0
    if z > 0.99:
        return wp.sin(pi * x) * wp.sin(pi * y)
    return 0.0


def solve_laplace_3d(
    grid_resolution: int = 32,
    polynomial_degree: int = 1,
    quiet: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve 3D Laplace equation with non-homogeneous Dirichlet BCs.

    Solves: ∇²u = 0 on [0,1]³
    with u(x,y,1) = sin(πx)sin(πy) on top face, u = 0 elsewhere.

    Args:
        grid_resolution: Number of cells in each dimension
        polynomial_degree: Polynomial degree for Lagrange elements
        quiet: If True, suppress solver output

    Returns:
        Tuple of (solution_values, node_positions)
        - solution_values: shape (N,) array of u at each node
        - node_positions: shape (N, 3) array of (x, y, z) coordinates
    """
    # Create 3D grid geometry on unit cube [0, 1]³
    geometry = fem.Grid3D(
        bounds_lo=wp.vec3(0.0, 0.0, 0.0),
        bounds_hi=wp.vec3(1.0, 1.0, 1.0),
        res=wp.vec3i(grid_resolution, grid_resolution, grid_resolution),
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

    # Assemble stiffness matrix: ∫∇u·∇v dx
    # (No source term for Laplace equation)
    stiffness_matrix = fem.integrate(
        laplace_bilinear_form,
        fields={"u": trial, "v": test},
        output_dtype=float,
    )

    # RHS vector starts as zero (no source term)
    rhs_vector = wp.zeros(scalar_space.node_count(), dtype=float)

    # Apply non-homogeneous Dirichlet BC on all boundary faces
    boundary = fem.BoundarySides(geometry)
    boundary_trial = fem.make_trial(space=scalar_space, domain=boundary)
    boundary_test = fem.make_test(space=scalar_space, domain=boundary)

    # Create boundary value field
    boundary_value_field = fem.ImplicitField(
        domain=boundary,
        func=dirichlet_boundary_value_function,
    )

    # Assemble boundary projector matrix
    boundary_projector = fem.integrate(
        boundary_projector_form,
        fields={"u": boundary_trial, "v": boundary_test},
        assembly="nodal",
        output_dtype=float,
    )

    # Assemble boundary value RHS contribution
    boundary_rhs = fem.integrate(
        boundary_value_linear_form,
        fields={"v": boundary_test, "boundary_value_field": boundary_value_field},
        assembly="nodal",
        output_dtype=float,
    )

    # Project linear system to enforce non-homogeneous Dirichlet BC
    fem.project_linear_system(stiffness_matrix, rhs_vector, boundary_projector, boundary_rhs)

    # Solve with Conjugate Gradient
    solution = wp.zeros_like(rhs_vector)
    fem_example_utils.bsr_cg(
        stiffness_matrix,
        b=rhs_vector,
        x=solution,
        quiet=quiet,
    )

    # Extract node positions
    scalar_field = scalar_space.make_field()
    scalar_field.dof_values = solution
    node_positions = scalar_space.node_positions().numpy()

    return solution.numpy(), node_positions


def solve(grid_resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unified solver interface for CLI.

    Args:
        grid_resolution: Number of cells in each dimension

    Returns:
        Tuple of (solution_values, node_positions)
        - solution_values: shape (N,) array of u at each node
        - node_positions: shape (N, 3) array of (x, y, z) coordinates
    """
    import warp as wp
    wp.init()
    return solve_laplace_3d(grid_resolution=grid_resolution, quiet=True)
