"""Warp FEM solver for 2D Poisson equation.

Solves: -∇²u = f  on [0,1]²
with homogeneous Dirichlet BC: u = 0 on boundary

Based on Warp's example_diffusion.py structure.
"""

import numpy as np
from typing import Tuple

import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils


@fem.integrand
def poisson_bilinear_form(
    sample: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    """Bilinear form for Poisson equation: (∇u, ∇v)"""
    return wp.dot(fem.grad(u, sample), fem.grad(v, sample))


@fem.integrand
def source_linear_form(
    sample: fem.Sample,
    v: fem.Field,
    source_field: fem.Field,
):
    """Linear form for source term: (f, v)"""
    return source_field(sample) * v(sample)


@fem.integrand
def boundary_projector_form(
    sample: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    """Bilinear form for boundary projection: (u, v) on boundary"""
    return u(sample) * v(sample)


@wp.func
def manufactured_source_function(position: wp.vec2):
    """Source term f(x,y) = 2π²sin(πx)sin(πy)"""
    x = position[0]
    y = position[1]
    pi = 3.14159265358979323846
    return 2.0 * pi * pi * wp.sin(pi * x) * wp.sin(pi * y)


def solve_poisson_2d(
    grid_resolution: int = 32,
    polynomial_degree: int = 1,
    quiet: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve 2D Poisson equation with manufactured solution.

    Solves: -∇²u = 2π²sin(πx)sin(πy) on [0,1]²
    with u = 0 on boundary.

    Args:
        grid_resolution: Number of cells in each dimension
        polynomial_degree: Polynomial degree for Lagrange elements
        quiet: If True, suppress solver output

    Returns:
        Tuple of (solution_values, node_positions)
        - solution_values: shape (N,) array of u at each node
        - node_positions: shape (N, 2) array of (x, y) coordinates
    """
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

    # Create source field using ImplicitField
    source_field = fem.ImplicitField(
        domain=domain,
        func=manufactured_source_function,
    )

    # Assemble stiffness matrix: ∫∇u·∇v dx
    stiffness_matrix = fem.integrate(
        poisson_bilinear_form,
        fields={"u": trial, "v": test},
        output_dtype=float,
    )

    # Assemble RHS vector: ∫f·v dx
    rhs_vector = fem.integrate(
        source_linear_form,
        fields={"v": test, "source_field": source_field},
        output_dtype=float,
    )

    # Apply homogeneous Dirichlet BC on all boundaries
    boundary = fem.BoundarySides(geometry)
    boundary_trial = fem.make_trial(space=scalar_space, domain=boundary)
    boundary_test = fem.make_test(space=scalar_space, domain=boundary)

    boundary_projector = fem.integrate(
        boundary_projector_form,
        fields={"u": boundary_trial, "v": boundary_test},
        assembly="nodal",
        output_dtype=float,
    )

    # Project linear system to enforce BC (homogeneous, so no BC rhs needed)
    fem.project_linear_system(stiffness_matrix, rhs_vector, boundary_projector)

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
        - node_positions: shape (N, 2) array of (x, y) coordinates
    """
    import warp as wp
    wp.init()
    return solve_poisson_2d(grid_resolution=grid_resolution, quiet=True)
