"""Warp FEM solver for 3D Laplace equation with mixed boundary conditions.

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


@wp.func
def dirichlet_boundary_value_function(position: wp.vec3):
    """Boundary value function.

    Top (z=1): u = sin(πx)sin(πy)
    4 walls (x=0,1, y=0,1): u = 0
    Bottom (z=0): Neumann BC (not used here)

    Args:
        position: Point on the boundary

    Returns:
        Prescribed Dirichlet boundary value at this point
    """
    x = position[0]
    y = position[1]
    z = position[2]
    pi = 3.14159265358979323846

    # Top boundary (z ≈ 1): u = sin(πx)sin(πy)
    if z > 0.99:
        return wp.sin(pi * x) * wp.sin(pi * y)

    # All other Dirichlet boundaries (4 walls): u = 0
    return 0.0


@fem.integrand
def dirichlet_boundary_projector_form(
    sample: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    """Bilinear form for boundary projection on Dirichlet boundaries only.

    Applies to the 4 vertical walls (x=0, x=1, y=0, y=1) AND the top (z=1).
    Excludes bottom (z=0) which has Neumann BC.

    Selection logic using normals:
    - x=0 wall: normal = (-1, 0, 0)
    - x=1 wall: normal = (+1, 0, 0)
    - y=0 wall: normal = (0, -1, 0)
    - y=1 wall: normal = (0, +1, 0)
    - z=0 floor: normal = (0, 0, -1)  <- EXCLUDE (Neumann)
    - z=1 ceiling: normal = (0, 0, +1) <- INCLUDE (Dirichlet)

    We want: |normal[0]| + |normal[1]| + max(0, normal[2])
    This gives 1 for walls and top, but 0 for bottom.
    """
    normal = fem.normal(domain, sample)

    # Select Dirichlet boundaries: all except bottom (z=0)
    # |normal[0]| = 1 on x walls, 0 elsewhere
    # |normal[1]| = 1 on y walls, 0 elsewhere
    # max(0, normal[2]) = 1 on top (z=1), 0 on bottom (z=0)
    is_dirichlet = wp.abs(normal[0]) + wp.abs(normal[1]) + wp.max(0.0, normal[2])

    return is_dirichlet * u(sample) * v(sample)


@fem.integrand
def dirichlet_boundary_value_linear_form(
    sample: fem.Sample,
    domain: fem.Domain,
    v: fem.Field,
    boundary_value_field: fem.Field,
):
    """Linear form for non-homogeneous Dirichlet BC: (g, v) on Dirichlet boundary.

    Only contributes on Dirichlet boundaries (walls + top, not bottom).
    """
    normal = fem.normal(domain, sample)

    # Same selection as projector form
    is_dirichlet = wp.abs(normal[0]) + wp.abs(normal[1]) + wp.max(0.0, normal[2])

    return is_dirichlet * boundary_value_field(sample) * v(sample)


def solve_laplace_3d_mixed_bc(
    grid_resolution: int = 32,
    polynomial_degree: int = 1,
    quiet: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve 3D Laplace equation with mixed boundary conditions.

    Solves: ∇²u = 0 on [0,1]³
    with:
        - Dirichlet BC u = sin(πx)sin(πy) on top (z=1)
        - Dirichlet BC u = 0 on 4 vertical walls
        - Neumann BC ∂u/∂n = 0 on bottom (z=0)

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

    # Apply mixed Dirichlet BCs (top + 4 walls) - exclude bottom (Neumann)
    boundary = fem.BoundarySides(geometry)
    boundary_trial = fem.make_trial(space=scalar_space, domain=boundary)
    boundary_test = fem.make_test(space=scalar_space, domain=boundary)

    # Create boundary value field
    boundary_value_field = fem.ImplicitField(
        domain=boundary,
        func=dirichlet_boundary_value_function,
    )

    # Assemble boundary projector (contributes on top + walls, excludes bottom)
    boundary_projector = fem.integrate(
        dirichlet_boundary_projector_form,
        fields={
            "u": boundary_trial,
            "v": boundary_test,
        },
        assembly="nodal",
        output_dtype=float,
    )

    # Assemble boundary value RHS contribution (non-homogeneous on top)
    boundary_rhs = fem.integrate(
        dirichlet_boundary_value_linear_form,
        fields={
            "v": boundary_test,
            "boundary_value_field": boundary_value_field,
        },
        assembly="nodal",
        output_dtype=float,
    )

    # Project linear system to enforce Dirichlet BCs
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
    return solve_laplace_3d_mixed_bc(grid_resolution=grid_resolution, quiet=True)
