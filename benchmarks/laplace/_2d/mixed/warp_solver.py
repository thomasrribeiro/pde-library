"""Warp FEM solver for 2D Laplace equation with mixed boundary conditions.

Solves: ∇²u = 0 on [0,1]²

Boundary conditions:
    u(x, 1) = sin(πx)        (top - Dirichlet, heated)
    u(0, y) = 0              (left - Dirichlet, cold wall)
    u(1, y) = 0              (right - Dirichlet, cold wall)
    ∂u/∂y(x, 0) = 0          (bottom - Neumann, insulating)

Physical interpretation:
    Heat source on top with sin(πx) profile, cold walls on the sides,
    and an insulated bottom that reflects heat back up.

Note on BC compatibility:
    sin(πx) = 0 at x=0 and x=1, so the top BC is perfectly compatible
    with the cold wall (u=0) Dirichlet BCs on the sides. No corner issues!

The Neumann (insulating) BC on bottom is natural in the weak form and
doesn't require explicit enforcement.
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
def dirichlet_boundary_value_function(position: wp.vec2):
    """Boundary value function for Dirichlet BCs.

    Applies to top (y=1), left (x=0), and right (x=1) boundaries.
    Bottom (y=0) has natural (Neumann) BC.

    Args:
        position: Point on the boundary

    Returns:
        Prescribed boundary value at this point
    """
    x = position[0]
    y = position[1]
    pi = 3.14159265358979323846

    # Top boundary (y ≈ 1): u = sin(πx)
    if y > 0.99:
        return wp.sin(pi * x)
    # Left boundary (x ≈ 0): u = 0
    if x < 0.01:
        return 0.0
    # Right boundary (x ≈ 1): u = 0
    if x > 0.99:
        return 0.0
    # Bottom boundary: return 0 but won't be used (natural BC)
    return 0.0


@fem.integrand
def dirichlet_boundary_projector_form(
    sample: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    """Bilinear form for boundary projection on Dirichlet boundaries.

    Applies to top (n_y = +1), left (n_x = -1), and right (n_x = +1).
    Excludes bottom (n_y = -1) which has Neumann BC.

    Uses max(normal[1], 0) to select top only (not bottom),
    plus |normal[0]| to select left and right.
    """
    normal = fem.normal(domain, sample)
    # Top: normal[1] = +1, Bottom: normal[1] = -1
    # Left: normal[0] = -1, Right: normal[0] = +1
    # We want top + left + right, excluding bottom
    is_dirichlet = wp.max(normal[1], 0.0) + wp.abs(normal[0])
    return is_dirichlet * u(sample) * v(sample)


@fem.integrand
def dirichlet_boundary_value_form(
    sample: fem.Sample,
    domain: fem.Domain,
    v: fem.Field,
    boundary_value_field: fem.Field,
):
    """Linear form for Dirichlet BC on top, left, and right boundaries."""
    normal = fem.normal(domain, sample)
    is_dirichlet = wp.max(normal[1], 0.0) + wp.abs(normal[0])
    return is_dirichlet * boundary_value_field(sample) * v(sample)


def solve_laplace_2d_mixed_bc(
    grid_resolution: int = 32,
    polynomial_degree: int = 1,
    quiet: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve 2D Laplace equation with mixed boundary conditions.

    Solves: ∇²u = 0 on [0,1]²
    with Dirichlet BCs on top/bottom and Neumann (natural) BCs on left/right.

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

    # Assemble stiffness matrix: ∫∇u·∇v dx
    stiffness_matrix = fem.integrate(
        laplace_bilinear_form,
        fields={"u": trial, "v": test},
        output_dtype=float,
    )

    # RHS vector starts as zero (no source term, homogeneous Neumann is natural)
    rhs_vector = wp.zeros(scalar_space.node_count(), dtype=float)

    # Apply Dirichlet BCs on top, left, and right boundaries
    # Bottom boundary has natural (Neumann) BC - homogeneous ∂u/∂y = 0 (insulating)
    boundary = fem.BoundarySides(geometry)
    boundary_trial = fem.make_trial(space=scalar_space, domain=boundary)
    boundary_test = fem.make_test(space=scalar_space, domain=boundary)

    # Create boundary value field
    boundary_value_field = fem.ImplicitField(
        domain=boundary,
        func=dirichlet_boundary_value_function,
    )

    # Assemble boundary projector (contributes on top, left, right - excludes bottom)
    boundary_projector = fem.integrate(
        dirichlet_boundary_projector_form,
        fields={
            "u": boundary_trial,
            "v": boundary_test,
        },
        assembly="nodal",
        output_dtype=float,
    )

    # Assemble boundary value RHS (contributes on top, left, right - excludes bottom)
    boundary_rhs = fem.integrate(
        dirichlet_boundary_value_form,
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
        - node_positions: shape (N, 2) array of (x, y) coordinates
    """
    import warp as wp
    wp.init()
    return solve_laplace_2d_mixed_bc(grid_resolution=grid_resolution, quiet=True)
