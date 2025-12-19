"""Warp FEM solver for 3D Poisson equation with mixed boundary conditions.

Solves: -∇²u = f on [0,1]³

Boundary conditions:
    u(0, y, z) = 0        (x=0 wall - Dirichlet, cold)
    u(1, y, z) = 0        (x=1 wall - Dirichlet, cold)
    u(x, 0, z) = 0        (y=0 wall - Dirichlet, cold)
    u(x, 1, z) = 0        (y=1 wall - Dirichlet, cold)
    ∂u/∂z(x, y, 0) = 0    (bottom - Neumann, insulated floor)
    ∂u/∂z(x, y, 1) = 0    (top - Neumann, insulated ceiling)

Manufactured solution: u(x,y,z) = sin(πx)sin(πy)cos(πz)
Source term: f(x,y,z) = 3π²sin(πx)sin(πy)cos(πz)

Physical interpretation:
    Heat conduction in a block with cold vertical walls (T=0) and
    insulated ceiling/floor, with internal heating.
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


@wp.func
def manufactured_source_function(position: wp.vec3):
    """Source term f(x,y,z) = 3π²sin(πx)sin(πy)cos(πz)"""
    x = position[0]
    y = position[1]
    z = position[2]
    pi = 3.14159265358979323846
    return 3.0 * pi * pi * wp.sin(pi * x) * wp.sin(pi * y) * wp.cos(pi * z)


@fem.integrand
def dirichlet_boundary_projector_form(
    sample: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    """Bilinear form for boundary projection on Dirichlet boundaries only.

    Applies to the 4 vertical walls (x=0, x=1, y=0, y=1).
    Excludes top (z=1) and bottom (z=0) which have Neumann BCs.

    Uses |normal[0]| + |normal[1]| to select x and y walls only.
    """
    normal = fem.normal(domain, sample)
    # x=0 wall: normal[0] = -1, x=1 wall: normal[0] = +1
    # y=0 wall: normal[1] = -1, y=1 wall: normal[1] = +1
    # z=0 floor: normal[2] = -1, z=1 ceiling: normal[2] = +1
    # We want x and y walls only (Dirichlet), excluding z faces (Neumann)
    is_dirichlet = wp.abs(normal[0]) + wp.abs(normal[1])
    return is_dirichlet * u(sample) * v(sample)


def solve_poisson_3d_mixed_bc(
    grid_resolution: int = 32,
    polynomial_degree: int = 1,
    quiet: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve 3D Poisson equation with mixed boundary conditions.

    Solves: -∇²u = 3π²sin(πx)sin(πy)cos(πz) on [0,1]³
    with Dirichlet BCs (u=0) on 4 vertical walls and Neumann BCs (∂u/∂n=0) on top/bottom.

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

    # Apply Dirichlet BCs on 4 vertical walls only
    # Top and bottom faces have natural (Neumann) BC - homogeneous ∂u/∂z = 0
    boundary = fem.BoundarySides(geometry)
    boundary_trial = fem.make_trial(space=scalar_space, domain=boundary)
    boundary_test = fem.make_test(space=scalar_space, domain=boundary)

    # Assemble boundary projector (contributes on x and y walls - excludes z faces)
    boundary_projector = fem.integrate(
        dirichlet_boundary_projector_form,
        fields={
            "u": boundary_trial,
            "v": boundary_test,
        },
        assembly="nodal",
        output_dtype=float,
    )

    # Project linear system to enforce homogeneous Dirichlet BC (u=0) on walls
    # No boundary_rhs needed since BCs are homogeneous
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
        - node_positions: shape (N, 3) array of (x, y, z) coordinates
    """
    import warp as wp
    wp.init()
    return solve_poisson_3d_mixed_bc(grid_resolution=grid_resolution, quiet=True)
