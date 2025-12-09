"""Warp FEM solver for 2D Helmholtz plane wave equation.

Solves: -∇²u - k₀²u = 0 on [0,1]²

Mixed boundary conditions (matching FEniCSx example from Ihlenburg's book):
- Dirichlet: u = u_exact on x=0 and y=0
- Neumann: ∂u/∂n = g on x=1 and y=1

Exact solution: u(x,y) = A·cos(k₀·(cos(θ)·x + sin(θ)·y))

The weak form with Neumann BC:
∫(∇u·∇v - k₀²uv)dx = ∫g·v ds  (on Neumann boundary)

Reference: Ihlenburg, "Finite Element Analysis of Acoustic Scattering" (p138-139)
"""

import numpy as np
from typing import Tuple

import warp as wp
import warp.fem as fem


# Wave parameters (matching FEniCSx example from Ihlenburg's book)
PI = 3.14159265358979323846
WAVE_NUMBER_K0 = 4.0 * PI  # k₀ = 4π (~12.57)
WAVE_NUMBER_SQUARED_K0 = WAVE_NUMBER_K0 * WAVE_NUMBER_K0
PROPAGATION_ANGLE_THETA = PI / 4.0  # 45 degrees
AMPLITUDE_A = 1.0

# Wave direction components (precomputed for efficiency)
WAVE_DIRECTION_X = 0.7071067811865476  # cos(π/4) = √2/2
WAVE_DIRECTION_Y = 0.7071067811865476  # sin(π/4) = √2/2


@wp.func
def compute_phase(x: float, y: float):
    """Compute the phase φ = k₀·(cos(θ)·x + sin(θ)·y)."""
    return WAVE_NUMBER_K0 * (WAVE_DIRECTION_X * x + WAVE_DIRECTION_Y * y)


@wp.func
def exact_solution_at_point(x: float, y: float):
    """Compute exact solution u = A·cos(φ) at a point."""
    phase = compute_phase(x, y)
    return AMPLITUDE_A * wp.cos(phase)


@wp.func
def neumann_flux_at_point(x: float, y: float, normal_x: float, normal_y: float):
    """Compute Neumann flux g = ∇u · n at a boundary point."""
    phase = compute_phase(x, y)
    direction_dot_normal = WAVE_DIRECTION_X * normal_x + WAVE_DIRECTION_Y * normal_y
    return -AMPLITUDE_A * WAVE_NUMBER_K0 * wp.sin(phase) * direction_dot_normal


@fem.integrand
def helmholtz_bilinear_form(
    sample: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    """Bilinear form for Helmholtz: (∇u, ∇v) - k₀²(u, v)."""
    grad_term = wp.dot(fem.grad(u, sample), fem.grad(v, sample))
    mass_term = WAVE_NUMBER_SQUARED_K0 * u(sample) * v(sample)
    return grad_term - mass_term


@fem.integrand
def neumann_boundary_form(
    sample: fem.Sample,
    domain: fem.Domain,
    v: fem.Field,
):
    """Linear form for Neumann BC: ∫g·v ds on x=1 and y=1 boundaries."""
    # Get position and normal
    pos = fem.position(domain, sample)
    nor = fem.normal(domain, sample)

    # Only apply on Neumann boundaries (x=1 or y=1)
    # x=1 has normal.x > 0, y=1 has normal.y > 0
    is_neumann = wp.max(nor[0], nor[1])

    # Compute Neumann flux g = ∇u · n
    flux = neumann_flux_at_point(pos[0], pos[1], nor[0], nor[1])

    return is_neumann * flux * v(sample)


@fem.integrand
def dirichlet_projector_form(
    sample: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    """Boundary projection for Dirichlet BC on x=0 and y=0.

    Uses normal vector to select boundaries:
    - x=0 has normal = (-1, 0), so normal.x < 0
    - y=0 has normal = (0, -1), so normal.y < 0
    """
    nor = fem.normal(domain, sample)

    # Dirichlet on x=0 (normal.x < 0) or y=0 (normal.y < 0)
    is_dirichlet = wp.max(-nor[0], -nor[1])

    return is_dirichlet * u(sample) * v(sample)


@fem.integrand
def dirichlet_value_form(
    sample: fem.Sample,
    domain: fem.Domain,
    v: fem.Field,
):
    """Linear form for Dirichlet values on x=0 and y=0 boundaries."""
    pos = fem.position(domain, sample)
    nor = fem.normal(domain, sample)

    # Dirichlet on x=0 (normal.x < 0) or y=0 (normal.y < 0)
    is_dirichlet = wp.max(-nor[0], -nor[1])

    # Exact solution value
    u_exact = exact_solution_at_point(pos[0], pos[1])

    return is_dirichlet * u_exact * v(sample)


def solve_helmholtz_2d(
    grid_resolution: int = 64,
    polynomial_degree: int = 1,
    quiet: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve 2D Helmholtz plane wave equation with mixed BCs.

    Solves: -∇²u - k₀²u = 0 on [0,1]²
    with:
        - Dirichlet BC: u = u_exact on x=0 and y=0
        - Neumann BC: ∂u/∂n = g on x=1 and y=1

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

    # Volume domain for integration
    domain = fem.Cells(geometry)
    trial = fem.make_trial(space=scalar_space, domain=domain)
    test = fem.make_test(space=scalar_space, domain=domain)

    # Step 1: Assemble Helmholtz matrix
    helmholtz_matrix = fem.integrate(
        helmholtz_bilinear_form,
        fields={"u": trial, "v": test},
        output_dtype=float,
    )

    # Step 2: Assemble Neumann BC contribution to RHS
    boundary = fem.BoundarySides(geometry)
    boundary_test = fem.make_test(space=scalar_space, domain=boundary)

    rhs_vector = fem.integrate(
        neumann_boundary_form,
        fields={"v": boundary_test},
        output_dtype=float,
    )

    # Step 3: Apply Dirichlet BC on x=0 and y=0
    boundary_trial = fem.make_trial(space=scalar_space, domain=boundary)

    # Selective Dirichlet projector (uses normal to select boundaries)
    dirichlet_projector = fem.integrate(
        dirichlet_projector_form,
        fields={"u": boundary_trial, "v": boundary_test},
        assembly="nodal",
        output_dtype=float,
    )

    # Dirichlet BC RHS
    dirichlet_rhs = fem.integrate(
        dirichlet_value_form,
        fields={"v": boundary_test},
        assembly="nodal",
        output_dtype=float,
    )

    # Project linear system to enforce Dirichlet BC
    fem.project_linear_system(helmholtz_matrix, rhs_vector, dirichlet_projector, dirichlet_rhs)

    # Solve using BiCGSTAB (faster than GMRES for this indefinite system)
    from warp.optim.linear import bicgstab

    solution = wp.zeros_like(rhs_vector)

    # Use BiCGSTAB for indefinite systems - faster than GMRES
    final_iter, residual, atol = bicgstab(
        A=helmholtz_matrix,
        b=rhs_vector,
        x=solution,
        tol=1e-6,
        maxiter=5000,
        check_every=50 if not quiet else 0,
    )

    if not quiet:
        print(f"BiCGSTAB: Converged in {final_iter} iterations, residual = {residual:.6e}")

    # Extract node positions
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
    wp.init()
    return solve_helmholtz_2d(grid_resolution=grid_resolution, quiet=True)
