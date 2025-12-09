"""Analytical solution for 2D heat/diffusion equation benchmark.

Heat equation: u_t = κ∇²u  on [0,1]²

Initial condition: Gaussian blob at center (0.5, 0.5)
    u₀(x,y) = A·exp(-((x-0.5)² + (y-0.5)²) / σ²)

    - Moderate width (σ = 0.1) for good numerical resolution
    - Centered heat source that diffuses outward

Boundary condition: u = 0 on all boundaries (homogeneous Dirichlet)

Analytical solution: Double Fourier sine series
    u(x,y,t) = Σₘ Σₙ Bₘₙ exp(-π²κ(m² + n²)t) sin(mπx) sin(nπy)

    where Bₘₙ = 4 ∫∫ u₀(x,y) sin(mπx) sin(nπy) dx dy

The heat diffuses outward from the center and is absorbed at the boundaries.

Default parameters:
    κ (diffusivity) = 0.01  (moderate diffusivity)
    σ (Gaussian width) = 0.1 (well-resolved at typical grid resolutions)
    A (amplitude) = 1.0
    T (final time) = 0.5 (enough time to see significant spreading)
"""

import numpy as np
from typing import Tuple


# Gaussian initial condition parameters
GAUSSIAN_CENTER_X = 0.5
GAUSSIAN_CENTER_Y = 0.5
GAUSSIAN_WIDTH_SIGMA = 0.1  # Moderate width (well-resolved at typical resolutions)
GAUSSIAN_AMPLITUDE = 1.0

# Diffusion parameters
DEFAULT_DIFFUSIVITY = 0.01  # Moderate diffusivity

# Time parameters
DEFAULT_FINAL_TIME = 0.5  # Enough time to see spreading
DEFAULT_NUM_OUTPUT_STEPS = 51  # More frames for smooth animation

# Fourier series truncation
# With σ=0.1, we need fewer modes since Gaussian is smoother
NUM_FOURIER_MODES = 50


def compute_initial_condition(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute the initial condition: Gaussian blob at center.

    u₀(x,y) = A·exp(-((x-x₀)² + (y-y₀)²) / σ²)

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates

    Returns:
        Array of initial condition values at each point
    """
    dx = x_coordinates - GAUSSIAN_CENTER_X
    dy = y_coordinates - GAUSSIAN_CENTER_Y
    r_squared = dx**2 + dy**2
    return GAUSSIAN_AMPLITUDE * np.exp(-r_squared / (GAUSSIAN_WIDTH_SIGMA**2))


def precompute_fourier_coefficients(
    num_modes: int = NUM_FOURIER_MODES,
) -> np.ndarray:
    """Precompute all Fourier coefficients up to num_modes (vectorized).

    Bₘₙ = 4 ∫₀¹ ∫₀¹ u₀(x,y) sin(mπx) sin(nπy) dx dy

    Uses vectorized computation for speed.

    Args:
        num_modes: Maximum mode number in each direction

    Returns:
        2D array of shape (num_modes+1, num_modes+1) with coefficients
        Index [m, n] gives coefficient Bₘₙ (indices 0 are unused)
    """
    # Use numerical integration with moderate grid (vectorized)
    num_quadrature_points = 200
    x_quad = np.linspace(0, 1, num_quadrature_points)
    y_quad = np.linspace(0, 1, num_quadrature_points)
    dx = x_quad[1] - x_quad[0]
    dy = y_quad[1] - y_quad[0]

    x_grid, y_grid = np.meshgrid(x_quad, y_quad, indexing='ij')

    # Compute initial condition once
    u0 = compute_initial_condition(x_grid, y_grid)

    # Precompute all sine values for efficiency
    # sin_x[m, i] = sin(m * pi * x[i]) for m in 1..num_modes
    m_values = np.arange(1, num_modes + 1)
    n_values = np.arange(1, num_modes + 1)

    # Shape: (num_modes, num_quad_points)
    sin_mx_all = np.sin(np.outer(m_values, np.pi * x_quad))
    sin_ny_all = np.sin(np.outer(n_values, np.pi * y_quad))

    # Compute all coefficients at once using einsum
    # For each (m, n), we need: sum over (i, j) of u0[i,j] * sin_mx[m,i] * sin_ny[n,j]
    # u0 has shape (nx, ny)
    # sin_mx_all has shape (num_modes, nx)
    # sin_ny_all has shape (num_modes, ny)

    # First compute: temp[m, j] = sum_i (u0[i, j] * sin_mx[m, i])
    temp = np.einsum('mi,ij->mj', sin_mx_all, u0)

    # Then compute: coeff[m, n] = sum_j (temp[m, j] * sin_ny[n, j])
    coefficients_inner = np.einsum('mj,nj->mn', temp, sin_ny_all)

    # Scale by 4 * dx * dy
    coefficients_inner *= 4.0 * dx * dy

    # Create full array with zero indices
    coefficients = np.zeros((num_modes + 1, num_modes + 1))
    coefficients[1:, 1:] = coefficients_inner

    return coefficients


# Precompute coefficients at module load time
_FOURIER_COEFFICIENTS = None


def get_fourier_coefficients() -> np.ndarray:
    """Get precomputed Fourier coefficients (lazy initialization)."""
    global _FOURIER_COEFFICIENTS
    if _FOURIER_COEFFICIENTS is None:
        _FOURIER_COEFFICIENTS = precompute_fourier_coefficients()
    return _FOURIER_COEFFICIENTS


def compute_analytical_solution(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    time: float,
    diffusivity: float = DEFAULT_DIFFUSIVITY,
    num_modes: int = NUM_FOURIER_MODES,
) -> np.ndarray:
    """Compute the exact Fourier series solution.

    u(x,y,t) = Σₘ Σₙ Bₘₙ exp(-π²κ(m² + n²)t) sin(mπx) sin(nπy)

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates
        time: Time at which to evaluate the solution
        diffusivity: Diffusivity coefficient κ
        num_modes: Number of Fourier modes to include

    Returns:
        Array of solution values at each point
    """
    coefficients = get_fourier_coefficients()

    # Initialize solution
    solution = np.zeros_like(x_coordinates)

    # Sum over all modes
    pi_squared = np.pi**2
    for m in range(1, num_modes + 1):
        sin_mx = np.sin(m * np.pi * x_coordinates)
        m_squared = m * m

        for n in range(1, num_modes + 1):
            b_mn = coefficients[m, n]
            if np.abs(b_mn) < 1e-15:
                continue  # Skip negligible terms

            n_squared = n * n
            decay = np.exp(-pi_squared * diffusivity * (m_squared + n_squared) * time)
            sin_ny = np.sin(n * np.pi * y_coordinates)

            solution += b_mn * decay * sin_mx * sin_ny

    return solution


def compute_analytical_solution_at_points(
    points: np.ndarray,
    time: float,
    diffusivity: float = DEFAULT_DIFFUSIVITY,
) -> np.ndarray:
    """Compute the exact solution at an array of 2D points.

    Args:
        points: Array of shape (N, 2) containing (x, y) coordinates
        time: Time at which to evaluate the solution
        diffusivity: Diffusivity coefficient κ

    Returns:
        Array of shape (N,) with solution values
    """
    x_coordinates = points[:, 0]
    y_coordinates = points[:, 1]
    return compute_analytical_solution(x_coordinates, y_coordinates, time, diffusivity)


def solve(grid_resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unified solver interface for CLI - generates analytical solution on a grid.

    Evaluates the analytical Fourier series solution at multiple time points.

    Args:
        grid_resolution: Number of cells in each dimension

    Returns:
        Tuple of (solution_values, node_positions, time_values)
        - solution_values: shape (num_time_steps, num_nodes) array of u at each node over time
        - node_positions: shape (num_nodes, 2) array of (x, y) coordinates
        - time_values: shape (num_time_steps,) array of time values
    """
    # Generate node positions matching linear FEM (nodes at cell corners)
    nodes_per_dimension = grid_resolution + 1
    x_values = np.linspace(0.0, 1.0, nodes_per_dimension)
    y_values = np.linspace(0.0, 1.0, nodes_per_dimension)

    # Create meshgrid and flatten to get all node positions
    # Use indexing='ij' with default C-order ravel to match Warp/DOLFINx ordering
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing='ij')
    node_positions = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    # Generate time values from t=0 to T
    time_values = np.linspace(0.0, DEFAULT_FINAL_TIME, DEFAULT_NUM_OUTPUT_STEPS)

    # Evaluate analytical solution at all nodes for each time point
    num_nodes = node_positions.shape[0]
    solution_values = np.zeros((DEFAULT_NUM_OUTPUT_STEPS, num_nodes))

    for time_index, time in enumerate(time_values):
        if time == 0.0:
            # Use initial condition directly (more accurate than Fourier series at t=0)
            solution_values[time_index, :] = compute_initial_condition(
                node_positions[:, 0], node_positions[:, 1]
            )
        else:
            solution_values[time_index, :] = compute_analytical_solution_at_points(
                node_positions,
                time=time,
                diffusivity=DEFAULT_DIFFUSIVITY,
            )

    return solution_values, node_positions, time_values
