"""Analytical solution for 2D convection-diffusion equation benchmark.

Convection-diffusion equation: ∂u/∂t + c⃗·∇u = κ∇²u  on [0,1]²

This combines advection (transport) with diffusion (spreading).
The solution is a Gaussian blob that both moves and spreads over time.

For a Gaussian initial condition in an unbounded domain, the exact solution is:
    u(x,y,t) = A·σ²/(σ² + 4κt) · exp(-((x-x₀-cx·t)² + (y-y₀-cy·t)²)/(σ² + 4κt))

The Gaussian:
    - Translates with velocity c⃗ (advection)
    - Spreads with effective width σ_eff(t) = √(σ² + 4κt) (diffusion)
    - Decreases in amplitude to conserve total mass

This solution is accurate when the Gaussian stays well inside the domain
(doesn't interact significantly with boundaries).

Initial condition: Gaussian blob at (0.3, 0.3)
    u₀(x,y) = A·exp(-((x-x₀)² + (y-y₀)²) / σ²)

Velocity field: Constant c⃗ = (0.4, 0.4) (diagonal transport)

Default parameters:
    c⃗ = (0.4, 0.4)  (diagonal transport)
    κ = 0.01        (diffusivity - moderate spreading)
    σ = 0.1         (initial Gaussian width)
    A = 1.0         (initial amplitude)
    (x₀, y₀) = (0.3, 0.3)  (initial center)
    T = 1.0         (final time - blob moves to ~(0.7, 0.7) while spreading)
"""

import numpy as np
from typing import Tuple


# Velocity field (constant) - same as advection benchmark
VELOCITY_X = 0.4
VELOCITY_Y = 0.4

# Diffusivity
DIFFUSIVITY = 0.01

# Gaussian initial condition parameters
GAUSSIAN_CENTER_X = 0.3
GAUSSIAN_CENTER_Y = 0.3
GAUSSIAN_WIDTH_SIGMA = 0.1
GAUSSIAN_AMPLITUDE = 1.0

# Time parameters
DEFAULT_FINAL_TIME = 1.0
DEFAULT_NUM_OUTPUT_STEPS = 51


def compute_initial_condition(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute the initial condition: Gaussian blob.

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


def compute_analytical_solution(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    time: float,
) -> np.ndarray:
    """Compute the exact convection-diffusion solution.

    u(x,y,t) = A·σ²/(σ² + 4κt) · exp(-((x-x₀-cx·t)² + (y-y₀-cy·t)²)/(σ² + 4κt))

    The Gaussian translates by (cx·t, cy·t) and spreads with width √(σ² + 4κt).

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates
        time: Time at which to evaluate the solution

    Returns:
        Array of solution values at each point
    """
    sigma_squared = GAUSSIAN_WIDTH_SIGMA**2

    # Effective variance at time t (spreading due to diffusion)
    effective_variance = sigma_squared + 4.0 * DIFFUSIVITY * time

    # Amplitude decreases to conserve mass
    amplitude = GAUSSIAN_AMPLITUDE * sigma_squared / effective_variance

    # Center position at time t (advection)
    center_x = GAUSSIAN_CENTER_X + VELOCITY_X * time
    center_y = GAUSSIAN_CENTER_Y + VELOCITY_Y * time

    # Compute Gaussian at translated and spread coordinates
    dx = x_coordinates - center_x
    dy = y_coordinates - center_y
    r_squared = dx**2 + dy**2

    return amplitude * np.exp(-r_squared / effective_variance)


def compute_analytical_solution_at_points(
    points: np.ndarray,
    time: float,
) -> np.ndarray:
    """Compute the exact solution at an array of 2D points.

    Args:
        points: Array of shape (N, 2) containing (x, y) coordinates
        time: Time at which to evaluate the solution

    Returns:
        Array of shape (N,) with solution values
    """
    x_coordinates = points[:, 0]
    y_coordinates = points[:, 1]
    return compute_analytical_solution(x_coordinates, y_coordinates, time)


def solve(grid_resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unified solver interface for CLI - generates analytical solution on a grid.

    Evaluates the analytical convection-diffusion solution at multiple time points.

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
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing='ij')
    node_positions = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    # Generate time values from t=0 to T
    time_values = np.linspace(0.0, DEFAULT_FINAL_TIME, DEFAULT_NUM_OUTPUT_STEPS)

    # Evaluate analytical solution at all nodes for each time point
    num_nodes = node_positions.shape[0]
    solution_values = np.zeros((DEFAULT_NUM_OUTPUT_STEPS, num_nodes))

    for time_index, time in enumerate(time_values):
        solution_values[time_index, :] = compute_analytical_solution_at_points(
            node_positions, time
        )

    return solution_values, node_positions, time_values
