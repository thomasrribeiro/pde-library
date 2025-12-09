"""Analytical solution for 2D linear advection equation benchmark.

Linear advection equation: ∂u/∂t + c⃗·∇u = 0  on [0,1]²

This is pure transport with no diffusion - the solution is simply
the initial condition translated by the velocity field.

Initial condition: Gaussian blob at (x₀, y₀)
    u₀(x,y) = A·exp(-((x-x₀)² + (y-y₀)²) / σ²)

Velocity field: Constant c⃗ = (cx, cy)

Analytical solution: u(x,y,t) = u₀(x - cx·t, y - cy·t)

The Gaussian blob moves diagonally across the domain without changing shape.

Boundary conditions: Inflow (Dirichlet) on left/bottom, outflow on right/top
    - Inflow: u = 0 (no incoming signal after initial blob passes)
    - Outflow: Natural BC (zero flux, solution exits freely)

Default parameters:
    c⃗ = (0.4, 0.4)  (diagonal transport)
    σ = 0.1         (Gaussian width)
    A = 1.0         (amplitude)
    (x₀, y₀) = (0.3, 0.3)  (initial center)
    T = 1.0         (final time - blob moves to (0.7, 0.7))
"""

import numpy as np
from typing import Tuple


# Velocity field (constant)
# Chosen so blob moves from (0.3, 0.3) to (0.7, 0.7) over T=1
VELOCITY_X = 0.4
VELOCITY_Y = 0.4

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
    """Compute the exact translated Gaussian solution.

    u(x,y,t) = u₀(x - cx·t, y - cy·t)

    For points outside [0,1]², the solution is zero (the blob has exited).

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates
        time: Time at which to evaluate the solution

    Returns:
        Array of solution values at each point
    """
    # Translate coordinates back by velocity * time
    x_origin = x_coordinates - VELOCITY_X * time
    y_origin = y_coordinates - VELOCITY_Y * time

    # Compute Gaussian at origin coordinates
    dx = x_origin - GAUSSIAN_CENTER_X
    dy = y_origin - GAUSSIAN_CENTER_Y
    r_squared = dx**2 + dy**2

    return GAUSSIAN_AMPLITUDE * np.exp(-r_squared / (GAUSSIAN_WIDTH_SIGMA**2))


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

    Evaluates the analytical translated Gaussian solution at multiple time points.

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
