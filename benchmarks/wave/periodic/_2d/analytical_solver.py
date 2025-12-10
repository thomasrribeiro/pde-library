"""Analytical solution for 2D wave equation benchmark (manufactured solution).

Wave equation: ∂²u/∂t² = c²∇²u  on [0,1]² with periodic boundaries

Manufactured solution: Standing wave (separable in space and time)
    u(x,y,t) = cos(2πmx)·cos(2πny)·cos(ωt)

where ω = c·2π√(m² + n²) satisfies the dispersion relation.

This is an exact solution to the homogeneous wave equation with periodic BCs.
No source term is needed - the initial conditions alone determine the solution.

Initial conditions:
    u(x,y,0) = cos(2πmx)·cos(2πny)
    ∂u/∂t(x,y,0) = 0

The standing wave oscillates in time without changing its spatial pattern.

Default parameters:
    c = 1.0         (wave speed, normalized)
    m = 2, n = 2    (mode numbers - 2 wavelengths in each direction)
    ω = 2π√8 ≈ 17.77  (angular frequency from dispersion relation)
    T = 1.0         (final time - ~2.8 oscillation periods)
"""

import numpy as np
from typing import Tuple


# Wave speed (normalized)
WAVE_SPEED = 1.0

# Mode numbers (number of wavelengths in each direction)
MODE_NUMBER_X = 2
MODE_NUMBER_Y = 2

# Angular frequency from dispersion relation: ω = c·2π√(m² + n²)
ANGULAR_FREQUENCY = WAVE_SPEED * 2.0 * np.pi * np.sqrt(
    MODE_NUMBER_X**2 + MODE_NUMBER_Y**2
)

# Time parameters
DEFAULT_FINAL_TIME = 1.0
DEFAULT_NUM_OUTPUT_STEPS = 101


def compute_initial_displacement(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute the initial displacement: standing wave at t=0.

    u(x,y,0) = cos(2πmx)·cos(2πny)

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates

    Returns:
        Array of initial displacement values at each point
    """
    return (
        np.cos(2.0 * np.pi * MODE_NUMBER_X * x_coordinates) *
        np.cos(2.0 * np.pi * MODE_NUMBER_Y * y_coordinates)
    )


def compute_initial_velocity(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute the initial velocity: zero for standing wave.

    ∂u/∂t(x,y,0) = 0

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates

    Returns:
        Array of zeros (initial velocity is zero)
    """
    return np.zeros_like(x_coordinates)


def compute_analytical_solution(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    time: float,
) -> np.ndarray:
    """Compute the exact wave equation solution at given time.

    u(x,y,t) = cos(2πmx)·cos(2πny)·cos(ωt)

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates
        time: Time at which to evaluate the solution

    Returns:
        Array of solution values at each point
    """
    spatial_part = compute_initial_displacement(x_coordinates, y_coordinates)
    temporal_part = np.cos(ANGULAR_FREQUENCY * time)
    return spatial_part * temporal_part


def compute_analytical_velocity(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    time: float,
) -> np.ndarray:
    """Compute the exact velocity (∂u/∂t) at given time.

    ∂u/∂t = -ω·cos(2πmx)·cos(2πny)·sin(ωt)

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates
        time: Time at which to evaluate the velocity

    Returns:
        Array of velocity values at each point
    """
    spatial_part = compute_initial_displacement(x_coordinates, y_coordinates)
    temporal_part = -ANGULAR_FREQUENCY * np.sin(ANGULAR_FREQUENCY * time)
    return spatial_part * temporal_part


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

    Evaluates the analytical wave equation solution at multiple time points.

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
