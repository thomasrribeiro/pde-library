"""Analytical solutions for 2D heat/diffusion equation benchmark.

Heat equation: u_t = κ∇²u  on [0,1]²

Initial condition: u₀(x,y) = sin(πx)sin(πy)
    - Peak value of 1.0 at center (0.5, 0.5)
    - Zero on all boundaries

Boundary condition: u = 0 on all boundaries (homogeneous Dirichlet)

Closed-form solution: u(x,y,t) = exp(-2π²κt) sin(πx)sin(πy)
    - The solution decays exponentially in time
    - Higher diffusivity = faster decay
    - At t→∞, the solution approaches 0 everywhere

This tests time-stepping schemes, parabolic behavior, and stability limits.

Default parameters:
    κ (diffusivity) = 0.1  (high diffusivity for dramatic decay)
    T (final time) = 1.0   (long enough to see significant decay)
"""

import numpy as np
from typing import Tuple


# Default diffusivity coefficient (high value for dramatic diffusion)
DEFAULT_DIFFUSIVITY = 0.1

# Default final time (long enough to see significant decay)
DEFAULT_FINAL_TIME = 1.0

# Default number of output time steps
DEFAULT_NUM_OUTPUT_STEPS = 11  # Includes t=0


def compute_initial_condition(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute the initial condition u₀(x,y) = sin(πx)sin(πy).

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates

    Returns:
        Array of initial condition values at each point
    """
    return np.sin(np.pi * x_coordinates) * np.sin(np.pi * y_coordinates)


def compute_analytical_solution(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    time: float = DEFAULT_FINAL_TIME,
    diffusivity: float = DEFAULT_DIFFUSIVITY,
) -> np.ndarray:
    """Compute the exact solution u(x,y,t) = exp(-2π²κt) sin(πx)sin(πy).

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates
        time: Time at which to evaluate the solution
        diffusivity: Diffusivity coefficient κ

    Returns:
        Array of solution values at each point
    """
    decay_factor = np.exp(-2.0 * np.pi**2 * diffusivity * time)
    spatial_part = np.sin(np.pi * x_coordinates) * np.sin(np.pi * y_coordinates)
    return decay_factor * spatial_part


def compute_analytical_solution_at_points(
    points: np.ndarray,
    time: float = DEFAULT_FINAL_TIME,
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

    Evaluates the analytical solution at multiple time points from t=0 to T
    with default diffusivity κ=0.01.

    Args:
        grid_resolution: Number of cells in each dimension

    Returns:
        Tuple of (solution_values, node_positions, time_values)
        - solution_values: shape (num_time_steps, num_nodes) array of u at each node over time
        - node_positions: shape (num_nodes, 2) array of (x, y) coordinates
        - time_values: shape (num_time_steps,) array of time values
    """
    # Generate node positions matching linear FEM (nodes at cell corners)
    # For a grid with N cells per dimension, we have N+1 nodes per dimension
    nodes_per_dimension = grid_resolution + 1
    x_values = np.linspace(0.0, 1.0, nodes_per_dimension)
    y_values = np.linspace(0.0, 1.0, nodes_per_dimension)

    # Create meshgrid and flatten to get all node positions
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    node_positions = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    # Generate time values from t=0 to T
    time_values = np.linspace(0.0, DEFAULT_FINAL_TIME, DEFAULT_NUM_OUTPUT_STEPS)

    # Evaluate analytical solution at all nodes for each time point
    num_nodes = node_positions.shape[0]
    solution_values = np.zeros((DEFAULT_NUM_OUTPUT_STEPS, num_nodes))

    for time_index, time in enumerate(time_values):
        solution_values[time_index, :] = compute_analytical_solution_at_points(
            node_positions,
            time=time,
            diffusivity=DEFAULT_DIFFUSIVITY,
        )

    return solution_values, node_positions, time_values
