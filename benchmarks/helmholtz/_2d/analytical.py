"""Analytical solutions for 2D Helmholtz equation benchmark.

Helmholtz equation: -∇²u - k²u = f  on [0,1]²

Manufactured solution: u(x,y) = sin(πx)sin(πy)

This satisfies:
    -∇²u - k²u = f  where f(x,y) = (2π² - k²)sin(πx)sin(πy)
    u = 0           on all boundaries of [0,1]²

Default wave number: k = π (chosen so k² = π², giving f = π²sin(πx)sin(πy))
"""

import numpy as np


# Default wave number squared (k² = π²)
DEFAULT_WAVE_NUMBER_SQUARED = np.pi**2


def compute_analytical_solution(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute the exact solution u(x,y) = sin(πx)sin(πy).

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates

    Returns:
        Array of solution values at each point
    """
    return np.sin(np.pi * x_coordinates) * np.sin(np.pi * y_coordinates)


def compute_source_term(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    wave_number_squared: float = DEFAULT_WAVE_NUMBER_SQUARED,
) -> np.ndarray:
    """Compute the source term f(x,y) = (2π² - k²)sin(πx)sin(πy).

    This is derived from -∇²u - k²u where u = sin(πx)sin(πy).

    For the Helmholtz equation:
        -∇²u = 2π²sin(πx)sin(πy)  (Laplacian of sin(πx)sin(πy))
        -k²u = -k²sin(πx)sin(πy)
        f = -∇²u - k²u = (2π² - k²)sin(πx)sin(πy)

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates
        wave_number_squared: k² value (default: π²)

    Returns:
        Array of source term values at each point
    """
    coefficient = 2.0 * np.pi**2 - wave_number_squared
    return coefficient * np.sin(np.pi * x_coordinates) * np.sin(np.pi * y_coordinates)


def compute_analytical_solution_at_points(
    points: np.ndarray,
) -> np.ndarray:
    """Compute the exact solution at an array of 2D points.

    Args:
        points: Array of shape (N, 2) containing (x, y) coordinates

    Returns:
        Array of shape (N,) with solution values
    """
    x_coordinates = points[:, 0]
    y_coordinates = points[:, 1]
    return compute_analytical_solution(x_coordinates, y_coordinates)


def solve(grid_resolution: int) -> tuple:
    """Unified solver interface for CLI - generates analytical solution on a grid.

    Creates a grid matching the FEM solver's node positions and evaluates
    the analytical solution at each node.

    Args:
        grid_resolution: Number of cells in each dimension

    Returns:
        Tuple of (solution_values, node_positions)
        - solution_values: shape (N,) array of u at each node
        - node_positions: shape (N, 2) array of (x, y) coordinates
    """
    # Generate node positions matching linear FEM (nodes at cell corners)
    # For a grid with N cells per dimension, we have N+1 nodes per dimension
    nodes_per_dimension = grid_resolution + 1
    x_values = np.linspace(0.0, 1.0, nodes_per_dimension)
    y_values = np.linspace(0.0, 1.0, nodes_per_dimension)

    # Create meshgrid and flatten to get all node positions
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    node_positions = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    # Evaluate analytical solution at all nodes
    solution_values = compute_analytical_solution_at_points(node_positions)

    return solution_values, node_positions
