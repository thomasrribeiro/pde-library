"""Analytical solutions for 2D Laplace equation benchmark.

Problem: ∇²u = 0 on [0,1]²

Boundary conditions:
    u(x, 0) = 0          (bottom)
    u(x, 1) = sin(πx)    (top)
    u(0, y) = 0          (left)
    u(1, y) = 0          (right)

Analytical solution: u(x,y) = sin(πx) · sinh(πy) / sinh(π)
"""

import numpy as np


def compute_analytical_solution(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute the exact solution u(x,y) = sin(πx) · sinh(πy) / sinh(π).

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates

    Returns:
        Array of solution values at each point
    """
    sinh_pi = np.sinh(np.pi)
    return np.sin(np.pi * x_coordinates) * np.sinh(np.pi * y_coordinates) / sinh_pi


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


def compute_top_boundary_values(
    x_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute boundary values u(x, 1) = sin(πx) on the top boundary.

    Args:
        x_coordinates: Array of x coordinates along top boundary

    Returns:
        Array of boundary values
    """
    return np.sin(np.pi * x_coordinates)


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
    # Use Fortran (column-major) ravel order to match Warp's node ordering:
    # y varies first (along columns), then x increases
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    node_positions = np.column_stack([x_grid.ravel(order='F'), y_grid.ravel(order='F')])

    # Evaluate analytical solution at all nodes
    solution_values = compute_analytical_solution_at_points(node_positions)

    return solution_values, node_positions
