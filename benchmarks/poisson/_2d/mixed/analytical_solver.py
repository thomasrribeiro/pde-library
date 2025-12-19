"""Analytical solutions for 2D Poisson equation with mixed boundary conditions.

Manufactured solution: u(x,y) = sin(πx)cos(πy)

This satisfies:
    -∇²u = f  where f(x,y) = 2π²sin(πx)cos(πy)

Boundary conditions:
    u(0, y) = 0           (left - Dirichlet, cold wall)
    u(1, y) = 0           (right - Dirichlet, cold wall)
    ∂u/∂y(x, 0) = 0       (bottom - Neumann, insulated)
    ∂u/∂y(x, 1) = 0       (top - Neumann, insulated)

Verification:
    - At x=0: u = sin(0)cos(πy) = 0 ✓
    - At x=1: u = sin(π)cos(πy) = 0 ✓
    - At y=0: ∂u/∂y = -πsin(πx)sin(0) = 0 ✓
    - At y=1: ∂u/∂y = -πsin(πx)sin(π) = 0 ✓
"""

import numpy as np
from typing import Tuple


def compute_analytical_solution(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute the exact solution u(x,y) = sin(πx)cos(πy).

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates

    Returns:
        Array of solution values at each point
    """
    return np.sin(np.pi * x_coordinates) * np.cos(np.pi * y_coordinates)


def compute_source_term(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute the source term f(x,y) = 2π²sin(πx)cos(πy).

    This is derived from -∇²u where u = sin(πx)cos(πy).

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates

    Returns:
        Array of source term values at each point
    """
    return 2.0 * np.pi**2 * np.sin(np.pi * x_coordinates) * np.cos(np.pi * y_coordinates)


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


def solve(grid_resolution: int) -> Tuple[np.ndarray, np.ndarray]:
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
