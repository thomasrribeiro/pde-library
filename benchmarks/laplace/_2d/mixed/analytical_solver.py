"""Analytical solutions for 2D Laplace equation with mixed boundary conditions.

Problem: ∇²u = 0 on [0,1]²

Boundary conditions:
    u(x, 1) = sin(πx)        (top - heated)
    u(0, y) = 0              (left - cold wall)
    u(1, y) = 0              (right - cold wall)
    ∂u/∂y(x, 0) = 0          (bottom - insulating/Neumann)

Physical interpretation:
    Heat source on top with sin(πx) profile, cold walls on the sides,
    and an insulated bottom that reflects heat back up.

Note on BC compatibility:
    sin(πx) = 0 at x=0 and x=1, so the top BC is perfectly compatible
    with the cold wall (u=0) Dirichlet BCs on the sides. No corner issues!

Analytical solution via separation of variables:
    u(x,y) = sin(πx) · cosh(π(1-y)) / cosh(π)

    This satisfies:
    - Laplace equation: ∇²u = 0
    - Top BC: u(x,1) = sin(πx) · cosh(0) / cosh(π) = sin(πx) · 1/cosh(π)...
      Wait, we need u(x,1) = sin(πx), so we use: u(x,y) = sin(πx) · cosh(π(1-y)) / cosh(π)
      At y=1: cosh(0) = 1, but 1/cosh(π) ≠ 1. Let me reconsider...

    Actually the correct solution is:
    u(x,y) = sin(πx) · cosh(πy) / cosh(π)

    But this gives u(x,1) = sin(πx) · cosh(π)/cosh(π) = sin(πx) ✓
    And ∂u/∂y(x,0) = sin(πx) · π·sinh(0) / cosh(π) = 0 ✓
    And u(0,y) = sin(0) · ... = 0 ✓
    And u(1,y) = sin(π) · ... = 0 ✓
"""

import numpy as np


# Precompute constant for efficiency
COSH_PI = np.cosh(np.pi)


def compute_analytical_solution(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute exact analytical solution for mixed BC Laplace equation.

    Solution: u(x,y) = sin(πx) · cosh(πy) / cosh(π)

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates

    Returns:
        Array of solution values at each point
    """
    return np.sin(np.pi * x_coordinates) * np.cosh(np.pi * y_coordinates) / COSH_PI


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
    nodes_per_dimension = grid_resolution + 1
    x_values = np.linspace(0.0, 1.0, nodes_per_dimension)
    y_values = np.linspace(0.0, 1.0, nodes_per_dimension)

    # Create meshgrid and flatten to get all node positions
    # Use Fortran (column-major) ravel order to match Warp's node ordering
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    node_positions = np.column_stack([x_grid.ravel(order='F'), y_grid.ravel(order='F')])

    # Evaluate analytical solution at all nodes
    solution_values = compute_analytical_solution_at_points(node_positions)

    return solution_values, node_positions
