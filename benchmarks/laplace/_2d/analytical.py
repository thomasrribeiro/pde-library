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
