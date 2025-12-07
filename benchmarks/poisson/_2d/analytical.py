"""Analytical solutions for 2D Poisson equation benchmark.

Manufactured solution: u(x,y) = sin(πx)sin(πy)

This satisfies:
    -∇²u = f  where f(x,y) = 2π²sin(πx)sin(πy)
    u = 0     on all boundaries of [0,1]²
"""

import numpy as np


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
) -> np.ndarray:
    """Compute the source term f(x,y) = 2π²sin(πx)sin(πy).

    This is derived from -∇²u where u = sin(πx)sin(πy).

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates

    Returns:
        Array of source term values at each point
    """
    return 2.0 * np.pi**2 * np.sin(np.pi * x_coordinates) * np.sin(np.pi * y_coordinates)


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
