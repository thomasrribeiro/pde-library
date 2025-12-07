"""Error metrics for comparing numerical and analytical solutions."""

import numpy as np
from typing import Dict, Optional


def compute_vector_magnitude_at_each_point(vector_field: np.ndarray) -> np.ndarray:
    """Compute the magnitude of vectors at each point.

    Args:
        vector_field: Array of shape (N, 3) containing 3D vectors

    Returns:
        Array of shape (N,) containing magnitudes
    """
    return np.linalg.norm(vector_field, axis=1)


def compute_absolute_difference_at_each_point(
    numerical_solution: np.ndarray,
    analytical_solution: np.ndarray
) -> np.ndarray:
    """Compute the absolute difference between numerical and analytical solutions.

    For vector fields, returns the magnitude of the difference vector at each point.
    For scalar fields, returns the absolute difference at each point.

    Args:
        numerical_solution: Numerical solution array, shape (N,) or (N, 3)
        analytical_solution: Analytical solution array, same shape

    Returns:
        Array of absolute errors at each point, shape (N,)
    """
    difference = numerical_solution - analytical_solution

    if difference.ndim == 2:
        # Vector field: compute magnitude of difference
        return np.linalg.norm(difference, axis=1)
    else:
        # Scalar field: absolute value
        return np.abs(difference)


def compute_relative_difference_at_each_point(
    numerical_solution: np.ndarray,
    analytical_solution: np.ndarray,
    minimum_denominator_to_avoid_division_by_zero: float = 1e-10
) -> np.ndarray:
    """Compute the relative difference between numerical and analytical solutions.

    Args:
        numerical_solution: Numerical solution array
        analytical_solution: Analytical solution array
        minimum_denominator_to_avoid_division_by_zero: Small value added to denominator

    Returns:
        Array of relative errors at each point, shape (N,)
    """
    absolute_error = compute_absolute_difference_at_each_point(
        numerical_solution, analytical_solution
    )

    if analytical_solution.ndim == 2:
        analytical_magnitude = np.linalg.norm(analytical_solution, axis=1)
    else:
        analytical_magnitude = np.abs(analytical_solution)

    return absolute_error / (analytical_magnitude + minimum_denominator_to_avoid_division_by_zero)


def compute_l2_error_norm(
    numerical_solution: np.ndarray,
    analytical_solution: np.ndarray,
    quadrature_weights: Optional[np.ndarray] = None
) -> float:
    """Compute the L2 (root mean square) error norm.

    L2 = sqrt( sum( w_i * |error_i|^2 ) )

    Args:
        numerical_solution: Numerical solution array
        analytical_solution: Analytical solution array
        quadrature_weights: Optional weights for integration (default: uniform)

    Returns:
        L2 error norm as a scalar
    """
    pointwise_errors = compute_absolute_difference_at_each_point(
        numerical_solution, analytical_solution
    )

    if quadrature_weights is None:
        quadrature_weights = np.ones(len(pointwise_errors)) / len(pointwise_errors)

    l2_error = np.sqrt(np.sum(quadrature_weights * pointwise_errors**2))
    return float(l2_error)


def compute_l_infinity_error_norm(
    numerical_solution: np.ndarray,
    analytical_solution: np.ndarray
) -> float:
    """Compute the L-infinity (maximum absolute) error norm.

    L∞ = max( |error_i| )

    Args:
        numerical_solution: Numerical solution array
        analytical_solution: Analytical solution array

    Returns:
        Maximum error as a scalar
    """
    pointwise_errors = compute_absolute_difference_at_each_point(
        numerical_solution, analytical_solution
    )
    return float(np.max(pointwise_errors))


def compute_mean_absolute_error(
    numerical_solution: np.ndarray,
    analytical_solution: np.ndarray,
    quadrature_weights: Optional[np.ndarray] = None
) -> float:
    """Compute the mean absolute error.

    Args:
        numerical_solution: Numerical solution array
        analytical_solution: Analytical solution array
        quadrature_weights: Optional weights (default: uniform)

    Returns:
        Mean absolute error as a scalar
    """
    pointwise_errors = compute_absolute_difference_at_each_point(
        numerical_solution, analytical_solution
    )

    if quadrature_weights is None:
        quadrature_weights = np.ones(len(pointwise_errors)) / len(pointwise_errors)

    return float(np.sum(quadrature_weights * pointwise_errors))


def compute_relative_l2_error_norm(
    numerical_solution: np.ndarray,
    analytical_solution: np.ndarray,
    quadrature_weights: Optional[np.ndarray] = None
) -> float:
    """Compute the relative L2 error (L2 error divided by L2 norm of analytical).

    Args:
        numerical_solution: Numerical solution array
        analytical_solution: Analytical solution array
        quadrature_weights: Optional weights for integration

    Returns:
        Relative L2 error as a scalar (0.01 means 1% error)
    """
    l2_error = compute_l2_error_norm(
        numerical_solution, analytical_solution, quadrature_weights
    )

    if quadrature_weights is None:
        quadrature_weights = np.ones(len(analytical_solution)) / len(analytical_solution)

    if analytical_solution.ndim == 2:
        analytical_magnitude = np.linalg.norm(analytical_solution, axis=1)
    else:
        analytical_magnitude = np.abs(analytical_solution)

    analytical_l2_norm = np.sqrt(np.sum(quadrature_weights * analytical_magnitude**2))

    if analytical_l2_norm > 1e-12:
        return float(l2_error / analytical_l2_norm)
    else:
        return float("inf") if l2_error > 1e-12 else 0.0


def compute_all_error_metrics(
    numerical_solution: np.ndarray,
    analytical_solution: np.ndarray,
    quadrature_weights: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute all standard error metrics between numerical and analytical solutions.

    Args:
        numerical_solution: Numerical solution array, shape (N,) or (N, 3)
        analytical_solution: Analytical solution array, same shape
        quadrature_weights: Optional weights for integration

    Returns:
        Dictionary containing:
            - l2_error: Root mean square error
            - l_infinity_error: Maximum absolute error
            - relative_l2_error: L2 error normalized by analytical norm
            - mean_absolute_error: Average absolute error
    """
    return {
        "l2_error": compute_l2_error_norm(
            numerical_solution, analytical_solution, quadrature_weights
        ),
        "l_infinity_error": compute_l_infinity_error_norm(
            numerical_solution, analytical_solution
        ),
        "relative_l2_error": compute_relative_l2_error_norm(
            numerical_solution, analytical_solution, quadrature_weights
        ),
        "mean_absolute_error": compute_mean_absolute_error(
            numerical_solution, analytical_solution, quadrature_weights
        ),
    }
