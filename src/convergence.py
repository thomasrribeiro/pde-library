"""Convergence analysis utilities for mesh refinement studies."""

import numpy as np
from typing import List, Dict, Callable, Any, Optional


def compute_convergence_rate_from_mesh_refinement(
    mesh_size_values: np.ndarray,
    error_values: np.ndarray
) -> float:
    """Compute convergence rate from mesh sizes and errors using linear regression.

    Fits the model: log(error) = rate * log(h) + constant
    where h is the mesh size and rate is the convergence order.

    For second-order methods, expect rate ≈ 2.0
    (error decreases by factor of 4 when mesh size halves)

    Args:
        mesh_size_values: Array of mesh sizes h (e.g., domain_size / resolution)
        error_values: Array of corresponding error norms

    Returns:
        Convergence rate (slope of log-log plot)
    """
    log_mesh_sizes = np.log(mesh_size_values)
    log_errors = np.log(error_values)

    # Linear regression using least squares
    mean_log_h = np.mean(log_mesh_sizes)
    mean_log_error = np.mean(log_errors)

    numerator = np.sum((log_mesh_sizes - mean_log_h) * (log_errors - mean_log_error))
    denominator = np.sum((log_mesh_sizes - mean_log_h) ** 2)

    convergence_rate = numerator / denominator
    return float(convergence_rate)


def run_mesh_convergence_study(
    solver_function: Callable,
    analytical_solution_function: Callable,
    error_computation_function: Callable,
    list_of_resolutions: List[int],
    domain_size: float = 1.0,
    **additional_solver_arguments: Any
) -> Dict[str, Any]:
    """Run a convergence study across multiple mesh resolutions.

    This function:
    1. Solves the problem at each resolution
    2. Computes error vs analytical solution
    3. Records solve times
    4. Computes convergence rate from the error data

    Args:
        solver_function: Function(resolution, **kwargs) -> (numerical_solution, sample_points)
        analytical_solution_function: Function(sample_points) -> analytical_solution
        error_computation_function: Function(numerical, analytical) -> dict with 'l2_error'
        list_of_resolutions: List of grid resolutions to test (e.g., [16, 32, 64, 128])
        domain_size: Size of the computational domain
        **additional_solver_arguments: Extra arguments passed to solver_function

    Returns:
        Dictionary containing:
            - resolutions: list of resolutions tested
            - mesh_sizes: h values (domain_size / resolution)
            - l2_errors: L2 error at each resolution
            - l_infinity_errors: L∞ error at each resolution
            - relative_l2_errors: relative L2 error at each resolution
            - solve_times_milliseconds: solve time for each resolution
            - convergence_rate: computed rate from L2 errors
    """
    from src.timer import create_timing_context, start_timing, stop_timing

    convergence_results = {
        "resolutions": list_of_resolutions,
        "mesh_sizes": [],
        "l2_errors": [],
        "l_infinity_errors": [],
        "relative_l2_errors": [],
        "solve_times_milliseconds": [],
    }

    for resolution in list_of_resolutions:
        mesh_size = domain_size / resolution
        convergence_results["mesh_sizes"].append(mesh_size)

        # Time the solver
        timing_context = create_timing_context(synchronize_gpu=True)
        start_timing(timing_context)

        numerical_solution, sample_points = solver_function(
            resolution=resolution,
            **additional_solver_arguments
        )

        elapsed_ms = stop_timing(timing_context)
        convergence_results["solve_times_milliseconds"].append(elapsed_ms)

        # Compute analytical solution at same points
        analytical_solution = analytical_solution_function(sample_points)

        # Compute error metrics
        error_metrics = error_computation_function(numerical_solution, analytical_solution)

        convergence_results["l2_errors"].append(error_metrics.get("l2_error", 0.0))
        convergence_results["l_infinity_errors"].append(error_metrics.get("l_infinity_error", 0.0))
        convergence_results["relative_l2_errors"].append(error_metrics.get("relative_l2_error", 0.0))

    # Compute convergence rate from L2 errors
    mesh_sizes_array = np.array(convergence_results["mesh_sizes"])
    l2_errors_array = np.array(convergence_results["l2_errors"])

    if len(l2_errors_array) >= 2 and np.all(l2_errors_array > 0):
        convergence_results["convergence_rate"] = compute_convergence_rate_from_mesh_refinement(
            mesh_sizes_array, l2_errors_array
        )
    else:
        convergence_results["convergence_rate"] = None

    return convergence_results


def format_convergence_results_as_table(convergence_results: Dict[str, Any]) -> str:
    """Format convergence study results as a human-readable text table.

    Args:
        convergence_results: Output from run_mesh_convergence_study

    Returns:
        Formatted string table ready for printing
    """
    lines = []
    lines.append("=" * 80)
    lines.append(
        f"{'Resolution':>12} {'Mesh Size h':>14} {'L2 Error':>16} "
        f"{'L∞ Error':>16} {'Time (ms)':>14}"
    )
    lines.append("-" * 80)

    for i, resolution in enumerate(convergence_results["resolutions"]):
        mesh_size = convergence_results["mesh_sizes"][i]
        l2_error = convergence_results["l2_errors"][i]
        linf_error = convergence_results["l_infinity_errors"][i]
        solve_time = convergence_results["solve_times_milliseconds"][i]

        lines.append(
            f"{resolution:>12} {mesh_size:>14.6f} {l2_error:>16.6e} "
            f"{linf_error:>16.6e} {solve_time:>14.2f}"
        )

    lines.append("=" * 80)

    if convergence_results.get("convergence_rate") is not None:
        rate = convergence_results["convergence_rate"]
        lines.append(f"Measured convergence rate: {rate:.2f}")
        lines.append("(Expected ~2.0 for second-order methods like linear Nedelec elements)")

    return "\n".join(lines)


def print_convergence_summary(convergence_results: Dict[str, Any]) -> None:
    """Print convergence study results to console."""
    print(format_convergence_results_as_table(convergence_results))
