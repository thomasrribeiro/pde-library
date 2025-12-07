"""Analyze convergence from cached results.

This script scans a results directory, loads cached solutions,
computes errors against an analytical solution, and generates
convergence plots.

Usage (from project root):
    source .venv/bin/activate && uv run python src/analyze_convergence.py benchmarks/poisson/_2d/results --analytical poisson
    source .venv/bin/activate && uv run python src/analyze_convergence.py benchmarks/laplace/_2d/results --analytical laplace
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.results import list_cached_resolutions, load_result
from src.metrics import compute_all_error_metrics
from src.visualization import create_convergence_plot, save_figure, show_figure


def get_analytical_solution_function(problem_type: str):
    """Get the analytical solution function for a given problem type.

    Args:
        problem_type: One of 'poisson', 'laplace'

    Returns:
        Function that takes (N, 2) array of points and returns (N,) solution values
    """
    if problem_type == "poisson":
        from benchmarks.poisson._2d.analytical import compute_analytical_solution_at_points
        return compute_analytical_solution_at_points
    elif problem_type == "laplace":
        from benchmarks.laplace._2d.analytical import compute_analytical_solution_at_points
        return compute_analytical_solution_at_points
    else:
        raise ValueError(f"Unknown problem type: {problem_type}. Use 'poisson' or 'laplace'.")


def compute_convergence_rate(mesh_sizes: list, errors: list) -> float:
    """Compute convergence rate from mesh refinement data.

    Args:
        mesh_sizes: List of mesh sizes h
        errors: List of corresponding errors

    Returns:
        Convergence rate (slope of log-log plot)
    """
    log_h = np.log(mesh_sizes)
    log_e = np.log(errors)

    n = len(log_h)
    rate = (n * np.sum(log_h * log_e) - np.sum(log_h) * np.sum(log_e)) / \
           (n * np.sum(log_h**2) - np.sum(log_h)**2)

    return float(rate)


def analyze_results_directory(
    results_directory: Path,
    analytical_solution_function,
    problem_name: str = "Problem",
    show_plot: bool = True,
) -> dict:
    """Analyze all cached results in a directory.

    Args:
        results_directory: Path to results folder
        analytical_solution_function: Function to compute analytical solution
        problem_name: Name for plot titles
        show_plot: Whether to display the plot

    Returns:
        Dict with analysis results
    """
    resolutions = list_cached_resolutions(results_directory)

    if not resolutions:
        print(f"No cached results found in {results_directory}")
        return {}

    print(f"Found cached results for resolutions: {resolutions}")
    print()

    results = []

    for resolution in resolutions:
        solution_values, node_positions, metadata = load_result(results_directory, resolution)
        analytical_solution = analytical_solution_function(node_positions)
        errors = compute_all_error_metrics(solution_values, analytical_solution)

        mesh_size = 1.0 / resolution
        results.append({
            "resolution": resolution,
            "mesh_size": mesh_size,
            **errors,
        })

    # Print summary table
    print("=" * 70)
    print("CONVERGENCE ANALYSIS FROM CACHED RESULTS")
    print("=" * 70)
    print(f"{'Resolution':>12} {'Mesh Size h':>14} {'L2 Error':>14} {'L∞ Error':>14} {'Rel. L2':>12}")
    print("-" * 70)

    for r in results:
        print(f"{r['resolution']:>12} {r['mesh_size']:>14.4f} {r['l2_error']:>14.6e} {r['l_infinity_error']:>14.6e} {r['relative_l2_error']:>11.4%}")

    # Compute convergence rate
    if len(results) >= 2:
        h_values = [r['mesh_size'] for r in results]
        l2_errors = [r['l2_error'] for r in results]
        rate = compute_convergence_rate(h_values, l2_errors)

        print("-" * 70)
        print(f"Measured convergence rate: {rate:.2f}")
        print("Expected rate for linear elements: ~2.0")
        print("=" * 70)

        # Generate and save convergence plot
        figure = create_convergence_plot(
            mesh_size_values=h_values,
            error_values=l2_errors,
            measured_convergence_rate=rate,
            plot_title=f"{problem_name} - Mesh Convergence (from cached results)",
        )
        output_path = results_directory / "convergence_analysis.png"
        save_figure(figure, output_path)

        if show_plot:
            show_figure(figure)

        return {
            "resolutions": resolutions,
            "results": results,
            "convergence_rate": rate,
        }

    return {
        "resolutions": resolutions,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze convergence from cached results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results_directory",
        type=str,
        help="Path to the results directory"
    )
    parser.add_argument(
        "--analytical",
        type=str,
        required=True,
        choices=["poisson", "laplace"],
        help="Problem type for analytical solution"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Don't display the plot (still saves to file)"
    )

    args = parser.parse_args()

    results_directory = Path(args.results_directory)
    if not results_directory.exists():
        print(f"Error: Results directory does not exist: {results_directory}")
        sys.exit(1)

    analytical_function = get_analytical_solution_function(args.analytical)
    problem_name = f"2D {args.analytical.capitalize()} Equation"

    analyze_results_directory(
        results_directory=results_directory,
        analytical_solution_function=analytical_function,
        problem_name=problem_name,
        show_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
