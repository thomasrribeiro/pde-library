"""2D Poisson Equation Benchmark

Verifies Warp FEM implementation against manufactured solution:
    -∇²u = 2π²sin(πx)sin(πy)  on [0,1]²
    u = 0                      on boundary

Exact solution: u(x,y) = sin(πx)sin(πy)

Usage (from project root):
    source .venv/bin/activate && uv run python benchmarks/poisson/_2d/main.py
    source .venv/bin/activate && uv run python benchmarks/poisson/_2d/main.py --resolution 64
    source .venv/bin/activate && uv run python benchmarks/poisson/_2d/main.py --resolution 8 16 32 64 --convergence
    source .venv/bin/activate && uv run python benchmarks/poisson/_2d/main.py --plot
    source .venv/bin/activate && uv run python benchmarks/poisson/_2d/main.py --force  # recompute despite cache
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import warp as wp
wp.init()

from benchmarks.poisson._2d.analytical import compute_analytical_solution_at_points
from benchmarks.poisson._2d.solver import solve_poisson_2d
from src.results import solve_with_cache, list_cached_resolutions
from src.metrics import compute_all_error_metrics
from src.visualization import (
    create_solution_comparison_from_points,
    create_convergence_plot,
    save_figure,
    show_figure,
)

# Results directory for this benchmark
RESULTS_DIR = Path(__file__).parent / "results"


def run_single_resolution(
    resolution: int,
    force_recompute: bool = False,
    quiet: bool = True,
    plot: bool = False,
) -> dict:
    """Run benchmark at a single resolution.

    Args:
        resolution: Grid resolution (cells per dimension)
        force_recompute: If True, recompute even if cached
        quiet: If True, suppress solver output
        plot: If True, show visualization

    Returns:
        Dict with resolution, mesh_size, and error metrics
    """
    print(f"\nResolution: {resolution}x{resolution}")
    print("-" * 40)

    # Solve with caching
    numerical_solution, node_positions, was_cached = solve_with_cache(
        solver_function=solve_poisson_2d,
        results_directory=RESULTS_DIR,
        grid_resolution=resolution,
        force_recompute=force_recompute,
        quiet=quiet,
    )

    if was_cached:
        print("(loaded from cache)")

    # Compute analytical solution at same points
    analytical_solution = compute_analytical_solution_at_points(node_positions)

    # Compute errors
    errors = compute_all_error_metrics(numerical_solution, analytical_solution)

    # Print results
    print(f"Number of nodes: {len(numerical_solution)}")
    print(f"L2 Error:          {errors['l2_error']:.6e}")
    print(f"L∞ Error:          {errors['l_infinity_error']:.6e}")
    print(f"Relative L2 Error: {errors['relative_l2_error']:.4%}")

    # Sanity checks
    print(f"\nNumerical  - min: {numerical_solution.min():.6f}, max: {numerical_solution.max():.6f}")
    print(f"Analytical - min: {analytical_solution.min():.6f}, max: {analytical_solution.max():.6f}")

    # Expected max is 1.0 at center (0.5, 0.5)
    center_idx = np.argmin(np.sum((node_positions - [0.5, 0.5])**2, axis=1))
    print(f"Value at center (0.5, 0.5): numerical={numerical_solution[center_idx]:.6f}, analytical={analytical_solution[center_idx]:.6f}")

    # Plot if requested
    if plot:
        figure = create_solution_comparison_from_points(
            node_positions,
            numerical_solution,
            analytical_solution,
            plot_title=f"2D Poisson Equation - Resolution {resolution}x{resolution}",
        )
        save_figure(figure, RESULTS_DIR / f"poisson_2d_res{resolution:03d}.png")
        show_figure(figure)

    return {
        "resolution": resolution,
        "mesh_size": 1.0 / resolution,
        **errors,
    }


def run_convergence_study(resolutions: list, force_recompute: bool = False) -> None:
    """Run convergence study across multiple resolutions.

    Args:
        resolutions: List of grid resolutions to test
        force_recompute: If True, recompute all even if cached
    """
    print("=" * 60)
    print("CONVERGENCE STUDY")
    print("=" * 60)

    results = []
    for res in resolutions:
        result = run_single_resolution(res, force_recompute=force_recompute, quiet=True)
        results.append(result)

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Resolution':>12} {'Mesh Size h':>14} {'L2 Error':>14} {'L∞ Error':>14}")
    print("-" * 60)

    for r in results:
        print(f"{r['resolution']:>12} {r['mesh_size']:>14.4f} {r['l2_error']:>14.6e} {r['l_infinity_error']:>14.6e}")

    # Compute and display convergence rate
    if len(results) >= 2:
        h_values = [r['mesh_size'] for r in results]
        l2_errors = [r['l2_error'] for r in results]
        rate = compute_convergence_rate(h_values, l2_errors)

        print("-" * 60)
        print(f"Measured convergence rate: {rate:.2f}")
        print("Expected rate for linear elements: ~2.0")
        print("=" * 60)

        # Save convergence plot
        figure = create_convergence_plot(
            mesh_size_values=h_values,
            error_values=l2_errors,
            measured_convergence_rate=rate,
            plot_title="2D Poisson Equation - Mesh Convergence",
        )
        save_figure(figure, RESULTS_DIR / "convergence.png")


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


def main():
    parser = argparse.ArgumentParser(
        description="2D Poisson Equation Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--resolution", type=int, nargs="+", default=[32],
        help="Grid resolution(s) for the run"
    )
    parser.add_argument(
        "--convergence", action="store_true",
        help="Run convergence study with specified resolutions"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force recomputation even if cached results exist"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress solver iteration output"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show visualization of numerical vs analytical solution"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("2D POISSON EQUATION BENCHMARK")
    print("=" * 60)
    print("Problem: -∇²u = 2π²sin(πx)sin(πy) on [0,1]²")
    print("BC:      u = 0 on boundary")
    print("Exact:   u(x,y) = sin(πx)sin(πy)")

    # Show cached results
    cached = list_cached_resolutions(RESULTS_DIR)
    if cached:
        print(f"Cached results: {cached}")

    if args.convergence:
        run_convergence_study(args.resolution, force_recompute=args.force)
    else:
        for res in args.resolution:
            run_single_resolution(res, force_recompute=args.force, quiet=args.quiet, plot=args.plot)


if __name__ == "__main__":
    main()
