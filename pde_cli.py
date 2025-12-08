#!/usr/bin/env python3
"""PDE Benchmark CLI - Run and compare PDE solvers.

Usage:
    pde run <benchmark_path> --solver <names> --resolution <N>...
    pde compare <benchmark_path> --solver <names> --reference <name> --resolution <N>...
    pde plot-convergence <benchmark_path> --solver <name> --reference <name>
    pde list

Examples:
    pde run benchmarks/poisson/_2d --solver warp --resolution 8 16 32 64
    pde compare benchmarks/poisson/_2d --solver warp --reference analytical --resolution 32
    pde plot-convergence benchmarks/poisson/_2d --solver warp --reference analytical
    pde list
"""

import sys
import argparse
import importlib.util
from pathlib import Path
from typing import List, Tuple
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.results import (
    save_result,
    load_result,
    result_exists,
    list_all_cached_results,
    list_cached_resolutions_for_solver,
)
from src.metrics import compute_all_error_metrics
from src.visualization import create_convergence_plot, save_figure, show_figure


def load_solver_module(benchmark_path: Path, solver_name: str):
    """Dynamically load a solver module from a benchmark directory.

    Args:
        benchmark_path: Path to benchmark directory
        solver_name: Name of the solver (e.g., 'warp', 'analytical')

    Returns:
        Loaded module with solve() function

    Raises:
        FileNotFoundError: If solver module doesn't exist
        AttributeError: If module doesn't have solve() function
    """
    solver_file = benchmark_path / f"{solver_name}.py"
    if not solver_file.exists():
        raise FileNotFoundError(f"Solver module not found: {solver_file}")

    spec = importlib.util.spec_from_file_location(solver_name, solver_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "solve"):
        raise AttributeError(f"Module {solver_file} does not have a solve() function")

    return module


def run_solver(
    benchmark_path: Path,
    solver_name: str,
    grid_resolution: int,
    force: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a solver and save results.

    Args:
        benchmark_path: Path to benchmark directory
        solver_name: Name of the solver
        grid_resolution: Grid resolution
        force: If True, recompute even if cached

    Returns:
        Tuple of (solution_values, node_positions)
    """
    results_dir = benchmark_path / "results"

    # Check cache
    if not force and result_exists(results_dir, solver_name, grid_resolution):
        print(f"  {solver_name} at resolution {grid_resolution}: loaded from cache")
        solution, positions, _ = load_result(results_dir, solver_name, grid_resolution)
        return solution, positions

    # Load and run solver
    print(f"  {solver_name} at resolution {grid_resolution}: running...", end=" ", flush=True)
    module = load_solver_module(benchmark_path, solver_name)
    solution, positions = module.solve(grid_resolution)

    # Save result
    save_result(results_dir, solver_name, grid_resolution, solution, positions)
    print(f"saved to {solver_name}_res{grid_resolution:03d}.npz")

    return solution, positions


def cmd_run(args):
    """Handle 'pde run' command."""
    benchmark_path = Path(args.benchmark_path).resolve()
    if not benchmark_path.exists():
        print(f"Error: Benchmark path does not exist: {benchmark_path}")
        sys.exit(1)

    print(f"Running solvers in {benchmark_path}")

    for solver_name in args.solver:
        for resolution in args.resolution:
            try:
                run_solver(benchmark_path, solver_name, resolution, args.force)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"Error running {solver_name} at resolution {resolution}: {e}")
                sys.exit(1)


def cmd_compare(args):
    """Handle 'pde compare' command."""
    benchmark_path = Path(args.benchmark_path).resolve()
    if not benchmark_path.exists():
        print(f"Error: Benchmark path does not exist: {benchmark_path}")
        sys.exit(1)

    results_dir = benchmark_path / "results"

    print(f"Comparing solvers in {benchmark_path}")
    print(f"Reference: {args.reference}")
    print()

    for resolution in args.resolution:
        # Ensure reference solution exists
        if not result_exists(results_dir, args.reference, resolution):
            print(f"Reference {args.reference} at resolution {resolution} not found, generating...")
            run_solver(benchmark_path, args.reference, resolution, force=False)

        # Load reference solution
        ref_solution, ref_positions, _ = load_result(results_dir, args.reference, resolution)

        print(f"Resolution {resolution}x{resolution}:")
        print("-" * 50)

        for solver_name in args.solver:
            # Ensure solver solution exists
            if not result_exists(results_dir, solver_name, resolution):
                print(f"  {solver_name} not found, generating...")
                run_solver(benchmark_path, solver_name, resolution, force=False)

            # Load solver solution
            solver_solution, solver_positions, _ = load_result(results_dir, solver_name, resolution)

            # Compute errors
            errors = compute_all_error_metrics(solver_solution, ref_solution)

            print(f"  {solver_name} vs {args.reference}:")
            print(f"    L2 Error:      {errors['l2_error']:.6e}")
            print(f"    L∞ Error:      {errors['l_infinity_error']:.6e}")
            print(f"    Relative L2:   {errors['relative_l2_error']:.4%}")

        print()


def compute_convergence_rate(mesh_sizes: List[float], errors: List[float]) -> float:
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


def cmd_plot_convergence(args):
    """Handle 'pde plot-convergence' command."""
    benchmark_path = Path(args.benchmark_path).resolve()
    if not benchmark_path.exists():
        print(f"Error: Benchmark path does not exist: {benchmark_path}")
        sys.exit(1)

    results_dir = benchmark_path / "results"
    solver_name = args.solver
    reference_name = args.reference

    # Find all cached resolutions for this solver
    solver_resolutions = list_cached_resolutions_for_solver(results_dir, solver_name)
    if not solver_resolutions:
        print(f"Error: No cached results found for solver '{solver_name}' in {results_dir}")
        sys.exit(1)

    print(f"Found {solver_name} results: {solver_resolutions}")

    # Collect data for convergence plot
    mesh_sizes = []
    l2_errors = []

    for resolution in solver_resolutions:
        # Ensure reference exists
        if not result_exists(results_dir, reference_name, resolution):
            print(f"Reference {reference_name} at resolution {resolution} not found, generating...")
            run_solver(benchmark_path, reference_name, resolution, force=False)

        # Load solutions
        solver_solution, _, _ = load_result(results_dir, solver_name, resolution)
        ref_solution, _, _ = load_result(results_dir, reference_name, resolution)

        # Compute error
        errors = compute_all_error_metrics(solver_solution, ref_solution)

        mesh_sizes.append(1.0 / resolution)
        l2_errors.append(errors['l2_error'])

    # Compute convergence rate
    if len(mesh_sizes) >= 2:
        rate = compute_convergence_rate(mesh_sizes, l2_errors)
        print(f"Convergence rate: {rate:.2f}")
    else:
        rate = None
        print("Warning: Need at least 2 resolutions to compute convergence rate")

    # Create and save plot
    figure = create_convergence_plot(
        mesh_size_values=mesh_sizes,
        error_values=l2_errors,
        measured_convergence_rate=rate,
        plot_title=f"{benchmark_path.name} - {solver_name} vs {reference_name}",
    )

    output_path = results_dir / f"convergence_{solver_name}_vs_{reference_name}.png"
    save_figure(figure, output_path)

    if args.show:
        show_figure(figure)


def cmd_list(args):
    """Handle 'pde list' command."""
    benchmarks_dir = PROJECT_ROOT / "benchmarks"

    if not benchmarks_dir.exists():
        print("No benchmarks directory found")
        return

    print("Available benchmarks:")
    print()

    # Find all benchmark directories (those containing warp.py or analytical.py)
    benchmark_paths = []
    for path in benchmarks_dir.rglob("*.py"):
        if path.name in ("warp.py", "analytical.py"):
            benchmark_dir = path.parent
            if benchmark_dir not in benchmark_paths:
                benchmark_paths.append(benchmark_dir)

    benchmark_paths = sorted(benchmark_paths)

    for benchmark_path in benchmark_paths:
        # Get relative path from project root
        rel_path = benchmark_path.relative_to(PROJECT_ROOT)
        print(f"  {rel_path}")

        # List available solvers
        solvers = [p.stem for p in benchmark_path.glob("*.py")
                   if p.stem not in ("__init__", "main")]
        if solvers:
            print(f"    solvers: {', '.join(sorted(solvers))}")

        # List cached results
        results_dir = benchmark_path / "results"
        cached = list_all_cached_results(results_dir)
        if cached:
            for solver_name, resolutions in sorted(cached.items()):
                print(f"    {solver_name}: {resolutions}")
        else:
            print("    (no cached results)")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="PDE Benchmark CLI - Run and compare PDE solvers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run solver(s) at specified resolutions")
    run_parser.add_argument("benchmark_path", help="Path to benchmark directory")
    run_parser.add_argument(
        "--solver", "-s", nargs="+", required=True,
        help="Solver name(s) to run (e.g., warp analytical)"
    )
    run_parser.add_argument(
        "--resolution", "-r", type=int, nargs="+", required=True,
        help="Grid resolution(s) to run"
    )
    run_parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force recomputation even if cached"
    )

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare solver(s) against a reference")
    compare_parser.add_argument("benchmark_path", help="Path to benchmark directory")
    compare_parser.add_argument(
        "--solver", "-s", nargs="+", required=True,
        help="Solver name(s) to compare"
    )
    compare_parser.add_argument(
        "--reference", "-ref", required=True,
        help="Reference solver name (e.g., analytical)"
    )
    compare_parser.add_argument(
        "--resolution", "-r", type=int, nargs="+", required=True,
        help="Grid resolution(s) to compare"
    )
    compare_parser.add_argument(
        "--plot", "-p", action="store_true",
        help="Generate comparison plots"
    )

    # plot-convergence command
    conv_parser = subparsers.add_parser("plot-convergence", help="Generate convergence plot")
    conv_parser.add_argument("benchmark_path", help="Path to benchmark directory")
    conv_parser.add_argument(
        "--solver", "-s", required=True,
        help="Solver name to plot"
    )
    conv_parser.add_argument(
        "--reference", "-ref", required=True,
        help="Reference solver name (e.g., analytical)"
    )
    conv_parser.add_argument(
        "--show", action="store_true",
        help="Display plot interactively"
    )

    # list command
    subparsers.add_parser("list", help="List available benchmarks")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "plot-convergence":
        cmd_plot_convergence(args)
    elif args.command == "list":
        cmd_list(args)


if __name__ == "__main__":
    main()
