#!/usr/bin/env python3
"""PDE Benchmark CLI - Run and compare PDE solvers."""

import sys
import argparse
import importlib.util
from pathlib import Path
from typing import List, Tuple, Optional
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
from src.visualization import (
    create_convergence_plot,
    create_solution_subplots,
    create_comparison_subplots,
    create_time_dependent_solution_subplots,
    create_time_dependent_comparison_subplots,
    create_time_dependent_convergence_plot,
    create_time_evolution_animation,
    save_figure,
    show_figure,
)


def parse_solver_path(solver_path_string: str) -> Tuple[Path, str]:
    """Parse a solver path into benchmark directory and solver name.

    Args:
        solver_path_string: Path to solver file (e.g., 'benchmarks/poisson/_2d/warp.py')

    Returns:
        Tuple of (benchmark_directory, solver_name)

    Raises:
        FileNotFoundError: If solver file doesn't exist
    """
    solver_path = Path(solver_path_string).resolve()
    if not solver_path.exists():
        raise FileNotFoundError(f"Solver file not found: {solver_path_string}")
    if not solver_path.suffix == ".py":
        raise ValueError(f"Solver must be a .py file: {solver_path_string}")

    benchmark_directory = solver_path.parent
    solver_name = solver_path.stem

    return benchmark_directory, solver_name


def load_solver_module(solver_path: Path):
    """Dynamically load a solver module.

    Args:
        solver_path: Path to the solver .py file

    Returns:
        Loaded module with solve() function

    Raises:
        FileNotFoundError: If solver module doesn't exist
        AttributeError: If module doesn't have solve() function
    """
    if not solver_path.exists():
        raise FileNotFoundError(f"Solver module not found: {solver_path}")

    spec = importlib.util.spec_from_file_location(solver_path.stem, solver_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "solve"):
        raise AttributeError(f"Module {solver_path} does not have a solve() function")

    return module


def run_solver(
    solver_path_string: str,
    grid_resolution: int,
    output_dir: Path,
    force: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]:
    """Run a solver and save results.

    Supports both static and time-dependent solvers:
    - Static solvers return (solution_values, node_positions)
    - Time-dependent solvers return (solution_values, node_positions, time_values)

    Args:
        solver_path_string: Path to solver file
        grid_resolution: Grid resolution
        output_dir: Directory to save results
        force: If True, recompute even if cached

    Returns:
        Tuple of (solution_values, node_positions, time_values, solver_name)
        time_values is None for static problems
    """
    benchmark_dir, solver_name = parse_solver_path(solver_path_string)
    solver_path = Path(solver_path_string).resolve()

    # Check cache
    if not force and result_exists(output_dir, solver_name, grid_resolution):
        print(f"  {solver_path_string} at resolution {grid_resolution}: loaded from cache")
        solution, positions, time_values, _ = load_result(output_dir, solver_name, grid_resolution)
        return solution, positions, time_values, solver_name

    # Load and run solver
    print(f"  {solver_path_string} at resolution {grid_resolution}: running...", end=" ", flush=True)
    module = load_solver_module(solver_path)
    result = module.solve(grid_resolution)

    # Handle both 2-tuple (static) and 3-tuple (time-dependent) returns
    if len(result) == 2:
        solution, positions = result
        time_values = None
    else:
        solution, positions, time_values = result

    # Save result
    save_result(output_dir, solver_name, grid_resolution, solution, positions, time_values)
    print(f"saved to {output_dir}/{solver_name}_res{grid_resolution:03d}.npz")

    return solution, positions, time_values, solver_name


def cmd_run(args):
    """Handle 'pde run' command."""
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for solver_path in args.solver:
        for resolution in args.resolution:
            try:
                run_solver(solver_path, resolution, output_dir, args.force)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"Error running {solver_path} at resolution {resolution}: {e}")
                sys.exit(1)


def cmd_compare(args):
    """Handle 'pde compare' command."""
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    force = args.force

    if len(args.solver) < 2:
        print("Error: Need at least 2 solvers to compare")
        sys.exit(1)

    # Parse all solver paths
    solver_paths = []
    solver_names = []
    for solver_path in args.solver:
        try:
            _, solver_name = parse_solver_path(solver_path)
            solver_paths.append(solver_path)
            solver_names.append(solver_name)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error with solver: {e}")
            sys.exit(1)

    print(f"Comparing: {', '.join(solver_names)}")
    print(f"Output: {output_dir}")
    print()

    for resolution in args.resolution:
        print(f"Resolution {resolution}x{resolution}:")
        print("-" * 50)

        # Ensure all solutions exist and load them
        solutions = {}
        time_values_dict = {}
        for solver_path, solver_name in zip(solver_paths, solver_names):
            if force or not result_exists(output_dir, solver_name, resolution):
                print(f"  {solver_path} at resolution {resolution}: generating...")
                run_solver(solver_path, resolution, output_dir, force=force)
            solution, _, time_values, _ = load_result(output_dir, solver_name, resolution)
            solutions[solver_name] = solution
            time_values_dict[solver_name] = time_values

        # Check if any solver is time-dependent (check all solvers, not just first)
        is_time_dependent = any(
            time_values_dict[name] is not None for name in solver_names
        )

        # Compare all pairs
        for i, (path_a, name_a) in enumerate(zip(solver_paths, solver_names)):
            for path_b, name_b in zip(solver_paths[i + 1:], solver_names[i + 1:]):
                sol_a = solutions[name_a]
                sol_b = solutions[name_b]

                if is_time_dependent:
                    # Compare over all time points
                    time_values = time_values_dict[name_a]
                    print(f"  {name_a} vs {name_b} (over {len(time_values)} time points):")

                    # Compute errors at each time point and aggregate
                    l2_errors_per_time = []
                    linf_errors_per_time = []
                    rel_l2_errors_per_time = []

                    for time_index in range(len(time_values)):
                        errors = compute_all_error_metrics(
                            sol_a[time_index, :], sol_b[time_index, :]
                        )
                        l2_errors_per_time.append(errors['l2_error'])
                        linf_errors_per_time.append(errors['l_infinity_error'])
                        rel_l2_errors_per_time.append(errors['relative_l2_error'])

                    # Report max and mean errors over time
                    print(f"    L2 Error:      max={np.max(l2_errors_per_time):.6e}, mean={np.mean(l2_errors_per_time):.6e}")
                    print(f"    L∞ Error:      max={np.max(linf_errors_per_time):.6e}, mean={np.mean(linf_errors_per_time):.6e}")
                    print(f"    Relative L2:   max={np.max(rel_l2_errors_per_time):.4%}, mean={np.mean(rel_l2_errors_per_time):.4%}")
                else:
                    # Static comparison
                    errors = compute_all_error_metrics(sol_a, sol_b)
                    print(f"  {name_a} vs {name_b}:")
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


def cmd_plot(args):
    """Handle 'pde plot' command."""
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    force = args.force

    # Parse all solver paths
    solver_paths = []
    solver_names = []
    for solver_path in args.solver:
        try:
            _, solver_name = parse_solver_path(solver_path)
            solver_paths.append(solver_path)
            solver_names.append(solver_name)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error with solver: {e}")
            sys.exit(1)

    # Use first resolution for solution plots
    plot_resolution = args.resolution[0]

    # Ensure all solutions exist and load them
    solutions_data = {}  # solver_name -> (solution, positions, time_values)
    for solver_path, solver_name in zip(solver_paths, solver_names):
        if force or not result_exists(output_dir, solver_name, plot_resolution):
            print(f"  {solver_path} at resolution {plot_resolution}: generating...")
            run_solver(solver_path, plot_resolution, output_dir, force=force)
        solution, positions, time_values, _ = load_result(output_dir, solver_name, plot_resolution)
        solutions_data[solver_name] = (solution, positions, time_values)

    figures_to_show = []

    # Build descriptive filename from solver names
    solvers_suffix = "_".join(solver_names)

    # Check if any solver is time-dependent (check all solvers, not just first)
    is_time_dependent = any(
        solutions_data[name][2] is not None for name in solver_names
    )

    # Validate --video flag: only valid for time-dependent PDEs
    if args.video and not is_time_dependent:
        print("Error: --video flag is only valid for time-dependent PDEs (like heat equation)")
        print("       The selected solver(s) do not produce time-dependent solutions.")
        sys.exit(1)

    if is_time_dependent:
        # Time-dependent plotting
        if args.compare and len(solver_names) >= 2:
            # Compare mode: 2x2 layout with spatial domains + origin values comparison
            print(f"Creating time-dependent comparison plots for {len(solver_names)} solvers at resolution {plot_resolution}...")

            # Build solution pairs with time data
            for i, name_a in enumerate(solver_names):
                for name_b in solver_names[i + 1:]:
                    sol_a, pos_a, time_a = solutions_data[name_a]
                    sol_b, pos_b, time_b = solutions_data[name_b]

                    figure = create_time_dependent_comparison_subplots(
                        solver_a_name=name_a,
                        solution_a=sol_a,
                        positions_a=pos_a,
                        time_values_a=time_a,
                        solver_b_name=name_b,
                        solution_b=sol_b,
                        positions_b=pos_b,
                        time_values_b=time_b,
                        plot_title=f"{name_a} vs {name_b} (resolution {plot_resolution})",
                    )

                    output_path = output_dir / f"comparison_{name_a}_{name_b}_res{plot_resolution:03d}.png"
                    save_figure(figure, output_path)
                    figures_to_show.append(figure)
        else:
            # Standard mode: spatial domain at final time + origin value over time
            print(f"Creating time-dependent solution plots for {len(solver_names)} solvers at resolution {plot_resolution}...")

            solutions_list = [
                (name, solutions_data[name][0], solutions_data[name][1], solutions_data[name][2])
                for name in solver_names
            ]

            figure = create_time_dependent_solution_subplots(
                solutions=solutions_list,
                plot_title=f"Solutions (resolution {plot_resolution})",
            )

            output_path = output_dir / f"solutions_{solvers_suffix}_res{plot_resolution:03d}.png"
            save_figure(figure, output_path)
            figures_to_show.append(figure)

        # Generate animated GIF if --video flag is set
        if args.video:
            print(f"Creating animated GIF for {len(solver_names)} solvers at resolution {plot_resolution}...")

            # Build solutions list for animation: (name, solution, positions, time_values)
            solutions_for_animation = [
                (name, solutions_data[name][0], solutions_data[name][1], solutions_data[name][2])
                for name in solver_names
            ]

            video_output_path = output_dir / f"animation_{solvers_suffix}_res{plot_resolution:03d}.gif"
            create_time_evolution_animation(
                solutions=solutions_for_animation,
                output_path=video_output_path,
            )
            print(f"  Saved animation to {video_output_path}")

        # Time-dependent convergence plot
        if args.convergence and len(args.resolution) >= 2 and len(solver_names) >= 2:
            print(f"Creating time-dependent convergence plots using resolutions {args.resolution}...")

            # For convergence, compare each solver against the first one as reference
            ref_name = solver_names[0]
            ref_path = solver_paths[0]

            for solver_path, solver_name in zip(solver_paths[1:], solver_names[1:]):
                mesh_sizes = []
                l2_errors_per_resolution = []  # List of arrays, one per resolution
                time_values_for_plot = None

                for resolution in args.resolution:
                    # Ensure both solutions exist
                    if force or not result_exists(output_dir, ref_name, resolution):
                        run_solver(ref_path, resolution, output_dir, force=force)
                    if force or not result_exists(output_dir, solver_name, resolution):
                        run_solver(solver_path, resolution, output_dir, force=force)

                    ref_solution, _, ref_time, _ = load_result(output_dir, ref_name, resolution)
                    solver_solution, _, solver_time, _ = load_result(output_dir, solver_name, resolution)

                    # Compute error at each time point
                    num_time_steps = len(ref_time)
                    l2_errors_at_times = []
                    for time_index in range(num_time_steps):
                        errors = compute_all_error_metrics(
                            solver_solution[time_index, :], ref_solution[time_index, :]
                        )
                        l2_errors_at_times.append(errors['l2_error'])

                    mesh_sizes.append(1.0 / resolution)
                    l2_errors_per_resolution.append(np.array(l2_errors_at_times))
                    time_values_for_plot = ref_time

                # Create time-dependent convergence plot
                figure = create_time_dependent_convergence_plot(
                    mesh_size_values=mesh_sizes,
                    errors_per_resolution=l2_errors_per_resolution,
                    time_values=time_values_for_plot,
                    plot_title=f"{solver_name} vs {ref_name}",
                )

                output_path = output_dir / f"convergence_{solver_name}_vs_{ref_name}.png"
                save_figure(figure, output_path)
                figures_to_show.append(figure)

                # Print max convergence rate
                max_errors = [np.max(errs) for errs in l2_errors_per_resolution]
                rate = compute_convergence_rate(mesh_sizes, max_errors)
                print(f"  {solver_name} vs {ref_name}: max error convergence rate = {rate:.2f}")

        elif args.convergence and len(args.resolution) < 2:
            print("Warning: --convergence requires at least 2 resolutions")
        elif args.convergence and len(solver_names) < 2:
            print("Warning: --convergence requires at least 2 solvers")
    else:
        # Static (non-time-dependent) plotting - original behavior
        if args.compare and len(solver_names) >= 2:
            # Compare mode: create comparison subplots for all pairs
            print(f"Creating comparison plots for {len(solver_names)} solvers at resolution {plot_resolution}...")

            # Build solution pairs with separate positions for each solver
            solution_pairs = []
            for i, name_a in enumerate(solver_names):
                for name_b in solver_names[i + 1:]:
                    sol_a, pos_a, _ = solutions_data[name_a]
                    sol_b, pos_b, _ = solutions_data[name_b]
                    solution_pairs.append((name_a, sol_a, pos_a, name_b, sol_b, pos_b))

            figure = create_comparison_subplots(
                solution_pairs=solution_pairs,
                plot_title=f"Solver Comparisons (resolution {plot_resolution})",
            )

            output_path = output_dir / f"comparison_{solvers_suffix}_res{plot_resolution:03d}.png"
            save_figure(figure, output_path)
            figures_to_show.append(figure)

        else:
            # Standard mode: show each solution side by side
            print(f"Creating solution plots for {len(solver_names)} solvers at resolution {plot_resolution}...")

            solutions_list = [
                (name, solutions_data[name][0], solutions_data[name][1])
                for name in solver_names
            ]

            figure = create_solution_subplots(
                solutions=solutions_list,
                plot_title=f"Solutions (resolution {plot_resolution})",
            )

            output_path = output_dir / f"solutions_{solvers_suffix}_res{plot_resolution:03d}.png"
            save_figure(figure, output_path)
            figures_to_show.append(figure)

        # Static convergence plot
        if args.convergence and len(args.resolution) >= 2 and len(solver_names) >= 2:
            print(f"Creating convergence plots using resolutions {args.resolution}...")

            # For convergence, compare each solver against the first one as reference
            ref_name = solver_names[0]
            ref_path = solver_paths[0]

            for solver_path, solver_name in zip(solver_paths[1:], solver_names[1:]):
                mesh_sizes = []
                l2_errors = []

                for resolution in args.resolution:
                    # Ensure both solutions exist
                    if force or not result_exists(output_dir, ref_name, resolution):
                        run_solver(ref_path, resolution, output_dir, force=force)
                    if force or not result_exists(output_dir, solver_name, resolution):
                        run_solver(solver_path, resolution, output_dir, force=force)

                    ref_solution, _, _, _ = load_result(output_dir, ref_name, resolution)
                    solver_solution, _, _, _ = load_result(output_dir, solver_name, resolution)

                    errors = compute_all_error_metrics(solver_solution, ref_solution)
                    mesh_sizes.append(1.0 / resolution)
                    l2_errors.append(errors['l2_error'])

                # Compute convergence rate
                rate = compute_convergence_rate(mesh_sizes, l2_errors)
                print(f"  {solver_name} vs {ref_name}: convergence rate = {rate:.2f}")

                figure = create_convergence_plot(
                    mesh_size_values=mesh_sizes,
                    error_values=l2_errors,
                    measured_convergence_rate=rate,
                    plot_title=f"{solver_name} vs {ref_name}",
                )

                output_path = output_dir / f"convergence_{solver_name}_vs_{ref_name}.png"
                save_figure(figure, output_path)
                figures_to_show.append(figure)

        elif args.convergence and len(args.resolution) < 2:
            print("Warning: --convergence requires at least 2 resolutions")
        elif args.convergence and len(solver_names) < 2:
            print("Warning: --convergence requires at least 2 solvers")

    if args.show:
        for figure in figures_to_show:
            show_figure(figure)


def cmd_list(args):
    """Handle 'pde list' command."""
    benchmarks_dir = PROJECT_ROOT / "benchmarks"

    if not benchmarks_dir.exists():
        print("No benchmarks directory found")
        return

    print("Available solvers:")
    print()

    # Group solvers by benchmark directory
    benchmarks = {}
    for path in sorted(benchmarks_dir.rglob("*.py")):
        if path.stem in ("__init__", "main"):
            continue

        benchmark_rel_path = path.parent.relative_to(PROJECT_ROOT)
        if benchmark_rel_path not in benchmarks:
            benchmarks[benchmark_rel_path] = []
        benchmarks[benchmark_rel_path].append(path)

    # Display grouped by benchmark
    for benchmark_path, solver_paths in benchmarks.items():
        print(f"{benchmark_path}/")

        solver_names = [p.name for p in solver_paths]
        print(f"  solvers: {', '.join(solver_names)}")

        # Check cached resolutions (assume all solvers in same dir share cache)
        results_dir = solver_paths[0].parent / "results"
        all_resolutions = set()
        for solver_path in solver_paths:
            resolutions = list_cached_resolutions_for_solver(results_dir, solver_path.stem)
            all_resolutions.update(resolutions)

        if all_resolutions:
            sorted_resolutions = sorted(all_resolutions)
            print(f"  cached resolutions: {sorted_resolutions}")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="PDE Benchmark CLI - Run and compare PDE solvers",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run solver(s) at specified resolutions")
    run_parser.add_argument(
        "solver", nargs="+",
        help="Solver file path(s) (e.g., warp.py or benchmarks/poisson/_2d/warp.py)"
    )
    run_parser.add_argument(
        "--output", "-o", default="results",
        help="Output directory for results (default: results)"
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
    compare_parser = subparsers.add_parser("compare", help="Compare all pairs of solvers")
    compare_parser.add_argument(
        "solver", nargs="+",
        help="Solver file paths to compare (at least 2 required)"
    )
    compare_parser.add_argument(
        "--output", "-o", default="results",
        help="Output directory containing results (default: results)"
    )
    compare_parser.add_argument(
        "--resolution", "-r", type=int, nargs="+", required=True,
        help="Grid resolution(s) to compare"
    )
    compare_parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force recomputation even if cached"
    )

    # plot command
    plot_parser = subparsers.add_parser("plot", help="Visualize solver solutions")
    plot_parser.add_argument(
        "solver", nargs="+",
        help="Solver file path(s) to plot"
    )
    plot_parser.add_argument(
        "--resolution", "-r", type=int, nargs="+", required=True,
        help="Grid resolution(s) - first used for solution plots, all used for convergence"
    )
    plot_parser.add_argument(
        "--output", "-o", default="results",
        help="Output directory for plots (default: results)"
    )
    plot_parser.add_argument(
        "--compare", "-c", action="store_true",
        help="Show pairwise comparisons with error visualization"
    )
    plot_parser.add_argument(
        "--convergence", action="store_true",
        help="Generate convergence plot (requires 2+ solvers and 2+ resolutions)"
    )
    plot_parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force recomputation even if cached"
    )
    plot_parser.add_argument(
        "--show", action="store_true",
        help="Display plots interactively"
    )
    plot_parser.add_argument(
        "--video", action="store_true",
        help="Generate animated GIF of time evolution (only for time-dependent PDEs)"
    )

    # list command
    subparsers.add_parser("list", help="List available solvers")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "plot":
        cmd_plot(args)
    elif args.command == "list":
        cmd_list(args)


if __name__ == "__main__":
    main()
