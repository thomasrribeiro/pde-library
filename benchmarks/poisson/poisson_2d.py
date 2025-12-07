"""2D Poisson Equation Benchmark

Verifies Warp FEM implementation against manufactured solution:
    -∇²u = 2π²sin(πx)sin(πy)  on [0,1]²
    u = 0                      on boundary

Exact solution: u(x,y) = sin(πx)sin(πy)

Usage (from project root):
    uv run python benchmarks/poisson/poisson_2d.py
    uv run python benchmarks/poisson/poisson_2d.py --resolution 64
    uv run python benchmarks/poisson/poisson_2d.py --convergence
    uv run python benchmarks/poisson/poisson_2d.py --plot
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import warp as wp
wp.init()

from benchmarks.poisson.analytical import compute_analytical_solution_at_points
from benchmarks.poisson.solver import solve_poisson_2d


def plot_solution_comparison(
    node_positions: np.ndarray,
    numerical_solution: np.ndarray,
    analytical_solution: np.ndarray,
    resolution: int,
) -> None:
    """Plot numerical vs analytical solution with error.

    Creates a 1x3 subplot showing:
    - Numerical solution
    - Analytical solution
    - Pointwise error

    Args:
        node_positions: (N, 2) array of node coordinates
        numerical_solution: (N,) array of numerical values
        analytical_solution: (N,) array of analytical values
        resolution: Grid resolution for title
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    # Create regular grid for plotting
    x_coords = node_positions[:, 0]
    y_coords = node_positions[:, 1]

    grid_points = 100
    xi = np.linspace(0, 1, grid_points)
    yi = np.linspace(0, 1, grid_points)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # Interpolate solutions onto regular grid
    numerical_grid = griddata(
        (x_coords, y_coords), numerical_solution, (xi_grid, yi_grid), method='cubic'
    )
    analytical_grid = griddata(
        (x_coords, y_coords), analytical_solution, (xi_grid, yi_grid), method='cubic'
    )
    error_grid = np.abs(numerical_grid - analytical_grid)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Shared colorbar range for numerical and analytical
    vmin = min(numerical_solution.min(), analytical_solution.min())
    vmax = max(numerical_solution.max(), analytical_solution.max())

    # Numerical solution
    im1 = axes[0].pcolormesh(xi_grid, yi_grid, numerical_grid, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
    axes[0].set_title('Numerical Solution')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    fig.colorbar(im1, ax=axes[0], label='u')

    # Analytical solution
    im2 = axes[1].pcolormesh(xi_grid, yi_grid, analytical_grid, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
    axes[1].set_title('Analytical Solution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    fig.colorbar(im2, ax=axes[1], label='u')

    # Error
    im3 = axes[2].pcolormesh(xi_grid, yi_grid, error_grid, cmap='Reds', shading='auto')
    axes[2].set_title('|Error|')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')
    fig.colorbar(im3, ax=axes[2], label='|error|')

    fig.suptitle(f'2D Poisson Equation - Resolution {resolution}x{resolution}')
    plt.tight_layout()
    output_path = Path(__file__).parent / "poisson_2d.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")


def compute_errors(numerical: np.ndarray, analytical: np.ndarray) -> dict:
    """Compute error metrics between numerical and analytical solutions.

    Uses normalized L2 error (RMS) so that error doesn't scale with number of nodes.
    This approximates the continuous L2 norm: ||e||_L2 ≈ sqrt(1/N * sum(e_i^2))
    """
    difference = numerical - analytical

    # Normalized L2 error (root mean square) - doesn't scale with mesh refinement
    l2_error_normalized = np.sqrt(np.mean(difference**2))
    l_infinity_error = np.abs(difference).max()

    # Relative error using normalized norms
    analytical_rms = np.sqrt(np.mean(analytical**2))
    relative_l2_error = l2_error_normalized / analytical_rms if analytical_rms > 0 else float('inf')

    return {
        "l2_error": l2_error_normalized,
        "l_infinity_error": l_infinity_error,
        "relative_l2_error": relative_l2_error,
        "mean_absolute_error": np.abs(difference).mean(),
    }


def run_single_resolution(resolution: int, quiet: bool = True, plot: bool = False) -> dict:
    """Run benchmark at a single resolution."""
    print(f"\nResolution: {resolution}x{resolution}")
    print("-" * 40)

    # Solve numerically
    numerical_solution, node_positions = solve_poisson_2d(
        grid_resolution=resolution,
        polynomial_degree=1,
        quiet=quiet,
    )

    # Compute analytical solution at same points
    analytical_solution = compute_analytical_solution_at_points(node_positions)

    # Compute errors
    errors = compute_errors(numerical_solution, analytical_solution)

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
        plot_solution_comparison(node_positions, numerical_solution, analytical_solution, resolution)

    return {
        "resolution": resolution,
        "mesh_size": 1.0 / resolution,
        **errors,
    }


def run_convergence_study(resolutions: list) -> None:
    """Run convergence study across multiple resolutions."""
    print("=" * 60)
    print("CONVERGENCE STUDY")
    print("=" * 60)

    results = []
    for res in resolutions:
        result = run_single_resolution(res, quiet=True)
        results.append(result)

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Resolution':>12} {'Mesh Size h':>14} {'L2 Error':>14} {'L∞ Error':>14}")
    print("-" * 60)

    for r in results:
        print(f"{r['resolution']:>12} {r['mesh_size']:>14.4f} {r['l2_error']:>14.6e} {r['l_infinity_error']:>14.6e}")

    # Compute convergence rate
    if len(results) >= 2:
        h_values = np.array([r['mesh_size'] for r in results])
        l2_errors = np.array([r['l2_error'] for r in results])

        # Log-log linear regression for convergence rate
        log_h = np.log(h_values)
        log_e = np.log(l2_errors)

        # rate = slope of log(error) vs log(h)
        n = len(log_h)
        rate = (n * np.sum(log_h * log_e) - np.sum(log_h) * np.sum(log_e)) / \
               (n * np.sum(log_h**2) - np.sum(log_h)**2)

        print("-" * 60)
        print(f"Measured convergence rate: {rate:.2f}")
        print("Expected rate for linear elements: ~2.0")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="2D Poisson Equation Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--resolution", type=int, default=32,
        help="Grid resolution for single run"
    )
    parser.add_argument(
        "--convergence", action="store_true",
        help="Run convergence study at multiple resolutions"
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

    if args.convergence:
        run_convergence_study([8, 16, 32, 64])
    else:
        run_single_resolution(args.resolution, quiet=args.quiet, plot=args.plot)


if __name__ == "__main__":
    main()
