"""Visualization utilities for FEM benchmark results."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Optional
from pathlib import Path


def create_convergence_plot(
    mesh_size_values: List[float],
    error_values: List[float],
    measured_convergence_rate: Optional[float] = None,
    plot_title: str = "Mesh Convergence Study",
    error_axis_label: str = "L2 Error",
) -> Figure:
    """Create log-log convergence plot with reference lines.

    Shows measured errors and O(h) / O(h^2) reference lines.

    Args:
        mesh_size_values: List of mesh sizes h
        error_values: List of corresponding errors
        measured_convergence_rate: Computed rate (shown in legend)
        plot_title: Plot title
        error_axis_label: Label for y-axis

    Returns:
        Matplotlib Figure with convergence plot
    """
    figure, axis = plt.subplots(figsize=(8, 6))

    # Measured data points
    legend_label = f"Measured (rate={measured_convergence_rate:.2f})" if measured_convergence_rate else "Measured"
    axis.loglog(
        mesh_size_values,
        error_values,
        "o-",
        markersize=8,
        linewidth=2,
        label=legend_label,
    )

    # Reference lines for O(h) and O(h^2)
    h_array = np.array(mesh_size_values)
    h_reference_range = np.array([h_array.min(), h_array.max()])

    # O(h) reference line - first order
    scale_for_first_order = error_values[-1] / h_array[-1]
    axis.loglog(
        h_reference_range,
        scale_for_first_order * h_reference_range,
        "--",
        color="gray",
        linewidth=1.5,
        label="O(h) - First Order",
    )

    # O(h^2) reference line - second order
    scale_for_second_order = error_values[-1] / h_array[-1]**2
    axis.loglog(
        h_reference_range,
        scale_for_second_order * h_reference_range**2,
        ":",
        color="gray",
        linewidth=1.5,
        label="O(h^2) - Second Order",
    )

    axis.set_xlabel("Mesh Size h")
    axis.set_ylabel(error_axis_label)
    axis.set_title(plot_title)
    axis.legend(loc="upper left")
    axis.grid(True, which="both", linestyle="-", alpha=0.3)

    plt.tight_layout()
    return figure


def create_solution_comparison_from_points(
    node_positions: np.ndarray,
    numerical_solution: np.ndarray,
    analytical_solution: np.ndarray,
    plot_title: str = "Numerical vs Analytical Solution",
    grid_interpolation_points: int = 100,
) -> Figure:
    """Create side-by-side comparison from scattered node data.

    Interpolates scattered FEM node data onto a regular grid for visualization.
    Shows three panels: Numerical, Analytical, and Absolute Error.

    Args:
        node_positions: (N, 2) array of node coordinates
        numerical_solution: (N,) array of numerical values at nodes
        analytical_solution: (N,) array of analytical values at nodes
        plot_title: Overall plot title
        grid_interpolation_points: Number of points in each direction for interpolation

    Returns:
        Matplotlib Figure with three subplots
    """
    from scipy.interpolate import griddata

    x_coords = node_positions[:, 0]
    y_coords = node_positions[:, 1]

    # Create regular grid for plotting
    xi = np.linspace(x_coords.min(), x_coords.max(), grid_interpolation_points)
    yi = np.linspace(y_coords.min(), y_coords.max(), grid_interpolation_points)
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
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Shared colorbar range for numerical and analytical
    vmin = min(numerical_solution.min(), analytical_solution.min())
    vmax = max(numerical_solution.max(), analytical_solution.max())

    # Numerical solution
    im1 = axes[0].pcolormesh(xi_grid, yi_grid, numerical_grid, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
    axes[0].set_title('Numerical Solution')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    figure.colorbar(im1, ax=axes[0], label='u')

    # Analytical solution
    im2 = axes[1].pcolormesh(xi_grid, yi_grid, analytical_grid, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
    axes[1].set_title('Analytical Solution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    figure.colorbar(im2, ax=axes[1], label='u')

    # Error
    im3 = axes[2].pcolormesh(xi_grid, yi_grid, error_grid, cmap='Reds', shading='auto')
    axes[2].set_title('|Error|')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')
    figure.colorbar(im3, ax=axes[2], label='|error|')

    figure.suptitle(plot_title)
    plt.tight_layout()
    return figure


def determine_problem_dimension(node_positions: np.ndarray) -> int:
    """Determine the spatial dimension of the problem from node positions.

    Args:
        node_positions: (N, D) array of node coordinates

    Returns:
        Spatial dimension (1, 2, or 3)
    """
    if node_positions.ndim == 1:
        return 1
    return node_positions.shape[1]


def create_solution_subplots(
    solutions: List[tuple],
    plot_title: str = "Solutions",
    grid_interpolation_points: int = 100,
) -> Figure:
    """Create subplots showing multiple solutions side by side.

    Args:
        solutions: List of (solver_name, solution_values, node_positions) tuples
        plot_title: Overall plot title
        grid_interpolation_points: Number of points for interpolation (2D only)

    Returns:
        Matplotlib Figure with subplots
    """
    from scipy.interpolate import griddata

    num_solutions = len(solutions)
    if num_solutions == 0:
        raise ValueError("No solutions provided")

    # Determine dimension from first solution
    _, _, first_positions = solutions[0]
    dimension = determine_problem_dimension(first_positions)

    if dimension == 1:
        # 1D: Line plots
        figure, axes = plt.subplots(1, num_solutions, figsize=(5 * num_solutions, 4))
        if num_solutions == 1:
            axes = [axes]

        for idx, (solver_name, solution, positions) in enumerate(solutions):
            axes[idx].plot(positions, solution, '-', linewidth=2)
            axes[idx].set_title(solver_name)
            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel('u')
            axes[idx].grid(True, alpha=0.3)

    elif dimension == 2:
        # 2D: imshow/pcolormesh
        figure, axes = plt.subplots(1, num_solutions, figsize=(5 * num_solutions, 4))
        if num_solutions == 1:
            axes = [axes]

        # Find global min/max for consistent colorbar
        all_values = np.concatenate([sol for _, sol, _ in solutions])
        vmin, vmax = all_values.min(), all_values.max()

        for idx, (solver_name, solution, positions) in enumerate(solutions):
            x_coords = positions[:, 0]
            y_coords = positions[:, 1]

            xi = np.linspace(x_coords.min(), x_coords.max(), grid_interpolation_points)
            yi = np.linspace(y_coords.min(), y_coords.max(), grid_interpolation_points)
            xi_grid, yi_grid = np.meshgrid(xi, yi)

            solution_grid = griddata(
                (x_coords, y_coords), solution, (xi_grid, yi_grid), method='cubic'
            )

            im = axes[idx].pcolormesh(xi_grid, yi_grid, solution_grid, cmap='viridis',
                                       vmin=vmin, vmax=vmax, shading='auto')
            axes[idx].set_title(solver_name)
            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel('y')
            axes[idx].set_aspect('equal')
            figure.colorbar(im, ax=axes[idx], label='u')

    else:
        # 3D: z=0 slice (imshow)
        figure, axes = plt.subplots(1, num_solutions, figsize=(5 * num_solutions, 4))
        if num_solutions == 1:
            axes = [axes]

        all_values = np.concatenate([sol for _, sol, _ in solutions])
        vmin, vmax = all_values.min(), all_values.max()

        for idx, (solver_name, solution, positions) in enumerate(solutions):
            # Filter to z=0 slice (with tolerance)
            z_coords = positions[:, 2]
            z_tolerance = (z_coords.max() - z_coords.min()) * 0.05
            z_slice_mask = np.abs(z_coords - z_coords.min()) < z_tolerance

            x_slice = positions[z_slice_mask, 0]
            y_slice = positions[z_slice_mask, 1]
            solution_slice = solution[z_slice_mask]

            xi = np.linspace(x_slice.min(), x_slice.max(), grid_interpolation_points)
            yi = np.linspace(y_slice.min(), y_slice.max(), grid_interpolation_points)
            xi_grid, yi_grid = np.meshgrid(xi, yi)

            solution_grid = griddata(
                (x_slice, y_slice), solution_slice, (xi_grid, yi_grid), method='cubic'
            )

            im = axes[idx].pcolormesh(xi_grid, yi_grid, solution_grid, cmap='viridis',
                                       vmin=vmin, vmax=vmax, shading='auto')
            axes[idx].set_title(f"{solver_name} (z=0 slice)")
            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel('y')
            axes[idx].set_aspect('equal')
            figure.colorbar(im, ax=axes[idx], label='u')

    figure.suptitle(plot_title)
    plt.tight_layout()
    return figure


def create_comparison_subplots(
    solution_pairs: List[tuple],
    plot_title: str = "Solution Comparisons",
    grid_interpolation_points: int = 100,
) -> Figure:
    """Create subplots comparing pairs of solutions with error visualization.

    Each row shows: Solution A, Solution B, |Error|

    Args:
        solution_pairs: List of (name_a, sol_a, pos_a, name_b, sol_b, pos_b) tuples
        plot_title: Overall plot title
        grid_interpolation_points: Number of points for interpolation (2D only)

    Returns:
        Matplotlib Figure with comparison subplots
    """
    from scipy.interpolate import griddata

    num_pairs = len(solution_pairs)
    if num_pairs == 0:
        raise ValueError("No solution pairs provided")

    # Determine dimension from first pair
    _, _, first_positions, _, _, _ = solution_pairs[0]
    dimension = determine_problem_dimension(first_positions)

    if dimension == 1:
        # 1D: Line plots with 3 columns per row
        figure, axes = plt.subplots(num_pairs, 3, figsize=(15, 4 * num_pairs))
        if num_pairs == 1:
            axes = axes.reshape(1, -1)

        for row, (name_a, sol_a, pos_a, name_b, sol_b, pos_b) in enumerate(solution_pairs):
            # Interpolate sol_b onto pos_a grid for error computation
            sol_b_interp = np.interp(pos_a, pos_b, sol_b)
            error = np.abs(sol_a - sol_b_interp)

            axes[row, 0].plot(pos_a, sol_a, '-', linewidth=2)
            axes[row, 0].set_title(name_a)
            axes[row, 0].set_xlabel('x')
            axes[row, 0].set_ylabel('u')
            axes[row, 0].grid(True, alpha=0.3)

            axes[row, 1].plot(pos_b, sol_b, '-', linewidth=2)
            axes[row, 1].set_title(name_b)
            axes[row, 1].set_xlabel('x')
            axes[row, 1].set_ylabel('u')
            axes[row, 1].grid(True, alpha=0.3)

            axes[row, 2].plot(pos_a, error, '-', linewidth=2, color='red')
            axes[row, 2].set_title(f'|{name_a} - {name_b}|')
            axes[row, 2].set_xlabel('x')
            axes[row, 2].set_ylabel('|error|')
            axes[row, 2].grid(True, alpha=0.3)

    elif dimension == 2:
        # 2D: pcolormesh with 3 columns per row
        figure, axes = plt.subplots(num_pairs, 3, figsize=(15, 4 * num_pairs))
        if num_pairs == 1:
            axes = axes.reshape(1, -1)

        for row, (name_a, sol_a, pos_a, name_b, sol_b, pos_b) in enumerate(solution_pairs):
            # Use positions from first solver to define the common grid
            x_coords_a = pos_a[:, 0]
            y_coords_a = pos_a[:, 1]
            x_coords_b = pos_b[:, 0]
            y_coords_b = pos_b[:, 1]

            # Create common grid spanning both domains
            x_min = min(x_coords_a.min(), x_coords_b.min())
            x_max = max(x_coords_a.max(), x_coords_b.max())
            y_min = min(y_coords_a.min(), y_coords_b.min())
            y_max = max(y_coords_a.max(), y_coords_b.max())

            xi = np.linspace(x_min, x_max, grid_interpolation_points)
            yi = np.linspace(y_min, y_max, grid_interpolation_points)
            xi_grid, yi_grid = np.meshgrid(xi, yi)

            # Interpolate each solution using its own positions
            grid_a = griddata((x_coords_a, y_coords_a), sol_a, (xi_grid, yi_grid), method='cubic')
            grid_b = griddata((x_coords_b, y_coords_b), sol_b, (xi_grid, yi_grid), method='cubic')
            error_grid = np.abs(grid_a - grid_b)

            vmin = min(sol_a.min(), sol_b.min())
            vmax = max(sol_a.max(), sol_b.max())

            im1 = axes[row, 0].pcolormesh(xi_grid, yi_grid, grid_a, cmap='viridis',
                                           vmin=vmin, vmax=vmax, shading='auto')
            axes[row, 0].set_title(name_a)
            axes[row, 0].set_xlabel('x')
            axes[row, 0].set_ylabel('y')
            axes[row, 0].set_aspect('equal')
            figure.colorbar(im1, ax=axes[row, 0], label='u')

            im2 = axes[row, 1].pcolormesh(xi_grid, yi_grid, grid_b, cmap='viridis',
                                           vmin=vmin, vmax=vmax, shading='auto')
            axes[row, 1].set_title(name_b)
            axes[row, 1].set_xlabel('x')
            axes[row, 1].set_ylabel('y')
            axes[row, 1].set_aspect('equal')
            figure.colorbar(im2, ax=axes[row, 1], label='u')

            im3 = axes[row, 2].pcolormesh(xi_grid, yi_grid, error_grid, cmap='Reds', shading='auto')
            axes[row, 2].set_title(f'|{name_a} - {name_b}|')
            axes[row, 2].set_xlabel('x')
            axes[row, 2].set_ylabel('y')
            axes[row, 2].set_aspect('equal')
            figure.colorbar(im3, ax=axes[row, 2], label='|error|')

    else:
        # 3D: z=0 slice
        figure, axes = plt.subplots(num_pairs, 3, figsize=(15, 4 * num_pairs))
        if num_pairs == 1:
            axes = axes.reshape(1, -1)

        for row, (name_a, sol_a, pos_a, name_b, sol_b, pos_b) in enumerate(solution_pairs):
            # Extract z=0 slice from each solver's positions separately
            z_coords_a = pos_a[:, 2]
            z_tolerance_a = (z_coords_a.max() - z_coords_a.min()) * 0.05
            z_slice_mask_a = np.abs(z_coords_a - z_coords_a.min()) < z_tolerance_a

            z_coords_b = pos_b[:, 2]
            z_tolerance_b = (z_coords_b.max() - z_coords_b.min()) * 0.05
            z_slice_mask_b = np.abs(z_coords_b - z_coords_b.min()) < z_tolerance_b

            x_slice_a = pos_a[z_slice_mask_a, 0]
            y_slice_a = pos_a[z_slice_mask_a, 1]
            sol_a_slice = sol_a[z_slice_mask_a]

            x_slice_b = pos_b[z_slice_mask_b, 0]
            y_slice_b = pos_b[z_slice_mask_b, 1]
            sol_b_slice = sol_b[z_slice_mask_b]

            # Create common grid spanning both domains
            x_min = min(x_slice_a.min(), x_slice_b.min())
            x_max = max(x_slice_a.max(), x_slice_b.max())
            y_min = min(y_slice_a.min(), y_slice_b.min())
            y_max = max(y_slice_a.max(), y_slice_b.max())

            xi = np.linspace(x_min, x_max, grid_interpolation_points)
            yi = np.linspace(y_min, y_max, grid_interpolation_points)
            xi_grid, yi_grid = np.meshgrid(xi, yi)

            # Interpolate each solution using its own positions
            grid_a = griddata((x_slice_a, y_slice_a), sol_a_slice, (xi_grid, yi_grid), method='cubic')
            grid_b = griddata((x_slice_b, y_slice_b), sol_b_slice, (xi_grid, yi_grid), method='cubic')
            error_grid = np.abs(grid_a - grid_b)

            vmin = min(sol_a_slice.min(), sol_b_slice.min())
            vmax = max(sol_a_slice.max(), sol_b_slice.max())

            im1 = axes[row, 0].pcolormesh(xi_grid, yi_grid, grid_a, cmap='viridis',
                                           vmin=vmin, vmax=vmax, shading='auto')
            axes[row, 0].set_title(f"{name_a} (z=0)")
            axes[row, 0].set_xlabel('x')
            axes[row, 0].set_ylabel('y')
            axes[row, 0].set_aspect('equal')
            figure.colorbar(im1, ax=axes[row, 0], label='u')

            im2 = axes[row, 1].pcolormesh(xi_grid, yi_grid, grid_b, cmap='viridis',
                                           vmin=vmin, vmax=vmax, shading='auto')
            axes[row, 1].set_title(f"{name_b} (z=0)")
            axes[row, 1].set_xlabel('x')
            axes[row, 1].set_ylabel('y')
            axes[row, 1].set_aspect('equal')
            figure.colorbar(im2, ax=axes[row, 1], label='u')

            im3 = axes[row, 2].pcolormesh(xi_grid, yi_grid, error_grid, cmap='Reds', shading='auto')
            axes[row, 2].set_title(f'|{name_a} - {name_b}| (z=0)')
            axes[row, 2].set_xlabel('x')
            axes[row, 2].set_ylabel('y')
            axes[row, 2].set_aspect('equal')
            figure.colorbar(im3, ax=axes[row, 2], label='|error|')

    figure.suptitle(plot_title)
    plt.tight_layout()
    return figure


def save_figure(figure: Figure, output_path: Path, dpi: int = 150) -> None:
    """Save a matplotlib figure to disk.

    Args:
        figure: Matplotlib Figure to save
        output_path: Path to save the figure
        dpi: Resolution in dots per inch
    """
    figure.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved plot to {output_path}")


def show_figure(figure: Figure) -> None:
    """Display a matplotlib figure.

    Args:
        figure: Matplotlib Figure to display
    """
    plt.show()
