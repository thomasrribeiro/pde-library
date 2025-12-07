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
