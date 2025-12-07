"""Visualization utilities using matplotlib for plotting."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Optional, Tuple


def create_heatmap_with_vector_arrows(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    vector_x_components: np.ndarray,
    vector_y_components: np.ndarray,
    plot_title: str = "Vector Field",
    colorbar_label: str = "Magnitude",
    arrow_scale_factor: float = 1.0,
) -> Figure:
    """Create a 2D heatmap of vector magnitude with arrow overlay.

    Args:
        x_coordinates: 1D array of x coordinates
        y_coordinates: 1D array of y coordinates
        vector_x_components: 2D array of x-components, shape (len(y), len(x))
        vector_y_components: 2D array of y-components, shape (len(y), len(x))
        plot_title: Title for the plot
        colorbar_label: Label for the colorbar
        arrow_scale_factor: Scaling factor for arrow length

    Returns:
        Matplotlib Figure object
    """
    vector_magnitude = np.sqrt(vector_x_components**2 + vector_y_components**2)

    figure, axis = plt.subplots(figsize=(8, 6))

    # Add heatmap for magnitude
    heatmap = axis.pcolormesh(
        x_coordinates,
        y_coordinates,
        vector_magnitude,
        cmap="viridis",
        shading="auto",
    )
    colorbar = figure.colorbar(heatmap, ax=axis, label=colorbar_label)

    # Subsample arrows to avoid visual clutter
    arrow_spacing = max(1, len(x_coordinates) // 15)
    x_subsampled = x_coordinates[::arrow_spacing]
    y_subsampled = y_coordinates[::arrow_spacing]
    vx_subsampled = vector_x_components[::arrow_spacing, ::arrow_spacing]
    vy_subsampled = vector_y_components[::arrow_spacing, ::arrow_spacing]

    # Create meshgrid for quiver
    x_mesh, y_mesh = np.meshgrid(x_subsampled, y_subsampled)

    # Add quiver plot for arrows
    axis.quiver(
        x_mesh,
        y_mesh,
        vx_subsampled,
        vy_subsampled,
        color="white",
        scale=arrow_scale_factor,
        width=0.005,
    )

    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(plot_title)
    axis.set_aspect("equal")

    plt.tight_layout()
    return figure


def create_side_by_side_scalar_field_comparison(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    numerical_field_values: np.ndarray,
    analytical_field_values: np.ndarray,
    plot_title: str = "Numerical vs Analytical",
) -> Figure:
    """Create side-by-side comparison of numerical and analytical scalar fields.

    Shows three panels: Numerical, Analytical, and Absolute Error.

    Args:
        x_coordinates: 1D array of x coordinates
        y_coordinates: 1D array of y coordinates
        numerical_field_values: 2D array of numerical solution
        analytical_field_values: 2D array of analytical solution
        plot_title: Overall plot title

    Returns:
        Matplotlib Figure with three subplots
    """
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Use common color range for numerical and analytical
    minimum_value = min(numerical_field_values.min(), analytical_field_values.min())
    maximum_value = max(numerical_field_values.max(), analytical_field_values.max())

    # Numerical solution
    im1 = axes[0].pcolormesh(
        x_coordinates,
        y_coordinates,
        numerical_field_values,
        cmap="viridis",
        vmin=minimum_value,
        vmax=maximum_value,
        shading="auto",
    )
    axes[0].set_title("Numerical Solution")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")

    # Analytical solution
    im2 = axes[1].pcolormesh(
        x_coordinates,
        y_coordinates,
        analytical_field_values,
        cmap="viridis",
        vmin=minimum_value,
        vmax=maximum_value,
        shading="auto",
    )
    axes[1].set_title("Analytical Solution")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    figure.colorbar(im2, ax=axes[1], label="Value")

    # Absolute error
    absolute_error = np.abs(numerical_field_values - analytical_field_values)
    im3 = axes[2].pcolormesh(
        x_coordinates,
        y_coordinates,
        absolute_error,
        cmap="Reds",
        shading="auto",
    )
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")
    figure.colorbar(im3, ax=axes[2], label="Error")

    figure.suptitle(plot_title)
    plt.tight_layout()
    return figure


def create_side_by_side_vector_field_magnitude_comparison(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    numerical_vector_x: np.ndarray,
    numerical_vector_y: np.ndarray,
    analytical_vector_x: np.ndarray,
    analytical_vector_y: np.ndarray,
    plot_title: str = "Vector Field Magnitude: Numerical vs Analytical",
    shared_colorbar_range: tuple = None,
) -> Figure:
    """Compare numerical and analytical vector field magnitudes.

    Shows three panels: |B_numerical|, |B_analytical|, and |Error|.
    Uses consistent colorbar range across numerical and analytical plots.

    Args:
        x_coordinates: 1D array of x coordinates
        y_coordinates: 1D array of y coordinates
        numerical_vector_x, numerical_vector_y: Numerical vector components (2D arrays)
        analytical_vector_x, analytical_vector_y: Analytical vector components (2D arrays)
        plot_title: Plot title
        shared_colorbar_range: Optional tuple (vmin, vmax) to override automatic range

    Returns:
        Matplotlib Figure with magnitude comparison
    """
    numerical_magnitude = np.sqrt(numerical_vector_x**2 + numerical_vector_y**2)
    analytical_magnitude = np.sqrt(analytical_vector_x**2 + analytical_vector_y**2)
    magnitude_error = np.abs(numerical_magnitude - analytical_magnitude)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Use shared colorbar range for numerical and analytical plots
    if shared_colorbar_range is not None:
        minimum_magnitude, maximum_magnitude = shared_colorbar_range
    else:
        minimum_magnitude = min(numerical_magnitude.min(), analytical_magnitude.min())
        maximum_magnitude = max(numerical_magnitude.max(), analytical_magnitude.max())

    # Numerical magnitude
    im1 = axes[0].pcolormesh(
        x_coordinates,
        y_coordinates,
        numerical_magnitude,
        cmap="viridis",
        vmin=minimum_magnitude,
        vmax=maximum_magnitude,
        shading="auto",
    )
    axes[0].set_title("|B| Numerical")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    figure.colorbar(im1, ax=axes[0], label="|B|")

    # Analytical magnitude
    im2 = axes[1].pcolormesh(
        x_coordinates,
        y_coordinates,
        analytical_magnitude,
        cmap="viridis",
        vmin=minimum_magnitude,
        vmax=maximum_magnitude,
        shading="auto",
    )
    axes[1].set_title("|B| Analytical")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    figure.colorbar(im2, ax=axes[1], label="|B|")

    # Error magnitude - also use same scale for better comparison
    im3 = axes[2].pcolormesh(
        x_coordinates,
        y_coordinates,
        magnitude_error,
        cmap="Reds",
        vmin=0,
        vmax=maximum_magnitude,
        shading="auto",
    )
    axes[2].set_title("|Error|")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")
    figure.colorbar(im3, ax=axes[2], label="Error")

    figure.suptitle(plot_title)
    plt.tight_layout()
    return figure


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


def create_timing_vs_resolution_plot(
    resolution_values: List[int],
    solve_times_milliseconds: List[float],
    plot_title: str = "Solve Time vs Resolution",
) -> Figure:
    """Plot solve time vs mesh resolution.

    Args:
        resolution_values: List of grid resolutions
        solve_times_milliseconds: List of solve times in milliseconds
        plot_title: Plot title

    Returns:
        Matplotlib Figure
    """
    figure, axis = plt.subplots(figsize=(8, 5))

    axis.loglog(
        resolution_values,
        solve_times_milliseconds,
        "o-",
        markersize=8,
        linewidth=2,
        label="Solve Time",
    )

    axis.set_xlabel("Grid Resolution")
    axis.set_ylabel("Solve Time (ms)")
    axis.set_title(plot_title)
    axis.grid(True, which="both", linestyle="-", alpha=0.3)

    plt.tight_layout()
    return figure


def create_error_distribution_histogram(
    pointwise_errors: np.ndarray,
    number_of_bins: int = 50,
    plot_title: str = "Error Distribution",
) -> Figure:
    """Create histogram of pointwise errors.

    Args:
        pointwise_errors: Array of error values at each point
        number_of_bins: Number of histogram bins
        plot_title: Plot title

    Returns:
        Matplotlib Figure with histogram
    """
    figure, axis = plt.subplots(figsize=(8, 5))

    axis.hist(
        pointwise_errors,
        bins=number_of_bins,
        edgecolor="black",
        alpha=0.7,
    )

    axis.set_xlabel("Absolute Error")
    axis.set_ylabel("Count")
    axis.set_title(plot_title)

    plt.tight_layout()
    return figure


def show_figure(figure: Figure) -> None:
    """Display a matplotlib figure.

    Args:
        figure: Matplotlib Figure to display
    """
    plt.show()
