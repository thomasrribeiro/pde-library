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


def find_origin_node_index(node_positions: np.ndarray) -> int:
    """Find the node index closest to the origin (0.5, 0.5) for 2D problems.

    For the heat equation on [0,1]^2, the origin is at the center (0.5, 0.5),
    which is where the initial condition sin(πx)sin(πy) has its maximum.

    Args:
        node_positions: (N, 2) array of node coordinates

    Returns:
        Index of the node closest to (0.5, 0.5)
    """
    center = np.array([0.5, 0.5])
    distances = np.linalg.norm(node_positions - center, axis=1)
    return int(np.argmin(distances))


def create_time_dependent_solution_subplots(
    solutions: List[tuple],
    plot_title: str = "Solutions",
    grid_interpolation_points: int = 100,
) -> Figure:
    """Create subplots for time-dependent solutions.

    For each solver, shows:
    - Top row: Spatial domain at final time
    - Bottom row: Value at origin over time

    Args:
        solutions: List of (solver_name, solution_values, node_positions, time_values) tuples
            - solution_values: shape (num_time_steps, num_nodes)
            - node_positions: shape (num_nodes, 2)
            - time_values: shape (num_time_steps,)
        plot_title: Overall plot title
        grid_interpolation_points: Number of points for interpolation (2D only)

    Returns:
        Matplotlib Figure with subplots
    """
    from scipy.interpolate import griddata

    num_solutions = len(solutions)
    if num_solutions == 0:
        raise ValueError("No solutions provided")

    # Create figure with 2 rows: spatial domain on top, time history on bottom
    figure, axes = plt.subplots(2, num_solutions, figsize=(5 * num_solutions, 8))
    if num_solutions == 1:
        axes = axes.reshape(-1, 1)

    # Use midpoint time index for spatial plots (halfway through simulation)
    mid_time_index = len(solutions[0][3]) // 2

    # Find global min/max for consistent colorbar across all solutions at midpoint time
    all_mid_values = np.concatenate([sol[mid_time_index, :] for _, sol, _, _ in solutions])
    vmin, vmax = all_mid_values.min(), all_mid_values.max()

    for idx, (solver_name, solution, positions, time_values) in enumerate(solutions):
        # Get solution at midpoint time for spatial plot
        mid_solution = solution[mid_time_index, :]
        mid_time = time_values[mid_time_index]

        # Spatial plot (top row) - 2D interpolation
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]

        xi = np.linspace(x_coords.min(), x_coords.max(), grid_interpolation_points)
        yi = np.linspace(y_coords.min(), y_coords.max(), grid_interpolation_points)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        solution_grid = griddata(
            (x_coords, y_coords), mid_solution, (xi_grid, yi_grid), method='cubic'
        )

        im = axes[0, idx].pcolormesh(xi_grid, yi_grid, solution_grid, cmap='viridis',
                                      vmin=vmin, vmax=vmax, shading='auto')
        axes[0, idx].set_title(f"{solver_name} (t={mid_time:.3f})")
        axes[0, idx].set_xlabel('x')
        axes[0, idx].set_ylabel('y')
        axes[0, idx].set_aspect('equal')
        figure.colorbar(im, ax=axes[0, idx], label='u')

        # Time history plot (bottom row) - value at origin over time
        origin_index = find_origin_node_index(positions)
        origin_values_over_time = solution[:, origin_index]

        # Style based on solver name: analytical gets red dashed, numerical gets black solid
        if 'analytical' in solver_name.lower():
            style = {'linestyle': '--', 'color': 'red'}
        else:
            style = {'linestyle': '-', 'color': 'black'}

        axes[1, idx].plot(time_values, origin_values_over_time, linestyle=style['linestyle'],
                          color=style['color'], linewidth=2)
        axes[1, idx].set_title(f"{solver_name}: u(0.5, 0.5, t)")
        axes[1, idx].set_xlabel('Time t')
        axes[1, idx].set_ylabel('u')
        axes[1, idx].grid(True, alpha=0.3)

    figure.suptitle(plot_title)
    plt.tight_layout()
    return figure


def create_time_dependent_comparison_subplots(
    solver_a_name: str,
    solution_a: np.ndarray,
    positions_a: np.ndarray,
    time_values_a: np.ndarray,
    solver_b_name: str,
    solution_b: np.ndarray,
    positions_b: np.ndarray,
    time_values_b: np.ndarray,
    plot_title: str = "Solution Comparison",
    grid_interpolation_points: int = 100,
) -> Figure:
    """Create 2x2 comparison plot for time-dependent solutions.

    Layout:
    - Top-left: Solver A spatial domain at final time
    - Top-right: Solver B spatial domain at final time
    - Bottom-left: Spatial error |A - B| at final time
    - Bottom-right: Origin values over time for A, B, and |A - B|

    Args:
        solver_a_name: Name of first solver
        solution_a: shape (num_time_steps, num_nodes) solution from solver A
        positions_a: shape (num_nodes, 2) node positions for solver A
        time_values_a: shape (num_time_steps,) time values for solver A
        solver_b_name: Name of second solver
        solution_b: shape (num_time_steps, num_nodes) solution from solver B
        positions_b: shape (num_nodes, 2) node positions for solver B
        time_values_b: shape (num_time_steps,) time values for solver B
        plot_title: Overall plot title
        grid_interpolation_points: Number of points for interpolation

    Returns:
        Matplotlib Figure with 2x2 subplots
    """
    from scipy.interpolate import griddata

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Use midpoint time index for spatial plots (halfway through simulation)
    mid_time_index_a = len(time_values_a) // 2
    mid_time_index_b = len(time_values_b) // 2
    mid_time_a = time_values_a[mid_time_index_a]
    mid_time_b = time_values_b[mid_time_index_b]

    # Get midpoint time solutions
    mid_sol_a = solution_a[mid_time_index_a, :]
    mid_sol_b = solution_b[mid_time_index_b, :]

    # Create common grid for interpolation
    x_min = min(positions_a[:, 0].min(), positions_b[:, 0].min())
    x_max = max(positions_a[:, 0].max(), positions_b[:, 0].max())
    y_min = min(positions_a[:, 1].min(), positions_b[:, 1].min())
    y_max = max(positions_a[:, 1].max(), positions_b[:, 1].max())

    xi = np.linspace(x_min, x_max, grid_interpolation_points)
    yi = np.linspace(y_min, y_max, grid_interpolation_points)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # Interpolate solutions onto common grid
    grid_a = griddata(
        (positions_a[:, 0], positions_a[:, 1]), mid_sol_a, (xi_grid, yi_grid), method='cubic'
    )
    grid_b = griddata(
        (positions_b[:, 0], positions_b[:, 1]), mid_sol_b, (xi_grid, yi_grid), method='cubic'
    )
    error_grid = np.abs(grid_a - grid_b)

    # Consistent colorbar range for solutions
    vmin = min(mid_sol_a.min(), mid_sol_b.min())
    vmax = max(mid_sol_a.max(), mid_sol_b.max())

    # Top-left: Solver A spatial domain
    im1 = axes[0, 0].pcolormesh(xi_grid, yi_grid, grid_a, cmap='viridis',
                                 vmin=vmin, vmax=vmax, shading='auto')
    axes[0, 0].set_title(f"{solver_a_name} (t={mid_time_a:.3f})")
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_aspect('equal')
    figure.colorbar(im1, ax=axes[0, 0], label='u')

    # Top-right: Solver B spatial domain
    im2 = axes[0, 1].pcolormesh(xi_grid, yi_grid, grid_b, cmap='viridis',
                                 vmin=vmin, vmax=vmax, shading='auto')
    axes[0, 1].set_title(f"{solver_b_name} (t={mid_time_b:.3f})")
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_aspect('equal')
    figure.colorbar(im2, ax=axes[0, 1], label='u')

    # Bottom-left: Spatial error at midpoint time
    im3 = axes[1, 0].pcolormesh(xi_grid, yi_grid, error_grid, cmap='Reds', shading='auto')
    axes[1, 0].set_title(f"|{solver_a_name} - {solver_b_name}| (t={mid_time_a:.3f})")
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_aspect('equal')
    figure.colorbar(im3, ax=axes[1, 0], label='|error|')

    # Bottom-right: Origin values over time
    origin_index_a = find_origin_node_index(positions_a)
    origin_index_b = find_origin_node_index(positions_b)

    origin_a_over_time = solution_a[:, origin_index_a]
    origin_b_over_time = solution_b[:, origin_index_b]

    # Compute error at origin over time (interpolate B to A's time values if needed)
    if np.allclose(time_values_a, time_values_b):
        origin_error_over_time = np.abs(origin_a_over_time - origin_b_over_time)
        time_for_error = time_values_a
    else:
        # Interpolate B's origin values to A's time grid
        origin_b_interp = np.interp(time_values_a, time_values_b, origin_b_over_time)
        origin_error_over_time = np.abs(origin_a_over_time - origin_b_interp)
        time_for_error = time_values_a

    # Style based on solver name: analytical gets red dashed, numerical gets black solid
    def get_solver_style(solver_name: str) -> dict:
        if 'analytical' in solver_name.lower():
            return {'linestyle': '--', 'color': 'red'}
        else:
            return {'linestyle': '-', 'color': 'black'}

    style_a = get_solver_style(solver_a_name)
    style_b = get_solver_style(solver_b_name)

    # Plot solutions on primary y-axis (left)
    ax_left = axes[1, 1]
    line_a, = ax_left.plot(time_values_a, origin_a_over_time, linestyle=style_a['linestyle'],
                           color=style_a['color'], linewidth=2, label=solver_a_name)
    line_b, = ax_left.plot(time_values_b, origin_b_over_time, linestyle=style_b['linestyle'],
                           color=style_b['color'], linewidth=2, label=solver_b_name)
    ax_left.set_title("u(0.5, 0.5, t)")
    ax_left.set_xlabel('Time t')
    ax_left.set_ylabel('u', color='black')
    ax_left.tick_params(axis='y', labelcolor='black')
    ax_left.grid(True, alpha=0.3)

    # Plot error on secondary y-axis (right) with different scale
    ax_right = ax_left.twinx()
    line_err, = ax_right.plot(time_for_error, origin_error_over_time, linestyle='-',
                              linewidth=2, color='green', alpha=0.8, label='|Error|')
    ax_right.set_ylabel('|Error|', color='green')
    ax_right.tick_params(axis='y', labelcolor='green')

    # Combined legend for both axes
    lines = [line_a, line_b, line_err]
    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, loc='upper right')

    figure.suptitle(plot_title)
    plt.tight_layout()
    return figure


def create_time_dependent_convergence_plot(
    mesh_size_values: List[float],
    errors_per_resolution: List[np.ndarray],
    time_values: np.ndarray,
    plot_title: str = "Time-Dependent Convergence",
) -> Figure:
    """Create convergence plot for time-dependent problems.

    Shows:
    - Left: Log-log convergence plot with max error over time
    - Right: Error vs time for each resolution

    Args:
        mesh_size_values: List of mesh sizes h (one per resolution)
        errors_per_resolution: List of error arrays, each of shape (num_time_steps,)
        time_values: shape (num_time_steps,) array of time values
        plot_title: Plot title

    Returns:
        Matplotlib Figure with convergence plots
    """
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Log-log convergence using max error over time
    max_errors = [np.max(errs) for errs in errors_per_resolution]

    # Compute convergence rate
    h_array = np.array(mesh_size_values)
    log_h = np.log(h_array)
    log_e = np.log(max_errors)
    n = len(log_h)
    rate = (n * np.sum(log_h * log_e) - np.sum(log_h) * np.sum(log_e)) / \
           (n * np.sum(log_h**2) - np.sum(log_h)**2)

    legend_label = f"Max Error (rate={rate:.2f})"
    axes[0].loglog(mesh_size_values, max_errors, 'o-', markersize=8, linewidth=2, label=legend_label)

    # Reference lines for O(h) and O(h^2)
    h_reference_range = np.array([h_array.min(), h_array.max()])

    # O(h) reference line - first order
    scale_for_first_order = max_errors[-1] / h_array[-1]
    axes[0].loglog(
        h_reference_range,
        scale_for_first_order * h_reference_range,
        "--",
        color="gray",
        linewidth=1.5,
        label="O(h) - First Order",
    )

    # O(h^2) reference line - second order
    scale_for_second_order = max_errors[-1] / h_array[-1]**2
    axes[0].loglog(
        h_reference_range,
        scale_for_second_order * h_reference_range**2,
        ":",
        color="gray",
        linewidth=1.5,
        label="O(h^2) - Second Order",
    )

    axes[0].set_xlabel("Mesh Size h")
    axes[0].set_ylabel("Max L2 Error (over time)")
    axes[0].set_title("Convergence (Max Error)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, which="both", linestyle="-", alpha=0.3)

    # Right plot: Error vs time for each resolution
    for i, (h, errors) in enumerate(zip(mesh_size_values, errors_per_resolution)):
        resolution = int(1.0 / h)
        axes[1].plot(time_values, errors, '-o', markersize=3, linewidth=1.5, label=f"res={resolution}")

    axes[1].set_xlabel("Time t")
    axes[1].set_ylabel("L2 Error")
    axes[1].set_title("Error vs Time")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

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


def create_time_evolution_animation(
    solutions: List[tuple],
    output_path: Path,
    grid_interpolation_points: int = 100,
    fps: int = 5,
    dpi: int = 150,
) -> None:
    """Create an animated GIF showing time evolution of solutions.

    For each frame, shows all solvers side by side at the current time step.

    Args:
        solutions: List of (solver_name, solution_values, node_positions, time_values) tuples
            - solution_values: shape (num_time_steps, num_nodes)
            - node_positions: shape (num_nodes, 2)
            - time_values: shape (num_time_steps,)
        output_path: Path to save the GIF
        grid_interpolation_points: Number of points for interpolation
        fps: Frames per second for the animation
        dpi: Resolution in dots per inch
    """
    from scipy.interpolate import griddata
    import matplotlib.animation as animation

    num_solutions = len(solutions)
    if num_solutions == 0:
        raise ValueError("No solutions provided")

    # Get number of time steps (assume all solutions have same time steps)
    num_time_steps = len(solutions[0][3])

    # Find global min/max across ALL time steps for consistent colorbar
    all_values = []
    for _, sol, _, _ in solutions:
        all_values.append(sol.flatten())
    all_values = np.concatenate(all_values)
    vmin, vmax = all_values.min(), all_values.max()

    # Create figure
    figure, axes = plt.subplots(1, num_solutions, figsize=(5 * num_solutions, 4))
    if num_solutions == 1:
        axes = [axes]

    # Pre-compute interpolation grids for each solver
    grids_info = []
    for solver_name, solution, positions, time_values in solutions:
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        xi = np.linspace(x_coords.min(), x_coords.max(), grid_interpolation_points)
        yi = np.linspace(y_coords.min(), y_coords.max(), grid_interpolation_points)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        grids_info.append({
            'solver_name': solver_name,
            'solution': solution,
            'positions': positions,
            'time_values': time_values,
            'x_coords': x_coords,
            'y_coords': y_coords,
            'xi_grid': xi_grid,
            'yi_grid': yi_grid,
        })

    # Initialize plots
    images = []
    colorbars = []
    for idx, (ax, info) in enumerate(zip(axes, grids_info)):
        # Initial frame at t=0
        sol_at_t = info['solution'][0, :]
        sol_grid = griddata(
            (info['x_coords'], info['y_coords']), sol_at_t,
            (info['xi_grid'], info['yi_grid']), method='cubic'
        )
        im = ax.pcolormesh(info['xi_grid'], info['yi_grid'], sol_grid,
                           cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(f"{info['solver_name']} (t={info['time_values'][0]:.3f})")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        cbar = figure.colorbar(im, ax=ax, label='u')
        images.append(im)
        colorbars.append(cbar)

    plt.tight_layout()

    def update(frame):
        """Update function for animation."""
        for idx, (ax, info, im) in enumerate(zip(axes, grids_info, images)):
            sol_at_t = info['solution'][frame, :]
            sol_grid = griddata(
                (info['x_coords'], info['y_coords']), sol_at_t,
                (info['xi_grid'], info['yi_grid']), method='cubic'
            )
            im.set_array(sol_grid.ravel())
            ax.set_title(f"{info['solver_name']} (t={info['time_values'][frame]:.3f})")
        return images

    # Create animation
    anim = animation.FuncAnimation(
        figure, update, frames=num_time_steps,
        interval=1000 // fps, blit=False
    )

    # Save as GIF
    anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    print(f"Saved animation to {output_path}")
    plt.close(figure)
