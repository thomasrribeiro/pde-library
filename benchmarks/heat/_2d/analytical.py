"""Analytical solutions for 2D heat/diffusion equation benchmark.

Heat equation: u_t = κ∇²u  on [0,1]²

Initial condition: u₀(x,y) = sin(πx)sin(πy)
Boundary condition: u = 0 on all boundaries (homogeneous Dirichlet)

Closed-form solution: u(x,y,t) = exp(-2π²κt) sin(πx)sin(πy)

This tests time-stepping schemes, parabolic behavior, and stability limits.

Default parameters:
    κ (diffusivity) = 0.01
    T (final time) = 0.1
"""

import numpy as np


# Default diffusivity coefficient
DEFAULT_DIFFUSIVITY = 0.01

# Default final time
DEFAULT_FINAL_TIME = 0.1


def compute_initial_condition(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute the initial condition u₀(x,y) = sin(πx)sin(πy).

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates

    Returns:
        Array of initial condition values at each point
    """
    return np.sin(np.pi * x_coordinates) * np.sin(np.pi * y_coordinates)


def compute_analytical_solution(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    time: float = DEFAULT_FINAL_TIME,
    diffusivity: float = DEFAULT_DIFFUSIVITY,
) -> np.ndarray:
    """Compute the exact solution u(x,y,t) = exp(-2π²κt) sin(πx)sin(πy).

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates
        time: Time at which to evaluate the solution
        diffusivity: Diffusivity coefficient κ

    Returns:
        Array of solution values at each point
    """
    decay_factor = np.exp(-2.0 * np.pi**2 * diffusivity * time)
    spatial_part = np.sin(np.pi * x_coordinates) * np.sin(np.pi * y_coordinates)
    return decay_factor * spatial_part


def compute_analytical_solution_at_points(
    points: np.ndarray,
    time: float = DEFAULT_FINAL_TIME,
    diffusivity: float = DEFAULT_DIFFUSIVITY,
) -> np.ndarray:
    """Compute the exact solution at an array of 2D points.

    Args:
        points: Array of shape (N, 2) containing (x, y) coordinates
        time: Time at which to evaluate the solution
        diffusivity: Diffusivity coefficient κ

    Returns:
        Array of shape (N,) with solution values
    """
    x_coordinates = points[:, 0]
    y_coordinates = points[:, 1]
    return compute_analytical_solution(x_coordinates, y_coordinates, time, diffusivity)


def solve(grid_resolution: int) -> tuple:
    """Unified solver interface for CLI - generates analytical solution on a grid.

    Evaluates the analytical solution at the default final time T=0.1
    with default diffusivity κ=0.01.

    Args:
        grid_resolution: Number of cells in each dimension

    Returns:
        Tuple of (solution_values, node_positions)
        - solution_values: shape (N,) array of u at each node at final time
        - node_positions: shape (N, 2) array of (x, y) coordinates
    """
    # Generate node positions matching linear FEM (nodes at cell corners)
    # For a grid with N cells per dimension, we have N+1 nodes per dimension
    nodes_per_dimension = grid_resolution + 1
    x_values = np.linspace(0.0, 1.0, nodes_per_dimension)
    y_values = np.linspace(0.0, 1.0, nodes_per_dimension)

    # Create meshgrid and flatten to get all node positions
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    node_positions = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    # Evaluate analytical solution at all nodes at final time
    solution_values = compute_analytical_solution_at_points(
        node_positions,
        time=DEFAULT_FINAL_TIME,
        diffusivity=DEFAULT_DIFFUSIVITY,
    )

    return solution_values, node_positions
