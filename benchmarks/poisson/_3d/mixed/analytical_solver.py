"""Analytical solutions for 3D Poisson equation with mixed boundary conditions.

Manufactured solution: u(x,y,z) = sin(πx)sin(πy)cos(πz)

This satisfies:
    -∇²u = f  where f(x,y,z) = 3π²sin(πx)sin(πy)cos(πz)

Boundary conditions:
    u(0, y, z) = 0        (x=0 wall - Dirichlet, cold)
    u(1, y, z) = 0        (x=1 wall - Dirichlet, cold)
    u(x, 0, z) = 0        (y=0 wall - Dirichlet, cold)
    u(x, 1, z) = 0        (y=1 wall - Dirichlet, cold)
    ∂u/∂z(x, y, 0) = 0    (bottom - Neumann, insulated floor)
    ∂u/∂z(x, y, 1) = 0    (top - Neumann, insulated ceiling)

Verification:
    - At x=0,1: u = sin(0 or π)sin(πy)cos(πz) = 0 ✓
    - At y=0,1: u = sin(πx)sin(0 or π)cos(πz) = 0 ✓
    - At z=0: ∂u/∂z = -πsin(πx)sin(πy)sin(0) = 0 ✓
    - At z=1: ∂u/∂z = -πsin(πx)sin(πy)sin(π) = 0 ✓
"""

import numpy as np
from typing import Tuple


def compute_analytical_solution(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    z_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute the exact solution u(x,y,z) = sin(πx)sin(πy)cos(πz).

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates
        z_coordinates: Array of z coordinates

    Returns:
        Array of solution values at each point
    """
    return (
        np.sin(np.pi * x_coordinates)
        * np.sin(np.pi * y_coordinates)
        * np.cos(np.pi * z_coordinates)
    )


def compute_source_term(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    z_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute the source term f(x,y,z) = 3π²sin(πx)sin(πy)cos(πz).

    This is derived from -∇²u where u = sin(πx)sin(πy)cos(πz).

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates
        z_coordinates: Array of z coordinates

    Returns:
        Array of source term values at each point
    """
    return (
        3.0 * np.pi**2
        * np.sin(np.pi * x_coordinates)
        * np.sin(np.pi * y_coordinates)
        * np.cos(np.pi * z_coordinates)
    )


def compute_analytical_solution_at_points(
    points: np.ndarray,
) -> np.ndarray:
    """Compute the exact solution at an array of 3D points.

    Args:
        points: Array of shape (N, 3) containing (x, y, z) coordinates

    Returns:
        Array of shape (N,) with solution values
    """
    x_coordinates = points[:, 0]
    y_coordinates = points[:, 1]
    z_coordinates = points[:, 2]
    return compute_analytical_solution(x_coordinates, y_coordinates, z_coordinates)


def solve(grid_resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unified solver interface for CLI - generates analytical solution on a 3D grid.

    Creates a grid matching the FEM solver's node positions and evaluates
    the analytical solution at each node.

    Args:
        grid_resolution: Number of cells in each dimension

    Returns:
        Tuple of (solution_values, node_positions)
        - solution_values: shape (N,) array of u at each node
        - node_positions: shape (N, 3) array of (x, y, z) coordinates
    """
    # Generate node positions matching linear FEM (nodes at cell corners)
    # For a grid with N cells per dimension, we have N+1 nodes per dimension
    nodes_per_dimension = grid_resolution + 1
    x_values = np.linspace(0.0, 1.0, nodes_per_dimension)
    y_values = np.linspace(0.0, 1.0, nodes_per_dimension)
    z_values = np.linspace(0.0, 1.0, nodes_per_dimension)

    # Create meshgrid with 'ij' indexing for consistent ordering
    # Flatten using Fortran (column-major) order to match Warp's node ordering
    x_grid, y_grid, z_grid = np.meshgrid(x_values, y_values, z_values, indexing='ij')

    node_positions = np.column_stack([
        x_grid.ravel(order='F'),
        y_grid.ravel(order='F'),
        z_grid.ravel(order='F')
    ])

    # Evaluate analytical solution at all nodes
    solution_values = compute_analytical_solution_at_points(node_positions)

    return solution_values, node_positions
