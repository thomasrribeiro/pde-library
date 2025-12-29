"""Analytical solutions for 3D Laplace equation with mixed boundary conditions.

Problem: ∇²u = 0 on [0,1]³

Boundary conditions:
    u(x, y, 1) = sin(πx)sin(πy)  (top/ceiling - Dirichlet, heated)
    u(0, y, z) = 0               (x=0 wall - Dirichlet, cold)
    u(1, y, z) = 0               (x=1 wall - Dirichlet, cold)
    u(x, 0, z) = 0               (y=0 wall - Dirichlet, cold)
    u(x, 1, z) = 0               (y=1 wall - Dirichlet, cold)
    ∂u/∂z(x, y, 0) = 0           (bottom/floor - Neumann, insulated)

Analytical solution: u(x,y,z) = sin(πx) · sin(πy) · cosh(√2·π·z) / cosh(√2·π)

Let k = √2·π ≈ 4.443

Verification:
    - At x=0,1: u = 0 ✓ (sin vanishes)
    - At y=0,1: u = 0 ✓ (sin vanishes)
    - At z=1: u = sin(πx)sin(πy) · cosh(k)/cosh(k) = sin(πx)sin(πy) ✓
    - At z=0: ∂u/∂z = sin(πx)sin(πy) · k · sinh(0) / cosh(k) = 0 ✓
    - ∇²u = sin(πx)sin(πy) · [-π² - π² + k²] · cosh(kz)/cosh(k)
          = sin(πx)sin(πy) · [-2π² + 2π²] · cosh(kz)/cosh(k) = 0 ✓

Note: At z=0, the floor temperature is u = sin(πx)sin(πy)/cosh(√2π) ≈ 0.024·sin(πx)sin(πy)
This is small but non-zero, which is physically correct for an insulated boundary
that equilibrates with its surroundings.
"""

import numpy as np
from typing import Tuple


# Define the wave number k = √2·π
K_CONSTANT = np.sqrt(2.0) * np.pi


def compute_analytical_solution(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    z_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute the exact solution u(x,y,z) = sin(πx)·sin(πy)·cosh(kz)/cosh(k).

    where k = √2·π

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates
        z_coordinates: Array of z coordinates

    Returns:
        Array of solution values at each point
    """
    cosh_k = np.cosh(K_CONSTANT)
    return (
        np.sin(np.pi * x_coordinates)
        * np.sin(np.pi * y_coordinates)
        * np.cosh(K_CONSTANT * z_coordinates)
        / cosh_k
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


def compute_top_boundary_values(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute boundary values u(x, y, 1) = sin(πx)sin(πy) on the top face.

    Args:
        x_coordinates: Array of x coordinates on top face
        y_coordinates: Array of y coordinates on top face

    Returns:
        Array of boundary values
    """
    return np.sin(np.pi * x_coordinates) * np.sin(np.pi * y_coordinates)


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
    # Flatten using C (row-major) order to match Warp's 3D node ordering:
    # z varies fastest, then y, then x
    x_grid, y_grid, z_grid = np.meshgrid(x_values, y_values, z_values, indexing='ij')

    node_positions = np.column_stack([
        x_grid.ravel(order='C'),
        y_grid.ravel(order='C'),
        z_grid.ravel(order='C')
    ])

    # Evaluate analytical solution at all nodes
    solution_values = compute_analytical_solution_at_points(node_positions)

    return solution_values, node_positions
