"""Analytical solution for 2D Helmholtz plane wave benchmark.

Homogeneous Helmholtz equation: -∇²u - k₀²u = 0 on [0,1]²

Exact solution (complex): u(x,y) = A·exp(i·k₀·(cos(θ)·x + sin(θ)·y))
Exact solution (real part): u(x,y) = A·cos(k₀·(cos(θ)·x + sin(θ)·y))

This represents a plane wave propagating at angle θ to the x-axis.

Mixed boundary conditions (matching FEniCSx example from Ihlenburg's book):
- Dirichlet: u = u_exact on x=0 and y=0
- Neumann: ∂u/∂n = g on x=1 and y=1

Reference: Ihlenburg, "Finite Element Analysis of Acoustic Scattering" (p138-139)
"""

import numpy as np
from typing import Tuple


# Wave parameters (matching FEniCSx example from Ihlenburg's book)
WAVE_NUMBER_K0 = 4.0 * np.pi  # k₀ = 4π (~12.57)
PROPAGATION_ANGLE_THETA_RADIANS = np.pi / 4.0  # 45 degrees
AMPLITUDE_A = 1.0

# Derived constants
WAVE_DIRECTION_X = np.cos(PROPAGATION_ANGLE_THETA_RADIANS)  # cos(π/4) = √2/2
WAVE_DIRECTION_Y = np.sin(PROPAGATION_ANGLE_THETA_RADIANS)  # sin(π/4) = √2/2
WAVELENGTH_METERS = 2.0 * np.pi / WAVE_NUMBER_K0  # λ = 2π/k₀ = 0.5


def compute_phase_at_coordinates(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute the phase φ = k₀·(cos(θ)·x + sin(θ)·y) at given coordinates.

    The phase represents the argument of the plane wave solution.

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates

    Returns:
        Array of phase values at each point
    """
    return WAVE_NUMBER_K0 * (
        WAVE_DIRECTION_X * x_coordinates + WAVE_DIRECTION_Y * y_coordinates
    )


def compute_analytical_solution_complex(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute the exact complex plane wave solution u(x,y) = A·exp(i·φ).

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates

    Returns:
        Array of complex solution values at each point
    """
    phase = compute_phase_at_coordinates(x_coordinates, y_coordinates)
    return AMPLITUDE_A * np.exp(1j * phase)


def compute_analytical_solution_real(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute the real part of plane wave solution u(x,y) = A·cos(φ).

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates

    Returns:
        Array of real solution values at each point
    """
    phase = compute_phase_at_coordinates(x_coordinates, y_coordinates)
    return AMPLITUDE_A * np.cos(phase)


def compute_analytical_solution(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute the exact plane wave solution (real part by default).

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates

    Returns:
        Array of solution values at each point
    """
    return compute_analytical_solution_real(x_coordinates, y_coordinates)


def compute_gradient_x_component(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute the x-component of ∇u for the real solution.

    For u = A·cos(φ): ∂u/∂x = -A·k₀·cos(θ)·sin(φ)

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates

    Returns:
        Array of ∂u/∂x values at each point
    """
    phase = compute_phase_at_coordinates(x_coordinates, y_coordinates)
    return -AMPLITUDE_A * WAVE_NUMBER_K0 * WAVE_DIRECTION_X * np.sin(phase)


def compute_gradient_y_component(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute the y-component of ∇u for the real solution.

    For u = A·cos(φ): ∂u/∂y = -A·k₀·sin(θ)·sin(φ)

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates

    Returns:
        Array of ∂u/∂y values at each point
    """
    phase = compute_phase_at_coordinates(x_coordinates, y_coordinates)
    return -AMPLITUDE_A * WAVE_NUMBER_K0 * WAVE_DIRECTION_Y * np.sin(phase)


def compute_neumann_boundary_flux(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    normal_x: np.ndarray,
    normal_y: np.ndarray,
) -> np.ndarray:
    """Compute the Neumann BC flux g = ∇u · n for the real solution.

    For the plane wave u = A·cos(φ):
    - ∇u = -A·k₀·sin(φ)·(cos(θ), sin(θ))
    - g = ∇u · n = -A·k₀·sin(φ)·(cos(θ)·nx + sin(θ)·ny)

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates
        normal_x: Array of outward normal x-components
        normal_y: Array of outward normal y-components

    Returns:
        Array of Neumann flux values g at each point
    """
    phase = compute_phase_at_coordinates(x_coordinates, y_coordinates)
    direction_dot_normal = WAVE_DIRECTION_X * normal_x + WAVE_DIRECTION_Y * normal_y
    return -AMPLITUDE_A * WAVE_NUMBER_K0 * np.sin(phase) * direction_dot_normal


def compute_analytical_solution_at_points(
    points: np.ndarray,
) -> np.ndarray:
    """Compute the exact plane wave solution at an array of 2D points.

    Args:
        points: Array of shape (N, 2) containing (x, y) coordinates

    Returns:
        Array of shape (N,) with solution values (real part)
    """
    x_coordinates = points[:, 0]
    y_coordinates = points[:, 1]
    return compute_analytical_solution(x_coordinates, y_coordinates)


def solve(grid_resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unified solver interface for CLI - generates analytical solution on a grid.

    Creates a grid matching the FEM solver's node positions and evaluates
    the analytical plane wave solution at each node.

    Args:
        grid_resolution: Number of cells in each dimension

    Returns:
        Tuple of (solution_values, node_positions)
        - solution_values: shape (N,) array of u at each node
        - node_positions: shape (N, 2) array of (x, y) coordinates
    """
    # Generate node positions matching linear FEM (nodes at cell corners)
    # For a grid with N cells per dimension, we have N+1 nodes per dimension
    nodes_per_dimension = grid_resolution + 1
    x_values = np.linspace(0.0, 1.0, nodes_per_dimension)
    y_values = np.linspace(0.0, 1.0, nodes_per_dimension)

    # Create meshgrid and flatten to get all node positions
    # Use indexing='ij' with default C-order ravel to match Warp/DOLFINx ordering
    # where y varies fastest: (x=0,y=0), (x=0,y=h), (x=0,y=2h), ..., (x=h,y=0), ...
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing='ij')
    node_positions = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    # Evaluate analytical solution at all nodes
    solution_values = compute_analytical_solution_at_points(node_positions)

    return solution_values, node_positions
