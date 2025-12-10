"""Semi-analytical solution for 2D wave equation with first-order system source injection.

The numerical solvers (jwave, warp, dolfinx) use a FIRST-ORDER velocity-pressure system:
    ∂p/∂t = -c²ρ∇·v + source    (mass conservation)
    ∂v/∂t = -∇p / ρ              (momentum conservation)

When the Ricker wavelet S(t) is injected into ∂p/∂t, the pressure response is the
TIME INTEGRAL of the response you'd get from the second-order wave equation
∂²u/∂t² = c²∇²u + S(t)·δ(x-xs,y-ys).

To match the first-order solvers, we compute the analytical solution for the
second-order equation with the DERIVATIVE of the Ricker wavelet as source:
    S'(t) = d/dt[(1 - 2·(π·f₀·(t-t_delay))²) · exp(-(π·f₀·(t-t_delay))²)]

This gives a pressure field that matches what the first-order solvers produce.

The 2D Green's function for a point source in an unbounded domain:
    G(r, t) = H(t - r/c) / (2π√(c²t² - r²))

The solution is computed via numerical convolution:
    p(x,y,t) = ∫₀ᵗ G(r, t-τ)·S'(τ) dτ

Note: 2D waves have a "tail" - the response persists after the wavefront passes.
This is physically correct behavior (unlike 3D where response is sharp).

IMPORTANT: Parameters must match jwave_solver.py for valid comparison:
    c = 1.0         (wave speed, normalized)
    f₀ = 2.0        (center frequency, properly resolved at N≥32)
    t_delay = 0.5   (source delay to ensure smooth wavelet start)
    T = 2.0         (final time, enough for wavelet to propagate)
"""

import numpy as np
from typing import Tuple
from scipy import integrate


# Wave speed (normalized)
WAVE_SPEED = 1.0

# Ricker wavelet parameters (must match jwave_solver.py)
CENTER_FREQUENCY = 2.0  # Hz
SOURCE_DELAY = 0.5  # s

# Source location (center of domain)
SOURCE_X = 0.5
SOURCE_Y = 0.5

# Time parameters (must match jwave_solver.py)
DEFAULT_FINAL_TIME = 2.0
DEFAULT_NUM_OUTPUT_STEPS = 101


def ricker_wavelet(time: float) -> float:
    """Compute Ricker wavelet (Mexican hat) at given time.

    S(t) = (1 - 2·(π·f₀·(t-t_delay))²) · exp(-(π·f₀·(t-t_delay))²)

    Args:
        time: Time at which to evaluate the wavelet

    Returns:
        Wavelet amplitude at the given time
    """
    t_shifted = time - SOURCE_DELAY
    arg = (np.pi * CENTER_FREQUENCY * t_shifted) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def ricker_wavelet_derivative(time: float) -> float:
    """Compute the time derivative of the Ricker wavelet.

    For the first-order system where source is injected into ∂p/∂t,
    we need to use dS/dt as the effective source to match the physics.

    d/dt[S(t)] = d/dt[(1 - 2u²)·exp(-u²)] where u = π·f₀·(t-t_delay)
               = [-4u·exp(-u²) + (1 - 2u²)·(-2u)·exp(-u²)] · π·f₀
               = -2u·exp(-u²)·[2 + (1 - 2u²)] · π·f₀
               = -2u·exp(-u²)·(3 - 2u²) · π·f₀

    Args:
        time: Time at which to evaluate the derivative

    Returns:
        Derivative of wavelet at the given time
    """
    t_shifted = time - SOURCE_DELAY
    u = np.pi * CENTER_FREQUENCY * t_shifted
    u_squared = u ** 2
    # dS/dt = -2u·exp(-u²)·(3 - 2u²) · π·f₀
    return -2.0 * u * np.exp(-u_squared) * (3.0 - 2.0 * u_squared) * np.pi * CENTER_FREQUENCY


def green_function_2d(distance: float, time: float) -> float:
    """Compute 2D Green's function for wave equation.

    G(r, t) = H(t - r/c) / (2π√(c²t² - r²))

    Args:
        distance: Distance from source point
        time: Time since source activation

    Returns:
        Green's function value (0 before wavefront arrives)
    """
    arrival_time = distance / WAVE_SPEED

    # Before wavefront arrives, response is zero
    if time <= arrival_time:
        return 0.0

    c_squared_t_squared = (WAVE_SPEED * time) ** 2
    r_squared = distance ** 2

    denominator_squared = c_squared_t_squared - r_squared

    # Numerical safety for very small values
    if denominator_squared < 1e-20:
        return 0.0

    return 1.0 / (2.0 * np.pi * np.sqrt(denominator_squared))


def compute_solution_at_point(
    x_coordinate: float,
    y_coordinate: float,
    time: float,
) -> float:
    """Compute pressure at a single point via Green's function convolution.

    For the first-order system, we use the Ricker wavelet derivative as source:
    p(x,y,t) = ∫₀ᵗ G(r, t-τ)·S'(τ) dτ

    Args:
        x_coordinate: x position
        y_coordinate: y position
        time: Time at which to evaluate the solution

    Returns:
        Pressure value at the given point and time
    """
    if time <= 0:
        return 0.0

    # Distance from source
    dx = x_coordinate - SOURCE_X
    dy = y_coordinate - SOURCE_Y
    distance = np.sqrt(dx ** 2 + dy ** 2)

    # Arrival time of wavefront at this point
    arrival_time = distance / WAVE_SPEED

    # If wavefront hasn't arrived yet, solution is zero
    if time <= arrival_time:
        return 0.0

    # Compute convolution integral numerically
    # p(t) = ∫₀ᵗ G(r, t-τ)·S'(τ) dτ
    # Change of variables: τ' = t - τ
    # p(t) = ∫₀ᵗ G(r, τ')·S'(t-τ') dτ'

    def integrand(tau_prime):
        if tau_prime <= arrival_time:
            return 0.0
        green = green_function_2d(distance, tau_prime)
        source_time = time - tau_prime
        if source_time < 0:
            return 0.0
        # Use derivative of Ricker wavelet for first-order system
        source = ricker_wavelet_derivative(source_time)
        return green * source

    # Use adaptive quadrature for accuracy
    result, _ = integrate.quad(
        integrand,
        arrival_time + 1e-10,  # Start just after arrival
        time,
        limit=100,
    )

    return result


def compute_analytical_solution(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    time: float,
) -> np.ndarray:
    """Compute the semi-analytical pressure solution at given time.

    Args:
        x_coordinates: Array of x coordinates
        y_coordinates: Array of y coordinates
        time: Time at which to evaluate the solution

    Returns:
        Array of pressure values at each point
    """
    num_points = len(x_coordinates)
    solution = np.zeros(num_points)

    for i in range(num_points):
        solution[i] = compute_solution_at_point(
            x_coordinates[i], y_coordinates[i], time
        )

    return solution


def compute_analytical_solution_at_points(
    points: np.ndarray,
    time: float,
) -> np.ndarray:
    """Compute the semi-analytical solution at an array of 2D points.

    Args:
        points: Array of shape (N, 2) containing (x, y) coordinates
        time: Time at which to evaluate the solution

    Returns:
        Array of shape (N,) with pressure values
    """
    x_coordinates = points[:, 0]
    y_coordinates = points[:, 1]
    return compute_analytical_solution(x_coordinates, y_coordinates, time)


def solve(grid_resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unified solver interface for CLI - generates semi-analytical solution on a grid.

    Note: This can be slow due to numerical convolution at each point.

    Args:
        grid_resolution: Number of cells in each dimension

    Returns:
        Tuple of (solution_values, node_positions, time_values)
        - solution_values: shape (num_time_steps, num_nodes) array of pressure
        - node_positions: shape (num_nodes, 2) array of (x, y) coordinates
        - time_values: shape (num_time_steps,) array of time values
    """
    # Generate node positions matching linear FEM (nodes at cell corners)
    nodes_per_dimension = grid_resolution + 1
    x_values = np.linspace(0.0, 1.0, nodes_per_dimension)
    y_values = np.linspace(0.0, 1.0, nodes_per_dimension)

    # Create meshgrid and flatten to get all node positions
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing='ij')
    node_positions = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    # Generate time values from t=0 to T
    time_values = np.linspace(0.0, DEFAULT_FINAL_TIME, DEFAULT_NUM_OUTPUT_STEPS)

    # Evaluate semi-analytical solution at all nodes for each time point
    num_nodes = node_positions.shape[0]
    solution_values = np.zeros((DEFAULT_NUM_OUTPUT_STEPS, num_nodes))

    print(f"Computing semi-analytical solution ({num_nodes} nodes, {DEFAULT_NUM_OUTPUT_STEPS} time steps)...")

    for time_index, time in enumerate(time_values):
        if time_index % 10 == 0:
            print(f"  Time step {time_index}/{DEFAULT_NUM_OUTPUT_STEPS}...")
        solution_values[time_index, :] = compute_analytical_solution_at_points(
            node_positions, time
        )

    return solution_values, node_positions, time_values
