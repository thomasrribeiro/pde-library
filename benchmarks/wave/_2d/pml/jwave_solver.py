"""jwave solver for 2D wave equation with Ricker wavelet source and PML absorbing boundaries.

Uses the first-order velocity-pressure system implemented by jwave:
    ∂p/∂t = -c²∇·v + source    (mass conservation)
    ∂v/∂t = -∇p / ρ            (momentum conservation)

jwave uses a staggered time-stepping scheme with Perfectly Matched Layer (PML)
for absorbing boundaries, and Fourier spectral methods for spatial derivatives.

IMPORTANT: The PML is placed OUTSIDE the physical domain [0,1]²:
    - Physical domain: [0, 1]² (where we care about the solution)
    - Computational domain: extended to include PML padding on all sides
    - Source is placed at center of physical domain (0.5, 0.5)

Source term: Point source at center (0.5, 0.5) with Ricker wavelet time signature
    f(x,y,t) = S(t) · δ(x - x_s, y - y_s)

where S(t) is the Ricker wavelet (Mexican hat):
    S(t) = (1 - 2·(π·f₀·(t-t_delay))²) · exp(-(π·f₀·(t-t_delay))²)

Default parameters:
    c = 1.0         (wave speed, normalized)
    f₀ = 2.0        (center frequency, properly resolved at N≥32)
    t_delay = 0.5   (source delay to ensure smooth wavelet start)
    T = 2.0         (final time, enough for wavelet to propagate and be absorbed)
"""

from typing import Tuple

import numpy as np
from jax import numpy as jnp

from jwave import FourierSeries
from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import Domain, Medium, Sources, TimeAxis


# Wave speed (normalized)
WAVE_SPEED = 1.0

# Ricker wavelet parameters
# For proper resolution, need ~10 points per wavelength
# At resolution N, h = 1/N, λ = c/f₀, need λ/h ≥ 10 → f₀ ≤ N/(10)
# Default f₀ = 2.0 works well for resolution 32+ (λ = 0.5, 16 points/wavelength at N=32)
CENTER_FREQUENCY = 2.0  # Hz (lower frequency for better resolution at coarse grids)
SOURCE_DELAY = 0.5  # s (delay = 1/(2*f₀) ensures smooth wavelet start)

# Source location (center of physical domain)
SOURCE_X = 0.5
SOURCE_Y = 0.5

# PML size (number of grid points for absorbing layer on EACH side)
PML_SIZE = 20

# Time parameters
DEFAULT_FINAL_TIME = 2.0  # Longer time for lower frequency wavelet
DEFAULT_NUM_OUTPUT_STEPS = 101


def ricker_wavelet_signal(time_values: np.ndarray) -> np.ndarray:
    """Compute Ricker wavelet signal at given time values.

    Args:
        time_values: Array of time values

    Returns:
        Array of Ricker wavelet amplitudes
    """
    t_shifted = time_values - SOURCE_DELAY
    arg = (np.pi * CENTER_FREQUENCY * t_shifted)**2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def solve_wave_equation_2d(
    grid_resolution: int = 32,
    final_time: float = DEFAULT_FINAL_TIME,
    num_output_steps: int = DEFAULT_NUM_OUTPUT_STEPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve 2D wave equation with Ricker source and PML boundaries using jwave.

    The computational domain is extended beyond [0,1]² to place PML outside
    the physical domain.

    Args:
        grid_resolution: Number of cells in each dimension for physical domain
        final_time: Final simulation time T
        num_output_steps: Number of output time steps (including t=0)

    Returns:
        Tuple of (solution_values, node_positions, time_values)
        - solution_values: shape (num_output_steps, num_nodes) array of pressure
                          (only the physical domain [0,1]² is returned)
        - node_positions: shape (num_nodes, 2) array of (x, y) coordinates
        - time_values: shape (num_output_steps,) array of time values
    """
    # Physical domain grid
    num_physical_points = grid_resolution + 1
    dx = 1.0 / grid_resolution

    # Extended computational domain with PML padding
    # PML is added OUTSIDE the physical domain [0,1]²
    num_total_points = num_physical_points + 2 * PML_SIZE

    # The physical domain [0,1] starts at index PML_SIZE in the extended grid
    # Extended domain spans [-PML_SIZE*dx, 1 + PML_SIZE*dx]

    # Create jwave domain for extended computational region
    domain = Domain(
        N=(num_total_points, num_total_points),
        dx=(dx, dx),
    )

    # Create homogeneous medium with PML
    medium = Medium(
        domain=domain,
        sound_speed=WAVE_SPEED,
        density=1.0,
        pml_size=PML_SIZE,
    )

    # Create time axis using CFL condition
    time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=final_time)
    dt = time_axis.dt
    num_time_steps = time_axis.Nt

    # Source position in extended grid indices
    # Physical (0.5, 0.5) maps to index PML_SIZE + 0.5/dx
    source_x_idx = PML_SIZE + int(SOURCE_X / dx)
    source_y_idx = PML_SIZE + int(SOURCE_Y / dx)

    # Generate Ricker wavelet signal for all time steps
    time_array = time_axis.to_array()
    ricker_signal = ricker_wavelet_signal(np.asarray(time_array))

    # Create Sources object (positions are tuples of index arrays)
    sources = Sources(
        positions=([source_x_idx], [source_y_idx]),
        signals=jnp.array([ricker_signal]),  # Shape: (num_sources, num_time_steps)
        dt=dt,
        domain=domain,
    )

    # Run simulation
    pressure_output = simulate_wave_propagation(
        medium,
        time_axis,
        sources=sources,
    )

    # pressure_output has shape (num_time_steps, Nx, Ny, 1)
    # Extract the pressure field (remove last dimension)
    pressure_all_times = np.asarray(pressure_output.params[..., 0])

    # Compute which simulation steps correspond to output steps
    output_time_values = np.linspace(0.0, final_time, num_output_steps)
    output_simulation_steps = (output_time_values / dt).astype(int)
    output_simulation_steps[0] = 0
    output_simulation_steps[-1] = min(num_time_steps - 1, output_simulation_steps[-1])

    # Create node positions array for PHYSICAL domain only [0,1]²
    x_coords = np.linspace(0.0, 1.0, num_physical_points)
    y_coords = np.linspace(0.0, 1.0, num_physical_points)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')
    node_positions = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    # Extract pressure at output times for PHYSICAL domain only
    # Physical domain is indices [PML_SIZE : PML_SIZE + num_physical_points] in each dimension
    num_nodes = num_physical_points * num_physical_points
    solution_values = np.zeros((num_output_steps, num_nodes))

    for output_idx, sim_step in enumerate(output_simulation_steps):
        # Get pressure at this time step (shape: Nx_total, Ny_total)
        pressure_field_full = pressure_all_times[sim_step]

        # Extract physical domain region
        pressure_field_physical = pressure_field_full[
            PML_SIZE : PML_SIZE + num_physical_points,
            PML_SIZE : PML_SIZE + num_physical_points
        ]

        # Flatten to match node ordering (x varies faster)
        solution_values[output_idx, :] = pressure_field_physical.ravel()

    return solution_values, node_positions, output_time_values


def solve(grid_resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unified solver interface for CLI.

    Args:
        grid_resolution: Number of cells in each dimension

    Returns:
        Tuple of (solution_values, node_positions, time_values)
        - solution_values: shape (num_time_steps, num_nodes) array
        - node_positions: shape (num_nodes, 2) array of (x, y) coordinates
        - time_values: shape (num_time_steps,) array of time values
    """
    return solve_wave_equation_2d(grid_resolution=grid_resolution)
