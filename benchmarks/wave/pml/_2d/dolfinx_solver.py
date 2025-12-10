"""DOLFINx-style solver for 2D wave equation with Ricker wavelet source and PML absorbing boundaries.

Uses the FIRST-ORDER velocity-pressure system with STAGGERED GRID:
    ∂p/∂t = -c²ρ∇·v + source    (mass conservation)
    ∂v/∂t = -∇p / ρ              (momentum conservation)

NOTE: This solver uses staggered grid finite differences (like warp_solver.py)
implemented in pure NumPy. The FEM discretization in DOLFINx doesn't work well
with staggered-in-time PML schemes for wave equations.

Staggered grid layout (avoids checkerboard instability):
    - Pressure p[i,j] at cell centers
    - Velocity vx[i,j] at x-faces (between p[i-1,j] and p[i,j])
    - Velocity vy[i,j] at y-faces (between p[i,j-1] and p[i,j])

IMPORTANT: The PML is placed OUTSIDE the physical domain [0,1]²:
    - Physical domain: [0, 1]² (where we care about the solution)
    - Computational domain: extended to include PML padding on all sides
    - Source is placed at center of physical domain (0.5, 0.5)

Source term: Discrete point source at center (0.5, 0.5) with Ricker wavelet time signature
    f(x,y,t) = S(t) · δ(x - x_s, y - y_s)

where S(t) is the Ricker wavelet (Mexican hat):
    S(t) = (1 - 2·(π·f₀·(t-t_delay))²) · exp(-(π·f₀·(t-t_delay))²)

Default parameters:
    c = 1.0         (wave speed, normalized)
    ρ = 1.0         (density, normalized)
    f₀ = 2.0        (center frequency, properly resolved at N≥32)
    t_delay = 0.5   (source delay to ensure smooth wavelet start)
    T = 2.0         (final time, enough for wavelet to propagate and be absorbed)
"""

from typing import Tuple

import numpy as np


# Wave speed and density (normalized)
WAVE_SPEED = 1.0
DENSITY = 1.0

# Ricker wavelet parameters (must match jwave_solver.py)
CENTER_FREQUENCY = 2.0  # Hz
SOURCE_DELAY = 0.5  # s

# Source location (center of physical domain)
SOURCE_X = 0.5
SOURCE_Y = 0.5

# PML parameters (jwave-style exponential PML)
PML_SIZE = 20  # Number of grid points for PML layer on EACH side
PML_EXPONENT = 4.0  # Polynomial exponent for PML profile (jwave default)
PML_ALPHA_MAX = 2.0  # Maximum decay factor (jwave default)

# Time parameters (must match jwave_solver.py)
DEFAULT_FINAL_TIME = 2.0
DEFAULT_NUM_OUTPUT_STEPS = 101


def ricker_wavelet(time: float) -> float:
    """Compute Ricker wavelet at given time."""
    t_shifted = time - SOURCE_DELAY
    arg = (np.pi * CENTER_FREQUENCY * t_shifted)**2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def compute_pml_exponential_1d(
    n_points: int,
    pml_size: int,
    dt: float,
    c0: float,
    dx: float,
    coord_shift: float = 0.0,
) -> np.ndarray:
    """Compute 1D exponential PML decay factors (jwave-style).

    Uses the exponential decay profile from jwave:
        alpha = exp(alpha_max * (-1) * x^exponent * dt * c0 / 2 / dx)

    The update formula is: field = pml * (pml * field + dt * derivative)

    Args:
        n_points: Total number of grid points
        pml_size: Number of PML points on each side
        dt: Time step
        c0: Wave speed
        dx: Grid spacing
        coord_shift: Coordinate shift for staggered grids (0.0 for cell centers, 0.5 for faces)

    Returns:
        1D array of PML decay factors (1.0 in physical domain, <1.0 in PML)
    """
    pml = np.ones(n_points)

    # Left PML: x goes from pml_size down to 1 (normalized to [0,1])
    x_left = (np.arange(pml_size, 0, -1) - coord_shift) / pml_size
    x_left = np.clip(x_left, 0, None)  # Ensure non-negative
    x_left = x_left ** PML_EXPONENT
    alpha_left = np.exp(PML_ALPHA_MAX * (-1) * x_left * dt * c0 / 2 / dx)
    pml[:pml_size] = alpha_left

    # Right PML: x goes from 1 to pml_size (normalized to [0,1])
    x_right = (np.arange(1, pml_size + 1) + coord_shift) / pml_size
    x_right = np.clip(x_right, None, 1)  # Cap at 1
    x_right = x_right ** PML_EXPONENT
    alpha_right = np.exp(PML_ALPHA_MAX * (-1) * x_right * dt * c0 / 2 / dx)
    pml[-pml_size:] = alpha_right

    return pml


def solve_wave_equation_2d(
    grid_resolution: int = 32,
    final_time: float = DEFAULT_FINAL_TIME,
    num_output_steps: int = DEFAULT_NUM_OUTPUT_STEPS,
    quiet: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve 2D wave equation using staggered grid finite differences.

    Uses the same algorithm as warp_solver.py but implemented in pure NumPy.

    Args:
        grid_resolution: Number of cells in each dimension for physical domain
        final_time: Final simulation time T
        num_output_steps: Number of output time steps (including t=0)
        quiet: If True, suppress progress output

    Returns:
        Tuple of (solution_values, node_positions, time_values)
        - solution_values: shape (num_output_steps, num_nodes) array of pressure
                          (only the physical domain [0,1]² is returned)
        - node_positions: shape (num_nodes, 2) array of (x, y) coordinates
        - time_values: shape (num_output_steps,) array of time values
    """
    c_squared = WAVE_SPEED**2
    c_squared_rho = c_squared * DENSITY
    inv_rho = 1.0 / DENSITY

    # Physical domain grid
    num_physical_points = grid_resolution + 1
    dx = 1.0 / grid_resolution
    dy = dx

    # Extended computational domain with PML padding
    nx_total = num_physical_points + 2 * PML_SIZE
    ny_total = num_physical_points + 2 * PML_SIZE

    # CFL condition
    cfl_dt = 0.3 * min(dx, dy) / (WAVE_SPEED * np.sqrt(2.0))
    num_time_steps = int(np.ceil(final_time / cfl_dt))
    dt = final_time / num_time_steps

    if not quiet:
        print(f"Physical grid resolution: {grid_resolution}, h = {dx:.4f}")
        print(f"Extended grid: {nx_total} x {ny_total}")
        print(f"dt = {dt:.6f}, num_steps = {num_time_steps}")

    # Output time steps
    output_time_values = np.linspace(0.0, final_time, num_output_steps)
    output_simulation_steps = (output_time_values / dt).astype(int)
    output_simulation_steps[0] = 0
    output_simulation_steps[-1] = num_time_steps

    # Exponential PML profiles (jwave-style)
    # Separate PML for velocity (staggered by 0.5) and pressure/density (at cell centers)
    pml_vx_x = compute_pml_exponential_1d(nx_total, PML_SIZE, dt, WAVE_SPEED, dx, coord_shift=0.5)
    pml_vy_y = compute_pml_exponential_1d(ny_total, PML_SIZE, dt, WAVE_SPEED, dy, coord_shift=0.5)
    pml_p_x = compute_pml_exponential_1d(nx_total, PML_SIZE, dt, WAVE_SPEED, dx, coord_shift=0.0)
    pml_p_y = compute_pml_exponential_1d(ny_total, PML_SIZE, dt, WAVE_SPEED, dy, coord_shift=0.0)

    # Source location in extended grid
    source_i = PML_SIZE + int(SOURCE_X / dx)
    source_j = PML_SIZE + int(SOURCE_Y / dy)

    # Initialize fields
    p = np.zeros((nx_total, ny_total))
    vx = np.zeros((nx_total, ny_total))
    vy = np.zeros((nx_total, ny_total))

    # Precompute constants
    inv_rho_dx = inv_rho / dx
    inv_rho_dy = inv_rho / dy
    c_sq_rho_dx = c_squared_rho / dx
    c_sq_rho_dy = c_squared_rho / dy

    # Node positions for physical domain output
    x_coords = np.linspace(0.0, 1.0, num_physical_points)
    y_coords = np.linspace(0.0, 1.0, num_physical_points)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')
    node_positions = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    # Output storage
    num_physical_nodes = num_physical_points * num_physical_points
    solution_values = np.zeros((num_output_steps, num_physical_nodes))

    # Store initial condition
    current_output_index = 0
    p_physical = p[PML_SIZE:PML_SIZE+num_physical_points,
                   PML_SIZE:PML_SIZE+num_physical_points]
    solution_values[current_output_index, :] = p_physical.ravel()
    current_output_index += 1

    # Source scaling factor for point source (matches warp_solver.py)
    source_scale = dt / dx

    # Time-stepping loop
    for step in range(num_time_steps):
        current_time = step * dt
        source_amplitude = ricker_wavelet(current_time) * source_scale

        # Step 1: Update vx (x-velocity on staggered grid) with jwave-style exponential PML
        # vx[i,j] is located at (i-0.5, j) - between p[i-1,j] and p[i,j]
        for i in range(1, nx_total):
            for j in range(ny_total):
                # Forward difference for dp/dx
                dp_dx = p[i, j] - p[i - 1, j]

                # Velocity time derivative: dvx/dt = -1/ρ * dp/dx
                dvx_dt = -inv_rho_dx * dp_dx

                # Exponential PML update (jwave-style)
                pml = pml_vx_x[i]
                vx[i, j] = pml * (pml * vx[i, j] + dt * dvx_dt)

        # Step 2: Update vy (y-velocity on staggered grid) with jwave-style exponential PML
        # vy[i,j] is located at (i, j-0.5) - between p[i,j-1] and p[i,j]
        for i in range(nx_total):
            for j in range(1, ny_total):
                # Forward difference for dp/dy
                dp_dy = p[i, j] - p[i, j - 1]

                # Velocity time derivative: dvy/dt = -1/ρ * dp/dy
                dvy_dt = -inv_rho_dy * dp_dy

                # Exponential PML update (jwave-style)
                pml = pml_vy_y[j]
                vy[i, j] = pml * (pml * vy[i, j] + dt * dvy_dt)

        # Step 3: Update pressure with jwave-style exponential PML
        # p[i,j] is at cell center
        # Divergence: div(v) = (vx[i+1,j] - vx[i,j])/dx + (vy[i,j+1] - vy[i,j])/dy
        for i in range(nx_total - 1):
            for j in range(ny_total - 1):
                # Divergence using staggered velocities
                dvx_dx = vx[i + 1, j] - vx[i, j]
                dvy_dy = vy[i, j + 1] - vy[i, j]

                # Pressure time derivative: dp/dt = -c²ρ * div(v)
                div_v = c_sq_rho_dx * dvx_dx + c_sq_rho_dy * dvy_dy
                dp_dt = -div_v

                # Combined PML (product of x and y components)
                pml = pml_p_x[i] * pml_p_y[j]

                # Exponential PML update (jwave-style)
                p[i, j] = pml * (pml * p[i, j] + dt * dp_dt)

        # Step 4: Add source
        p[source_i, source_j] += source_amplitude

        # Store output
        simulation_step_number = step + 1
        if current_output_index < num_output_steps and simulation_step_number == output_simulation_steps[current_output_index]:
            p_physical = p[PML_SIZE:PML_SIZE+num_physical_points,
                          PML_SIZE:PML_SIZE+num_physical_points]
            solution_values[current_output_index, :] = p_physical.ravel()
            current_output_index += 1

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
    return solve_wave_equation_2d(grid_resolution=grid_resolution, quiet=True)
