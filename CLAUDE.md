# CLAUDE.md - PDE Benchmark Codebase Guide

## Overview

This codebase benchmarks PDE solvers against analytical solutions for various physics domains. The framework measures accuracy via error norms and convergence rates. Currently supports [NVIDIA Warp](https://github.com/NVIDIA/warp)'s FEM solvers with plans to add FEniCS, Firedrake, and other packages.

## Project Structure

```
pde-benchmark/
├── pde_cli.py             # CLI entry point
├── CLAUDE.md              # This file - codebase guide and coding standards
├── README.md              # User-facing documentation
├── pyproject.toml         # Package configuration and dependencies
│
├── src/                   # Shared utilities
│   ├── metrics.py         # Error computation (L2, L∞, relative L2, MAE)
│   ├── results.py         # Results caching utilities
│   └── visualization.py   # Matplotlib-based visualizations
│
└── benchmarks/
    ├── poisson/_2d/       # 2D Poisson equation (static)
    │   ├── warp.py        # Warp FEM solver
    │   └── analytical.py  # Analytical solution
    │
    ├── laplace/_2d/       # 2D Laplace equation (static)
    │   ├── warp.py        # Warp FEM solver
    │   └── analytical.py  # Analytical solution
    │
    ├── helmholtz/_2d/     # 2D Helmholtz equation (static)
    │   ├── warp.py        # Warp FEM solver
    │   └── analytical.py  # Analytical solution
    │
    ├── heat/_2d/          # 2D Heat/diffusion equation (time-dependent)
    │   ├── warp.py        # Warp FEM solver (backward Euler)
    │   ├── dolfinx.py     # DOLFINx FEM solver (backward Euler)
    │   └── analytical.py  # Analytical solution (Fourier series)
    │
    ├── advection/_2d/     # 2D Linear advection equation (time-dependent)
    │   ├── warp.py        # Warp FEM solver (semi-Lagrangian)
    │   ├── dolfinx.py     # DOLFINx FEM solver (SUPG stabilization)
    │   └── analytical.py  # Analytical solution (translated Gaussian)
    │
    ├── convection_diffusion/_2d/  # 2D Convection-diffusion equation (time-dependent)
    │   ├── warp.py        # Warp FEM solver (semi-Lagrangian + implicit diffusion)
    │   ├── dolfinx.py     # DOLFINx FEM solver (SUPG + diffusion)
    │   └── analytical.py  # Analytical solution (spreading translated Gaussian)
    │
    └── wave/              # 2D Wave equation (time-dependent, second-order hyperbolic)
        ├── manufactured/_2d/  # Standing wave with Neumann boundaries
        │   ├── warp.py        # Warp FEM solver (Newmark-beta)
        │   ├── dolfinx.py     # DOLFINx FEM solver (Newmark-beta)
        │   └── analytical.py  # Analytical solution (exact standing wave)
        │
        └── ricker/_2d/        # Point source with absorbing boundaries
            ├── warp.py        # Warp FEM solver (Newmark-beta + damping sponge)
            ├── dolfinx.py     # DOLFINx FEM solver (Newmark-beta + damping sponge)
            └── analytical.py  # Semi-analytical (Green's function convolution)
```

## CLI Usage

The primary interface is the `pde` command (after activating the conda environment):

```bash
# List available solvers and cached results
pde list

# Run a solver at multiple resolutions
pde run benchmarks/poisson/_2d/warp.py --resolution 8 16 32 64

# Run multiple solvers
pde run benchmarks/poisson/_2d/warp.py benchmarks/poisson/_2d/analytical.py --resolution 8 16 32 64

# Compare solvers (all pairs) - prints error metrics
pde compare benchmarks/poisson/_2d/warp.py benchmarks/poisson/_2d/analytical.py --resolution 32

# Plot solutions side by side
pde plot benchmarks/poisson/_2d/warp.py benchmarks/poisson/_2d/analytical.py --resolution 32

# Plot with pairwise comparisons and error visualization
pde plot benchmarks/poisson/_2d/warp.py benchmarks/poisson/_2d/analytical.py --resolution 32 --compare

# Plot with convergence analysis (requires 2+ resolutions)
pde plot benchmarks/poisson/_2d/warp.py benchmarks/poisson/_2d/analytical.py --resolution 8 16 32 64 --convergence

# Show plots interactively
pde plot benchmarks/poisson/_2d/warp.py benchmarks/poisson/_2d/analytical.py --resolution 32 --show

# Generate animated GIF for time-dependent PDEs (heat equation)
pde plot benchmarks/heat/_2d/warp.py benchmarks/heat/_2d/analytical.py --resolution 32 --video
```

### CLI Flags Reference

| Flag | Commands | Description |
|------|----------|-------------|
| `--resolution, -r` | all | Grid resolution(s) to use |
| `--output, -o` | all | Output directory (default: `results`) |
| `--force, -f` | all | **Force recomputation, ignoring cached results** |
| `--compare, -c` | plot | Show pairwise comparisons with error visualization |
| `--convergence` | plot | Generate convergence plot (requires 2+ resolutions) |
| `--show` | plot | Display plots interactively |
| `--video` | plot | Generate animated GIF (time-dependent PDEs only) |

**IMPORTANT:** Always use `--force` when testing code changes to ensure fresh results are computed instead of using stale cached data.

## Coding Standards

### 1. Procedural Programming Only

**Avoid classes at all costs unless absolutely necessary.**

Instead of:
```python
class TimingContext:
    def __init__(self):
        self.start_time = None
    def start(self):
        self.start_time = time.perf_counter()
```

Use:
```python
def create_timing_context(synchronize_gpu=False):
    return {"start_time_seconds": None, "synchronize_gpu": synchronize_gpu}

def start_timing(timing_context):
    timing_context["start_time_seconds"] = time.perf_counter()
```

### 2. Descriptive Variable and Function Names

Names should be self-documenting. Length is acceptable if it improves clarity.

**Bad:**
```python
def compute_err(num, ana):
    return np.linalg.norm(num - ana)
```

**Good:**
```python
def compute_l2_error_norm_between_fields(
    numerical_field_values: np.ndarray,
    analytical_field_values: np.ndarray
) -> float:
    difference_between_fields = numerical_field_values - analytical_field_values
    return float(np.linalg.norm(difference_between_fields))
```

### 3. Naming Conventions

- **Functions**: Use verb phrases describing the action
  - `compute_magnetic_field_for_infinite_wire()`
  - `create_timing_context()`
  - `apply_homogeneous_dirichlet_boundary_conditions()`

- **Variables**: Include units or context where relevant
  - `wire_current_amperes`
  - `domain_half_size_meters`
  - `solve_time_milliseconds`
  - `radial_distance_from_wire`

- **Constants**: All caps with units
  - `VACUUM_PERMEABILITY_HENRIES_PER_METER`

### 4. Function Documentation

Include docstrings with:
- Brief description of what the function does
- Args section with parameter descriptions and units
- Returns section describing output

```python
def compute_magnetic_field_magnitude_for_infinite_wire(
    wire_current_amperes: float,
    radial_distances_meters: np.ndarray
) -> np.ndarray:
    """Compute magnetic field magnitude at given radial distances from an infinite wire.

    Uses the simple formula: |B| = (μ₀ I) / (2π r)

    Args:
        wire_current_amperes: Current in the wire [Amperes]
        radial_distances_meters: Radial distances from wire axis [meters]

    Returns:
        Magnetic field magnitudes [Tesla]
    """
```

## Unified Solver Interface

All solvers (`warp.py`, `fenics.py`, `analytical.py`, etc.) must implement:

```python
def solve(grid_resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the problem at given resolution.

    Args:
        grid_resolution: Number of cells in each dimension

    Returns:
        Tuple of (solution_values, node_positions)
        - solution_values: shape (N,) array of values at nodes
        - node_positions: shape (N, 2) array of (x, y) coordinates
    """
```

## Physics Background

### Poisson Equation (2D)

Solves `-∇²u = f` on [0,1]² with homogeneous Dirichlet BCs.

**Manufactured solution:** `u(x,y) = sin(πx)sin(πy)`

This gives:
- Source term: `f(x,y) = 2π²sin(πx)sin(πy)`
- Boundary values: `u = 0` (since sin(0) = sin(π) = 0)
- Peak value: `u(0.5, 0.5) = 1.0`

Expected convergence rate: ~2.0 for linear elements

### Laplace Equation (2D)

Solves `∇²u = 0` on [0,1]² with non-homogeneous Dirichlet BCs.

**Boundary conditions:**
- Top (y=1): `u = sin(πx)`
- Other boundaries: `u = 0`

**Analytical solution:** `u(x,y) = sin(πx) · sinh(πy) / sinh(π)`

Expected convergence rate: ~2.0 for linear elements

### Helmholtz Equation (2D)

Solves the homogeneous Helmholtz equation `-∇²u - k₀²u = 0` on [0,1]² with mixed boundary conditions.

**Plane wave solution:** `u(x,y) = A·cos(k₀(cos(θ)x + sin(θ)y))`

This represents acoustic/electromagnetic wave propagation at angle θ, based on Ihlenburg's "Finite Element Analysis of Acoustic Scattering" (p138-139).

Parameters:
- Wavenumber: `k₀ = 4π` (~12.57)
- Propagation angle: `θ = π/4` (45°)
- Amplitude: `A = 1.0`
- Wavelength: `λ = 2π/k₀ = 0.5` (2 wavelengths across domain)

**Mixed boundary conditions:**
- Dirichlet: `u = u_exact` on x=0 and y=0
- Neumann: `∂u/∂n = g` on x=1 and y=1, where `g = -∇u_exact · n`

**Resolution requirements:** The "rule of thumb" is ~10 elements per wavelength. With λ = 0.5, minimum resolution ~20 for reasonable accuracy. Higher resolutions (64+) recommended for convergence studies. Pollution error is evident at coarse resolutions.

Expected convergence rate: ~2.0 for linear elements

### Heat Equation (2D) - Time-Dependent

Solves `u_t = κ∇²u` on [0,1]² with homogeneous Dirichlet BCs (u = 0 on all boundaries).

**Initial condition:** Gaussian blob at center
`u₀(x,y) = A·exp(-((x-0.5)² + (y-0.5)²) / σ²)`

The Gaussian creates a concentrated heat source at the center that diffuses outward and is absorbed at the boundaries.

**Analytical solution:** Double Fourier sine series
`u(x,y,t) = Σₘ Σₙ Bₘₙ exp(-π²κ(m² + n²)t) sin(mπx) sin(nπy)`

where `Bₘₙ = 4 ∫∫ u₀(x,y) sin(mπx) sin(nπy) dx dy` are the Fourier coefficients computed via numerical quadrature.

Default parameters:
- Diffusivity: `κ = 0.01` (moderate diffusivity)
- Gaussian width: `σ = 0.1` (well-resolved at typical grid resolutions)
- Amplitude: `A = 1.0`
- Final time: `T = 0.5` (enough time to see significant spreading)
- Time steps: 1000 (backward Euler)
- Output steps: 51 (for smooth animation)
- Fourier modes: 50 (sufficient for moderate-width Gaussian)

The heat diffuses radially from the center and is absorbed at the boundaries. Expected spatial convergence rate: ~2.0 for linear elements.

### Linear Advection Equation (2D) - Time-Dependent

Solves `∂u/∂t + c⃗·∇u = 0` on [0,1]² - pure transport with no diffusion.

**Initial condition:** Gaussian blob at (0.3, 0.3)
`u₀(x,y) = A·exp(-((x-x₀)² + (y-y₀)²) / σ²)`

**Velocity field:** Constant diagonal transport `c⃗ = (0.4, 0.4)`

**Analytical solution:** Translated Gaussian
`u(x,y,t) = u₀(x - cx·t, y - cy·t)`

The Gaussian blob moves diagonally across the domain without changing shape.

**Boundary conditions:**
- Inflow (x=0, y=0 where c⃗·n < 0): `u = 0` (Dirichlet)
- Outflow (x=1, y=1 where c⃗·n > 0): Natural BC (free exit)

**Numerical methods:**
- Warp: Semi-Lagrangian advection (trace particles backward along characteristics)
- DOLFINx: SUPG (Streamline Upwind Petrov-Galerkin) stabilization

Default parameters:
- Velocity: `c⃗ = (0.4, 0.4)` (diagonal transport)
- Gaussian center: `(x₀, y₀) = (0.3, 0.3)`
- Gaussian width: `σ = 0.1`
- Amplitude: `A = 1.0`
- Final time: `T = 1.0` (blob moves to (0.7, 0.7))
- Time steps: 1000
- Output steps: 51

Expected spatial convergence rate: ~2.0 for linear elements with appropriate stabilization.

### Convection-Diffusion Equation (2D) - Time-Dependent

Solves `∂u/∂t + c⃗·∇u = κ∇²u` on [0,1]² - combines advection (transport) with diffusion (spreading).

**Initial condition:** Gaussian blob at (0.3, 0.3)
`u₀(x,y) = A·exp(-((x-x₀)² + (y-y₀)²) / σ²)`

**Velocity field:** Constant diagonal transport `c⃗ = (0.4, 0.4)`

**Analytical solution:** Translating and spreading Gaussian
`u(x,y,t) = A·σ²/(σ² + 4κt) · exp(-((x-x₀-cx·t)² + (y-y₀-cy·t)²)/(σ² + 4κt))`

The Gaussian:
- Translates with velocity c⃗ (advection)
- Spreads with effective width σ_eff(t) = √(σ² + 4κt) (diffusion)
- Amplitude decreases to conserve mass

**Boundary conditions:** Homogeneous Neumann (natural BC) - flux vanishes at boundaries.

**Numerical methods:**
- Warp: Semi-Lagrangian advection + implicit backward Euler diffusion
- DOLFINx: SUPG stabilization for advection + standard Galerkin for diffusion

**Péclet number:** Pe = |c|h/(2κ) determines advection vs diffusion dominance.
With κ = 0.01 and h ~ 1/32, Pe ≈ 0.9 (mixed regime).

Default parameters:
- Velocity: `c⃗ = (0.4, 0.4)` (diagonal transport)
- Diffusivity: `κ = 0.01` (moderate spreading)
- Gaussian center: `(x₀, y₀) = (0.3, 0.3)`
- Gaussian width: `σ = 0.1`
- Amplitude: `A = 1.0`
- Final time: `T = 1.0` (blob moves to ~(0.7, 0.7) while spreading)
- Time steps: 1000
- Output steps: 51

Expected spatial convergence rate: ~2.0 for linear elements.

### Wave Equation (2D) - Time-Dependent (Second-Order Hyperbolic)

Solves `∂²u/∂t² = c²∇²u` on [0,1]² - acoustic/elastic wave propagation.

**Two benchmark problems are provided:**

#### 1. Manufactured Solution (Standing Wave)

**Equation:** `∂²u/∂t² = c²∇²u` with homogeneous Neumann boundaries

**Solution:** Standing wave
`u(x,y,t) = cos(2πmx)·cos(2πny)·cos(ωt)`

where `ω = c·2π√(m² + n²)` satisfies the dispersion relation.

This is an exact solution - perfect for convergence testing.

Default parameters:
- Wave speed: `c = 1.0`
- Mode numbers: `m = n = 2` (2 wavelengths in each direction)
- Angular frequency: `ω = 2π√8 ≈ 17.77`
- Final time: `T = 1.0` (~2.8 oscillation periods)
- Time steps: 2000
- Output steps: 101

**Numerical methods:**
- Warp: Newmark-beta (β=0.25, γ=0.5) - unconditionally stable
- DOLFINx: Newmark-beta (β=0.25, γ=0.5) - unconditionally stable

#### 2. Ricker Wavelet with Absorbing Boundaries

**Equation:** `∂²u/∂t² + σ(x,y)·∂u/∂t = c²∇²u + f(x,y,t)` with damping sponge layer

**Source:** Point source at center (0.5, 0.5) with Ricker wavelet time signature
`S(t) = (1 - 2·(π·f₀·(t-t_delay))²) · exp(-(π·f₀·(t-t_delay))²)`

**Damping sponge layer:** σ(x,y) increases quadratically from 0 in physical domain [0.15, 0.85]² to σ_max at outer boundary. This absorbs outgoing waves.

**Semi-analytical solution:** Green's function convolution (computed numerically)

Default parameters:
- Wave speed: `c = 1.0`
- Center frequency: `f₀ = 15.0 Hz` (~10 wavelengths across domain)
- Source delay: `t_delay = 0.1 s`
- Sponge thickness: `0.15` (15% on each side)
- Maximum damping: `σ_max = 20.0`
- Final time: `T = 1.0`
- Time steps: 2000
- Output steps: 101

**Numerical methods:**
- Warp: Newmark-beta with spatially-varying damping matrix
- DOLFINx: Newmark-beta with spatially-varying damping matrix

**Note:** The semi-analytical solution can be slow due to numerical convolution at each grid point.

Expected spatial convergence rate: ~2.0 for linear elements.

## Key Components

### benchmarks/poisson/_2d/analytical.py
Manufactured solution functions:
- `compute_analytical_solution(x_coordinates, y_coordinates)` - Returns sin(πx)sin(πy)
- `compute_source_term(x_coordinates, y_coordinates)` - Returns 2π²sin(πx)sin(πy)
- `compute_analytical_solution_at_points(points)` - Evaluate at (N,2) array
- `solve(grid_resolution)` - Unified interface for CLI

### benchmarks/poisson/_2d/warp.py
Warp FEM Poisson solver:
- `solve_poisson_2d(grid_resolution, polynomial_degree, quiet)` - Main entry point
- `solve(grid_resolution)` - Unified interface for CLI
- Returns `(solution_values, node_positions)` as numpy arrays
- Uses Grid2D geometry, Lagrange elements, CG solver

### benchmarks/laplace/_2d/analytical.py
Laplace analytical solution:
- `compute_analytical_solution(x_coordinates, y_coordinates)` - Returns sin(πx)·sinh(πy)/sinh(π)
- `compute_analytical_solution_at_points(points)` - Evaluate at (N,2) array
- `compute_top_boundary_values(x_coordinates)` - Returns sin(πx) for top boundary
- `solve(grid_resolution)` - Unified interface for CLI

### benchmarks/laplace/_2d/warp.py
Warp FEM Laplace solver with non-homogeneous Dirichlet BCs:
- `solve_laplace_2d(grid_resolution, polynomial_degree, quiet)` - Main entry point
- `solve(grid_resolution)` - Unified interface for CLI

### benchmarks/helmholtz/_2d/analytical.py
Plane wave analytical solution:
- `compute_phase_at_coordinates(x, y)` - Computes φ = k₀(cos(θ)x + sin(θ)y)
- `compute_analytical_solution(x, y)` - Returns A·cos(φ)
- `compute_gradient_x_component(x, y)` - Returns ∂u/∂x
- `compute_gradient_y_component(x, y)` - Returns ∂u/∂y
- `compute_neumann_boundary_flux(x, y, nx, ny)` - Returns g = -∇u · n
- `solve(grid_resolution)` - Unified interface for CLI

### benchmarks/helmholtz/_2d/warp.py
Warp FEM Helmholtz solver with mixed BCs (Dirichlet on x=0, y=0; Neumann on x=1, y=1):
- `solve_helmholtz_2d(grid_resolution, polynomial_degree, quiet)` - Main entry point
- `solve(grid_resolution)` - Unified interface for CLI
- Uses selective boundary projector for mixed BCs
- Neumann BC via surface integral in weak form

### src/results.py
Results caching utilities:
- `save_result(results_directory, solver_name, grid_resolution, solution_values, node_positions, metadata)` - Save to `<solver>_res032.npz`
- `load_result(results_directory, solver_name, grid_resolution)` - Returns `(solution_values, node_positions, metadata)`
- `result_exists(results_directory, solver_name, grid_resolution)` - Check if cached result exists
- `list_cached_resolutions_for_solver(results_directory, solver_name)` - List resolutions for a solver
- `list_all_cached_results(results_directory)` - Dict of solver → resolutions

### src/metrics.py
Error computation utilities:
- `compute_l2_error_norm(numerical, analytical, quadrature_weights)` - L2 norm of difference
- `compute_l_infinity_error_norm(numerical, analytical)` - L∞ norm of difference
- `compute_relative_l2_error_norm(numerical, analytical, quadrature_weights)` - Relative L2 error
- `compute_mean_absolute_error(numerical, analytical, quadrature_weights)` - Mean absolute error
- `compute_all_error_metrics(numerical, analytical, quadrature_weights)` - Returns dict with all metrics

### src/visualization.py
Visualization utilities:
- `create_convergence_plot(mesh_size_values, error_values, measured_convergence_rate, ...)` - Log-log convergence plot
- `create_solution_subplots(solutions, ...)` - Side-by-side solution visualization (1D/2D/3D)
- `create_comparison_subplots(solution_pairs, ...)` - Pairwise comparisons with error visualization
- `create_solution_comparison_from_points(node_positions, numerical, analytical, ...)` - Legacy side-by-side comparison
- `determine_problem_dimension(node_positions)` - Detect spatial dimension from node data
- `save_figure(figure, output_path, dpi)` - Save matplotlib figure
- `show_figure(figure)` - Display matplotlib figure

## Adding New Solvers

To add a new solver (e.g., FEniCS):

1. Create `fenics.py` in the benchmark directory
2. Implement the `solve(grid_resolution)` function
3. Use the CLI: `pde run benchmarks/poisson/_2d/fenics.py --resolution 32`

## Adding New Benchmarks

1. Create a new folder under `benchmarks/<domain>/_<dimension>/`
2. Add `warp.py` with procedural Warp FEM solver implementing `solve()`
3. Add `analytical.py` with closed-form solution implementing `solve()`

## Results Caching

Results are automatically cached to `<benchmark>/results/` as `.npz` files:
- `warp_res032.npz` - Warp solution at resolution 32
- `analytical_res032.npz` - Analytical solution at resolution 32

Use `--force` to recompute cached results.

## Dependencies

Install with conda:

```bash
conda create -n pde-benchmarks python=3.13 fenics-dolfinx mpich pyvista numpy matplotlib scipy -c conda-forge
conda activate pde-benchmarks
pip install warp-lang
pip install -e .
```

- `fenics-dolfinx` - FEniCS DOLFINx for FEM
- `warp-lang` - NVIDIA Warp for GPU-accelerated FEM
- `numpy` - Array operations
- `matplotlib` - Visualizations
- `scipy` - Scientific computing utilities
- `pyvista` - 3D visualization
- `mpich` - MPI implementation

## Environment Setup (IMPORTANT)

**Always use the conda environment when running or debugging code.**

Before running any Python scripts or commands:

```bash
# Activate the conda environment (from project root)
conda activate pde-benchmarks

# Run Python files
python <script.py>

# Run the CLI
pde <command>

# Add new conda packages
conda install -c conda-forge <package-name>

# Add new pip packages
pip install <package-name>

# Update environment
conda env update -f environment.yml
```

**Never run Python without activating the conda environment first.** All debugging, script execution, and package management must go through the conda environment.

## Reference Source Code (IMPORTANT)

When implementing solvers, **always consult the local source code repositories** for accurate API usage, examples, and available features:

### NVIDIA Warp

**Location:** `/Users/thomasribeiro/code/warp`

Before writing any Warp FEM code, explore this repository to understand:
- Available solvers in `warp/optim/linear.py` (CG, BiCGSTAB, GMRES, CR)
- FEM utilities in `warp/fem/` directory
- Example implementations in `warp/examples/fem/`
- The `bsr_cg` utility in `warp/examples/fem/utils.py` supports multiple solvers via `method` parameter

**Key insight:** Warp has built-in iterative solvers for indefinite systems:
```python
from warp.optim.linear import gmres, bicgstab

# For indefinite matrices (like Helmholtz with -k² term), use GMRES:
gmres(A=matrix, b=rhs, x=solution, tol=1e-8, maxiter=5000)

# Or BiCGSTAB for non-symmetric systems:
bicgstab(A=matrix, b=rhs, x=solution, tol=1e-8, maxiter=5000)
```

### FEniCS DOLFINx

**Location:** `/Users/thomasribeiro/code/dolfinx`

Before writing any DOLFINx/FEniCSx code, explore this repository to understand:
- Python API in `python/dolfinx/`
- FEM module in `python/dolfinx/fem/`
- Example demos in `python/demo/`
- PETSc solver interfaces

**Always check these repositories first** rather than guessing API usage or relying on outdated documentation.
