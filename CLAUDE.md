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
    └── heat/_2d/          # 2D Heat/diffusion equation (time-dependent)
        ├── warp.py        # Warp FEM solver (backward Euler)
        └── analytical.py  # Analytical solution
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

Solves `-∇²u - k²u = f` on [0,1]² with homogeneous Dirichlet BCs.

**Manufactured solution:** `u(x,y) = sin(πx)sin(πy)` with `k² = π²`

This gives:
- Source term: `f(x,y) = (2π² - k²)sin(πx)sin(πy) = π²sin(πx)sin(πy)`
- Boundary values: `u = 0`
- Peak value: `u(0.5, 0.5) = 1.0`

Expected convergence rate: ~2.0 for linear elements

### Heat Equation (2D) - Time-Dependent

Solves `u_t = κ∇²u` on [0,1]² with homogeneous Dirichlet BCs.

**Initial condition:** `u₀(x,y) = sin(πx)sin(πy)`

**Analytical solution:** `u(x,y,t) = exp(-2π²κt) sin(πx)sin(πy)`

Default parameters:
- Diffusivity: `κ = 0.1`
- Final time: `T = 1.0`
- Time steps: 1000 (backward Euler)

The solution decays exponentially in time. Expected spatial convergence rate: ~2.0 for linear elements.

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
