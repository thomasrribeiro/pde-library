# CLAUDE.md - Warp Benchmark Codebase Guide

## Overview

This codebase benchmarks NVIDIA Warp's finite element method (FEM) solvers against analytical solutions for various physics domains. The framework measures accuracy via error norms, convergence rates, and timing performance.

## Project Structure

```
warp-benchmark/
├── CLAUDE.md              # This file - codebase guide and coding standards
├── README.md              # User-facing documentation
├── requirements.txt       # Python dependencies (install with: uv pip install -r requirements.txt)
├── src/                   # Reusable utility modules
│   ├── timer.py           # GPU-synchronized timing utilities
│   ├── metrics.py         # Error computation (L2, L∞, relative errors)
│   ├── convergence.py     # Mesh refinement convergence analysis
│   └── visualization.py   # Matplotlib-based visualizations
└── benchmarks/            # Physics domain benchmarks
    └── poisson/           # Poisson equation benchmarks
        ├── analytical.py  # Manufactured solution u(x,y) = sin(πx)sin(πy)
        ├── solver.py      # Warp FEM solver (procedural)
        └── poisson_2d.py  # Main benchmark script
```

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

## Physics Background

### Poisson Equation

The 2D Poisson equation benchmark solves:

```
-∇²u = f    on [0,1]²
u = 0       on boundary (Dirichlet BC)
```

Using manufactured solution `u(x,y) = sin(πx)sin(πy)`, which gives:
- Source term: `f(x,y) = 2π²sin(πx)sin(πy)`
- Boundary values: `u = 0` (since sin(0) = sin(π) = 0)
- Peak value: `u(0.5, 0.5) = 1.0`

This is ideal for verification because:
- Scalar Lagrange elements (simpler than vector Nedelec)
- Positive definite Laplacian (CG converges reliably)
- Exact analytical solution for error measurement

## Key Components

### benchmarks/poisson/analytical.py
Manufactured solution functions:
- `compute_analytical_solution(x, y)` - Returns sin(πx)sin(πy)
- `compute_source_term(x, y)` - Returns 2π²sin(πx)sin(πy)
- `compute_analytical_solution_at_points(points)` - Evaluate at (N,2) array

### benchmarks/poisson/solver.py
Warp FEM Poisson solver:
- `solve_poisson_2d(grid_resolution, polynomial_degree, quiet)` - Main entry point
- Returns `(solution_values, node_positions)` as numpy arrays
- Uses Grid2D geometry, Lagrange elements, CG solver

### benchmarks/poisson/poisson_2d.py
Main benchmark script:
- Single resolution run with error analysis
- Convergence study across multiple resolutions
- Prints L2, L∞, and relative errors

## Adding New Benchmarks

1. Create a new folder under `benchmarks/` for the physics domain
2. Add `analytical.py` with closed-form or manufactured solutions
3. Add `solver.py` with procedural Warp FEM solver functions
4. Add a main script (e.g., `problem_name.py`) with:
   - Problem parameters
   - Solver execution
   - Analytical solution computation
   - Error analysis (L2, L∞, relative errors)
   - Optional: convergence study

## Dependencies

Install with: `uv pip install -r requirements.txt`

- `warp-lang` - NVIDIA Warp for GPU-accelerated FEM
- `numpy` - Array operations
- `matplotlib` - Visualizations
- `scipy` - Scientific computing utilities

## Running Benchmarks

From the project root directory:

```bash
# Single resolution run
uv run python benchmarks/poisson/poisson_2d.py

# Specify resolution
uv run python benchmarks/poisson/poisson_2d.py --resolution 64

# Run convergence study
uv run python benchmarks/poisson/poisson_2d.py --convergence
```
