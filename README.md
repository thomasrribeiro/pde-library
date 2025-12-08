# PDE Benchmark

Benchmarking framework for comparing PDE solvers against analytical solutions. Currently supports [NVIDIA Warp](https://github.com/NVIDIA/warp)'s FEM solvers with plans to add FEniCS, Firedrake, and other packages.

## Overview

This project provides tools for validating and benchmarking PDE solvers across various physics domains:

- **Poisson Equation** (2D) - Manufactured solution benchmark
- **Laplace Equation** (2D) - Non-homogeneous Dirichlet BC benchmark
- More coming soon...

## Setup

```bash
# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## CLI Usage

The primary interface is `pde_cli.py`:

```bash
# List available benchmarks and cached results
python pde_cli.py list

# Run a solver at multiple resolutions
python pde_cli.py run benchmarks/poisson/_2d --solver warp --resolution 8 16 32 64

# Run both warp and analytical solutions
python pde_cli.py run benchmarks/poisson/_2d --solver warp analytical --resolution 8 16 32 64

# Compare solvers against analytical solution
python pde_cli.py compare benchmarks/poisson/_2d --solver warp --reference analytical --resolution 32

# Generate convergence plot
python pde_cli.py plot-convergence benchmarks/poisson/_2d --solver warp --reference analytical

# Show plot interactively
python pde_cli.py plot-convergence benchmarks/poisson/_2d --solver warp --reference analytical --show
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `list` | List available benchmarks and cached results |
| `run` | Run solver(s) at specified resolutions |
| `compare` | Compare solver(s) against a reference solution |
| `plot-convergence` | Generate convergence plot from cached results |

### Common Flags

- `--solver`, `-s`: Name(s) of solver modules to use (e.g., `warp`, `analytical`)
- `--reference`, `-ref`: Reference solver for comparison (e.g., `analytical`)
- `--resolution`, `-r`: Grid resolution(s) to run
- `--force`, `-f`: Force recomputation even if cached results exist
- `--show`: Display plot interactively (for plot-convergence)

## Project Structure

```
pde-benchmark/
├── pde_cli.py             # CLI entry point
├── README.md              # This file
├── CLAUDE.md              # Codebase guide and coding standards
├── requirements.txt       # Python dependencies
│
├── src/                   # Shared utilities
│   ├── metrics.py         # Error computation (L2, L∞, relative, MAE)
│   ├── results.py         # Results caching utilities
│   └── visualization.py   # Matplotlib-based visualizations
│
└── benchmarks/
    ├── poisson/_2d/       # 2D Poisson equation
    │   ├── warp.py        # Warp FEM solver
    │   ├── analytical.py  # Analytical solution
    │   └── main.py        # Legacy standalone script (deprecated)
    │
    └── laplace/_2d/       # 2D Laplace equation
        ├── warp.py        # Warp FEM solver
        ├── analytical.py  # Analytical solution
        └── main.py        # Legacy standalone script (deprecated)
```

## Adding New Solvers

Each solver must implement a `solve()` function:

```python
def solve(grid_resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the PDE at the given resolution.

    Args:
        grid_resolution: Number of cells in each dimension

    Returns:
        Tuple of (solution_values, node_positions)
        - solution_values: shape (N,) array of values at nodes
        - node_positions: shape (N, 2) array of (x, y) coordinates
    """
```

To add a new solver (e.g., FEniCS):
1. Create `fenics.py` in the benchmark directory
2. Implement the `solve(grid_resolution)` function
3. Use the CLI: `python pde_cli.py run benchmarks/poisson/_2d --solver fenics --resolution 32`

## Results Caching

Results are automatically cached to `<benchmark>/results/` as `.npz` files:
- `warp_res032.npz` - Warp solution at resolution 32
- `analytical_res032.npz` - Analytical solution at resolution 32

Use `--force` to recompute cached results.

## Benchmarks

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

## License

MIT
