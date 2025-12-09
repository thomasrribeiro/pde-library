# PDE Benchmark

Benchmarking framework for comparing PDE solvers against analytical solutions.

## Setup

```bash
# Create conda environment from environment.yml
conda env create -f environment.yml

# Install the package in editable mode
conda activate pde-benchmarks
pip install -e .

# Quick activation (alternative)
source activate.sh
```

<details>
<summary>Manual setup (if needed)</summary>

```bash
conda create -n pde-benchmarks python=3.13 fenics-dolfinx mpich pyvista numpy matplotlib scipy -c conda-forge
conda activate pde-benchmarks
pip install warp-lang
pip install -e .
```
</details>

## CLI Usage

The primary interface is the `pde` command:

```bash
# List available solvers and cached results
pde list

# Run solver(s) at multiple resolutions
pde run path/to/solver1.py path/to/solver2.py --resolution 8 16 32 64

# Compare solvers (all pairs) - prints error metrics
pde compare path/to/solver1.py path/to/solver2.py --resolution 32

# Plot solutions side by side
pde plot path/to/solver1.py path/to/solver2.py --resolution 32

# Plot with pairwise comparisons and error visualization
pde plot path/to/solver1.py path/to/solver2.py --resolution 32 --compare

# Plot with convergence analysis (requires 2+ resolutions)
pde plot path/to/solver1.py path/to/solver2.py --resolution 8 16 32 64 --convergence

# Show plots interactively
pde plot path/to/solver1.py path/to/solver2.py --resolution 32 --show
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `list` | List available solvers and cached results |
| `run` | Run solver(s) at specified resolutions |
| `compare` | Compare all pairs of solvers (prints error metrics) |
| `plot` | Visualize solutions with optional comparison and convergence plots |

### Common Arguments and Flags

- `solver`: Solver file path(s) - positional argument (e.g., `benchmarks/poisson/_2d/warp.py`)
- `--output`, `-o`: Output directory for results (default: `results/`)
- `--resolution`, `-r`: Grid resolution(s) to run
- `--force`, `-f`: Force recomputation even if cached results exist
- `--compare`, `-c`: Show pairwise comparisons with error visualization (plot command)
- `--convergence`: Generate convergence plot (plot command, requires 2+ solvers and resolutions)
- `--show`: Display plots interactively

## Project Structure

```
pde-benchmark/
├── pde_cli.py             # CLI entry point
├── README.md              # This file
├── CLAUDE.md              # Codebase guide and coding standards
├── pyproject.toml         # Package configuration and dependencies
│
├── src/                   # Shared utilities
│   ├── metrics.py         # Error computation (L2, L∞, relative, MAE)
│   ├── results.py         # Results caching utilities
│   └── visualization.py   # Matplotlib-based visualizations
│
└── benchmarks/
    ├── poisson/_2d/       # 2D Poisson equation
    │   ├── warp.py        # Warp FEM solver
    │   └── analytical.py  # Analytical solution
    │
    └── laplace/_2d/       # 2D Laplace equation
        ├── warp.py        # Warp FEM solver
        └── analytical.py  # Analytical solution
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

## Results Caching

Results are automatically cached to `<benchmark>/results/` as `.npz` files:
- `warp_res032.npz` - Warp solution at resolution 32
- `analytical_res032.npz` - Analytical solution at resolution 32

Use `--force` to recompute cached results.

## License

MIT
