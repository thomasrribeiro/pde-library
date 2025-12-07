# Warp Benchmark

Benchmarking framework for comparing [NVIDIA Warp](https://github.com/NVIDIA/warp)'s differential equation solvers against analytical solutions.

## Overview

This project provides tools for validating and benchmarking Warp's FEM solvers across various physics domains:

- **Magnetostatics** (implemented) - Infinite wire B-field
- Acoustic waves (planned)
- Thermal diffusion (planned)
- Electromagnetics (planned)
- Fluid dynamics (planned)

## Setup

```bash
# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

**Note**: Requires NVIDIA Warp to be installed. The benchmarks reference the Warp codebase at `/Users/thomasribeiro/code/warp` for FEM utilities.

## Usage

### Interactive Notebook

The primary interface is an interactive Python notebook (VS Code / PyCharm compatible):

```bash
# Open in VS Code and run cells interactively
code benchmarks/magnetostatics/infinite_wire.py
```

The notebook uses `#%%` cell markers for interactive execution.

### Running the Full Benchmark

```bash
python benchmarks/magnetostatics/infinite_wire.py
```

## Project Structure

```
warp-benchmark/
├── README.md
├── requirements.txt
│
├── src/                          # Shared utilities
│   ├── timer.py                  # Timing with GPU synchronization
│   ├── metrics.py                # Error norms (L2, Linf, relative)
│   ├── convergence.py            # Convergence rate analysis
│   └── visualization.py          # Plotly-based visualizations
│
└── benchmarks/
    └── magnetostatics/
        ├── analytical.py         # Analytical B-field solutions
        ├── solver.py             # Warp FEM solver wrapper
        └── infinite_wire.py      # Interactive benchmark notebook
```

## Magnetostatics Benchmark

### Problem: Infinite Current-Carrying Wire

The first benchmark validates Warp's magnetostatics solver against the analytical solution for an infinite wire:

**Analytical Solution:**
```
B = (μ₀ I) / (2π r)  in azimuthal direction
```

Where:
- μ₀ = 4π × 10⁻⁷ H/m (vacuum permeability)
- I = current [Amperes]
- r = radial distance from wire [meters]

### What the Benchmark Measures

1. **Accuracy**: L2 and L∞ error norms vs analytical solution
2. **Convergence Rate**: Error reduction with mesh refinement (expect O(h²))
3. **Performance**: Solve time vs resolution

### Expected Results

For linear Nedelec elements:
- Convergence rate ≈ 2.0 (second-order)
- Error reduces by ~4× when resolution doubles

## Utilities

### Timing

```python
from src.timer import TimingContext

with TimingContext("solve", synchronize=True) as timer:
    result = solve()
print(f"Elapsed: {timer.elapsed_ms:.2f} ms")
```

### Error Metrics

```python
from src.metrics import compute_errors

errors = compute_errors(numerical, analytical)
print(f"L2: {errors['L2']}, Linf: {errors['Linf']}")
```

### Visualization

All plots use Plotly for interactive, web-ready visualizations:

```python
from src.visualization import plot_convergence

fig = plot_convergence(h_values, errors, convergence_rate=2.0)
fig.show()
```

## Future Plans

- Web dashboard for interactive visualization
- Support for additional physics domains
- Comparison with other solvers (FEniCS, etc.)
- GPU performance profiling

## License

MIT
