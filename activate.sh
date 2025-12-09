#!/bin/bash
# Activation helper script for pde-benchmarks
# Usage: source activate.sh

# Initialize conda for this shell
eval "$(conda shell.bash hook)"

# Activate the environment
conda activate pde-benchmarks

echo "✓ pde-benchmarks conda environment activated"
echo "  - DOLFINx and Warp are ready to use"
echo "  - Run 'pde list' to see available solvers"
