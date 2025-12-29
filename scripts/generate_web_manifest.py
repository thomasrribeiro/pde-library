#!/usr/bin/env python3
"""Generate manifest.json for web app listing all available benchmark results.

This script scans the benchmarks directory for results/ folders and creates
a manifest that the web app uses to:
1. Only show problems that have computed results
2. Know which solvers and resolutions are available
3. Auto-select the best resolution (most solvers, then highest resolution)

Run this script after computing new results and commit the manifest.json to git.

Usage:
    python scripts/generate_web_manifest.py
"""

import json
import re
from pathlib import Path


def parse_result_filename(filename: str) -> dict | None:
    """Parse a result filename like 'warp_solver_res032.npz'.

    Returns dict with solver name and resolution, or None if not a valid result file.
    """
    # Match pattern: {solver}_solver_res{NNN}.npz or {solver}_res{NNN}.npz
    match = re.match(r'^(.+?)(?:_solver)?_res(\d+)\.npz$', filename)
    if match:
        solver_name = match.group(1)
        resolution = int(match.group(2))
        return {'solver': solver_name, 'resolution': resolution}
    return None


def scan_benchmark_results(benchmarks_dir: Path) -> dict:
    """Scan benchmarks directory and build manifest of available results.

    Returns:
        Dict with structure:
        {
            "equations": {
                "laplace": {
                    "label": "Laplace",
                    "formula": "∇²u = 0",
                    "dimensions": {
                        "2d": {
                            "boundary_conditions": {
                                "dirichlet": {
                                    "solvers": ["analytical", "warp", "dolfinx"],
                                    "resolutions": [8, 16, 32, 64],
                                    "resolution_solvers": {
                                        "32": ["analytical", "warp", "dolfinx"],
                                        "64": ["analytical", "warp"]
                                    },
                                    "default_resolution": 32
                                }
                            }
                        }
                    }
                }
            }
        }
    """
    # Equation metadata (labels and formulas)
    equation_metadata = {
        'laplace': {'label': 'Laplace', 'formula': '∇²u = 0'},
        'poisson': {'label': 'Poisson', 'formula': '−∇²u = f'},
        'helmholtz': {'label': 'Helmholtz', 'formula': '−∇²u − k²u = 0'},
        'diffusion': {'label': 'Diffusion', 'formula': '∂u/∂t = κ∇²u'},
        'advection': {'label': 'Advection', 'formula': '∂u/∂t + c⃗·∇u = 0'},
        'convection_diffusion': {'label': 'Convection-Diffusion', 'formula': '∂u/∂t + c⃗·∇u = κ∇²u'},
        'wave': {'label': 'Wave', 'formula': '∂²u/∂t² = c²∇²u'},
    }

    # BC metadata (labels and edge descriptions)
    # Generic labels
    bc_labels = {
        'dirichlet': 'Dirichlet',
        'neumann': 'Neumann',
        'mixed': 'Mixed',
        'periodic': 'Periodic',
        'pml': 'PML (Absorbing)',
        'inflow_outflow': 'Inflow/Outflow',
    }

    # Detailed BC edge descriptions per problem (equation/dimension/bc)
    # Format: list of "edge: condition" strings, one per line
    bc_edge_details = {
        # Laplace 2D
        ('laplace', '2d', 'dirichlet'): [
            'Top: u = sin(πx)',
            'Other: u = 0',
        ],
        ('laplace', '2d', 'mixed'): [
            'Left/Bottom: u = 0',
            'Right/Top: ∂u/∂n = 0',
        ],
        # Laplace 3D
        ('laplace', '3d', 'mixed'): [
            'z=0: u = sin(πx)sin(πy)',
            'z=1: ∂u/∂n = 0',
            'Sides: u = 0',
        ],
        # Poisson 2D
        ('poisson', '2d', 'dirichlet'): [
            'All edges: u = 0',
        ],
        # Helmholtz 2D
        ('helmholtz', '2d', 'mixed'): [
            'x=0, y=0: u = u_exact',
            'x=1, y=1: ∂u/∂n = g',
        ],
        # Diffusion 2D
        ('diffusion', '2d', 'dirichlet'): [
            'All edges: u = 0',
        ],
        # Advection 2D
        ('advection', '2d', 'inflow_outflow'): [
            'Inflow (x=0, y=0): u = 0',
            'Outflow: natural BC',
        ],
        # Convection-Diffusion 2D
        ('convection_diffusion', '2d', 'neumann'): [
            'All edges: ∂u/∂n = 0',
        ],
        # Wave 2D
        ('wave', '2d', 'periodic'): [
            'All edges: ∂u/∂n = 0 (reflecting)',
        ],
        ('wave', '2d', 'pml'): [
            'Absorbing sponge layer at boundaries',
        ],
    }

    manifest = {'equations': {}}

    # Scan each equation directory
    for equation_dir in sorted(benchmarks_dir.iterdir()):
        if not equation_dir.is_dir() or equation_dir.name.startswith('.'):
            continue

        equation_name = equation_dir.name
        equation_data = {
            'label': equation_metadata.get(equation_name, {}).get('label', equation_name.title()),
            'formula': equation_metadata.get(equation_name, {}).get('formula', ''),
            'dimensions': {}
        }

        # Scan dimension directories (_2d, _3d)
        for dim_dir in sorted(equation_dir.iterdir()):
            if not dim_dir.is_dir() or not dim_dir.name.startswith('_'):
                continue

            dimension = dim_dir.name[1:]  # Remove leading underscore
            dimension_data = {'boundary_conditions': {}}

            # Scan BC directories
            for bc_dir in sorted(dim_dir.iterdir()):
                if not bc_dir.is_dir() or bc_dir.name.startswith('.'):
                    continue

                bc_name = bc_dir.name
                results_dir = bc_dir / 'results'

                # Skip if no results directory
                if not results_dir.exists() or not results_dir.is_dir():
                    continue

                # Parse all result files
                resolution_solvers = {}  # resolution -> list of solvers

                for result_file in results_dir.iterdir():
                    if not result_file.is_file():
                        continue

                    parsed = parse_result_filename(result_file.name)
                    if parsed:
                        res_str = str(parsed['resolution'])
                        if res_str not in resolution_solvers:
                            resolution_solvers[res_str] = []
                        if parsed['solver'] not in resolution_solvers[res_str]:
                            resolution_solvers[res_str].append(parsed['solver'])

                # Skip if no valid results found
                if not resolution_solvers:
                    continue

                # Sort solvers within each resolution
                for res in resolution_solvers:
                    resolution_solvers[res] = sorted(resolution_solvers[res])

                # Get all unique solvers and resolutions
                all_solvers = sorted(set(
                    solver for solvers in resolution_solvers.values() for solver in solvers
                ))
                all_resolutions = sorted([int(r) for r in resolution_solvers.keys()])

                # Find default resolution: most solvers, then highest resolution
                best_resolution = max(
                    all_resolutions,
                    key=lambda r: (len(resolution_solvers[str(r)]), r)
                )

                # Get BC edge details if available (as list of strings)
                edge_key = (equation_name, dimension, bc_name)
                edge_detail = bc_edge_details.get(edge_key, [])

                bc_data = {
                    'label': bc_labels.get(bc_name, bc_name.title()),
                    'detail': edge_detail,
                    'solvers': all_solvers,
                    'resolutions': all_resolutions,
                    'resolution_solvers': resolution_solvers,
                    'default_resolution': best_resolution
                }

                dimension_data['boundary_conditions'][bc_name] = bc_data

            # Only add dimension if it has BCs with results
            if dimension_data['boundary_conditions']:
                equation_data['dimensions'][dimension] = dimension_data

        # Only add equation if it has dimensions with results
        if equation_data['dimensions']:
            manifest['equations'][equation_name] = equation_data

    return manifest


def regenerate_manifest(quiet: bool = False) -> Path:
    """Regenerate the web manifest from benchmark results.

    Args:
        quiet: If True, suppress output messages

    Returns:
        Path to the generated manifest file
    """
    project_root = Path(__file__).parent.parent
    benchmarks_dir = project_root / 'benchmarks'
    output_path = project_root / 'web' / 'public' / 'manifest.json'

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = scan_benchmark_results(benchmarks_dir)

    # Write manifest
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if not quiet:
        print(f"Manifest updated: {output_path}")

    return output_path


def main():
    """Generate manifest.json from benchmarks directory (verbose mode)."""
    project_root = Path(__file__).parent.parent
    benchmarks_dir = project_root / 'benchmarks'

    print(f"Scanning benchmarks in: {benchmarks_dir}")
    output_path = regenerate_manifest(quiet=True)
    print(f"Generated manifest at: {output_path}")

    # Print summary
    with open(output_path, 'r') as f:
        manifest = json.load(f)

    total_problems = 0
    for eq_name, eq_data in manifest['equations'].items():
        for dim, dim_data in eq_data['dimensions'].items():
            for bc_name, bc_data in dim_data['boundary_conditions'].items():
                total_problems += 1
                print(f"  {eq_name}/{dim}/{bc_name}: {len(bc_data['solvers'])} solvers, "
                      f"resolutions {bc_data['resolutions']}, default={bc_data['default_resolution']}")

    print(f"\nTotal: {total_problems} problems with results")


if __name__ == '__main__':
    main()
