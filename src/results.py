"""Results caching utilities for storing and loading solver outputs."""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List


def get_result_filename(solver_name: str, grid_resolution: int) -> str:
    """Generate filename for cached result based on solver and resolution.

    Args:
        solver_name: Name of the solver (e.g., 'warp', 'analytical', 'fenics')
        grid_resolution: Number of cells in each dimension

    Returns:
        Filename string like 'warp_res032.npz'
    """
    return f"{solver_name}_res{grid_resolution:03d}.npz"


def get_result_path(results_directory: Path, solver_name: str, grid_resolution: int) -> Path:
    """Get full path to cached result file.

    Args:
        results_directory: Path to results folder
        solver_name: Name of the solver
        grid_resolution: Number of cells in each dimension

    Returns:
        Full path to the .npz file
    """
    return results_directory / get_result_filename(solver_name, grid_resolution)


def result_exists(results_directory: Path, solver_name: str, grid_resolution: int) -> bool:
    """Check if a cached result exists for the given solver and resolution.

    Args:
        results_directory: Path to results folder
        solver_name: Name of the solver
        grid_resolution: Number of cells in each dimension

    Returns:
        True if cached result exists
    """
    return get_result_path(results_directory, solver_name, grid_resolution).exists()


def save_result(
    results_directory: Path,
    solver_name: str,
    grid_resolution: int,
    solution_values: np.ndarray,
    node_positions: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save solver result to disk.

    Args:
        results_directory: Path to results folder
        solver_name: Name of the solver
        grid_resolution: Number of cells in each dimension
        solution_values: Array of solution values at nodes
        node_positions: Array of node coordinates
        metadata: Optional dict of additional metadata

    Returns:
        Path to saved file
    """
    results_directory.mkdir(parents=True, exist_ok=True)
    result_path = get_result_path(results_directory, solver_name, grid_resolution)

    save_dict = {
        "solution_values": solution_values,
        "node_positions": node_positions,
        "grid_resolution": grid_resolution,
        "solver_name": solver_name,
    }

    if metadata is not None:
        for key, value in metadata.items():
            save_dict[key] = value

    np.savez_compressed(result_path, **save_dict)
    return result_path


def load_result(
    results_directory: Path,
    solver_name: str,
    grid_resolution: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load cached solver result from disk.

    Args:
        results_directory: Path to results folder
        solver_name: Name of the solver
        grid_resolution: Number of cells in each dimension

    Returns:
        Tuple of (solution_values, node_positions, metadata_dict)

    Raises:
        FileNotFoundError: If cached result doesn't exist
    """
    result_path = get_result_path(results_directory, solver_name, grid_resolution)

    if not result_path.exists():
        raise FileNotFoundError(f"No cached result at {result_path}")

    data = np.load(result_path, allow_pickle=True)

    solution_values = data["solution_values"]
    node_positions = data["node_positions"]

    # Collect any additional metadata
    metadata = {}
    for key in data.files:
        if key not in ("solution_values", "node_positions"):
            metadata[key] = data[key]

    return solution_values, node_positions, metadata


def list_cached_resolutions_for_solver(results_directory: Path, solver_name: str) -> List[int]:
    """List all cached resolutions for a specific solver.

    Args:
        results_directory: Path to results folder
        solver_name: Name of the solver

    Returns:
        Sorted list of grid resolutions with cached results for this solver
    """
    if not results_directory.exists():
        return []

    resolutions = []
    pattern = f"{solver_name}_res*.npz"
    for path in results_directory.glob(pattern):
        # Extract resolution from filename like 'warp_res032.npz'
        try:
            # Remove solver prefix and _res, then extract number
            stem = path.stem  # e.g., 'warp_res032'
            resolution_str = stem.replace(f"{solver_name}_res", "")
            resolution = int(resolution_str)
            resolutions.append(resolution)
        except ValueError:
            continue

    return sorted(resolutions)


def list_all_cached_results(results_directory: Path) -> Dict[str, List[int]]:
    """List all cached results grouped by solver name.

    Args:
        results_directory: Path to results folder

    Returns:
        Dict mapping solver names to lists of cached resolutions
    """
    if not results_directory.exists():
        return {}

    results = {}
    for path in results_directory.glob("*_res*.npz"):
        try:
            # Parse filename like 'warp_res032.npz'
            stem = path.stem
            # Find the last occurrence of '_res' to split solver name from resolution
            idx = stem.rfind("_res")
            if idx == -1:
                continue
            solver_name = stem[:idx]
            resolution_str = stem[idx + 4:]  # Skip '_res'
            resolution = int(resolution_str)

            if solver_name not in results:
                results[solver_name] = []
            results[solver_name].append(resolution)
        except ValueError:
            continue

    # Sort resolutions for each solver
    for solver_name in results:
        results[solver_name] = sorted(results[solver_name])

    return results
