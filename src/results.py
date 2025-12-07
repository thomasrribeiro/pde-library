"""Results caching utilities for storing and loading solver outputs."""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any


def get_result_filename(grid_resolution: int) -> str:
    """Generate filename for cached result based on resolution.

    Args:
        grid_resolution: Number of cells in each dimension

    Returns:
        Filename string like 'solution_res032.npz'
    """
    return f"solution_res{grid_resolution:03d}.npz"


def get_result_path(results_directory: Path, grid_resolution: int) -> Path:
    """Get full path to cached result file.

    Args:
        results_directory: Path to results folder
        grid_resolution: Number of cells in each dimension

    Returns:
        Full path to the .npz file
    """
    return results_directory / get_result_filename(grid_resolution)


def result_exists(results_directory: Path, grid_resolution: int) -> bool:
    """Check if a cached result exists for the given resolution.

    Args:
        results_directory: Path to results folder
        grid_resolution: Number of cells in each dimension

    Returns:
        True if cached result exists
    """
    return get_result_path(results_directory, grid_resolution).exists()


def save_result(
    results_directory: Path,
    grid_resolution: int,
    solution_values: np.ndarray,
    node_positions: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save solver result to disk.

    Args:
        results_directory: Path to results folder
        grid_resolution: Number of cells in each dimension
        solution_values: Array of solution values at nodes
        node_positions: Array of node coordinates
        metadata: Optional dict of additional metadata

    Returns:
        Path to saved file
    """
    results_directory.mkdir(parents=True, exist_ok=True)
    result_path = get_result_path(results_directory, grid_resolution)

    save_dict = {
        "solution_values": solution_values,
        "node_positions": node_positions,
        "grid_resolution": grid_resolution,
    }

    if metadata is not None:
        for key, value in metadata.items():
            save_dict[key] = value

    np.savez_compressed(result_path, **save_dict)
    return result_path


def load_result(
    results_directory: Path,
    grid_resolution: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load cached solver result from disk.

    Args:
        results_directory: Path to results folder
        grid_resolution: Number of cells in each dimension

    Returns:
        Tuple of (solution_values, node_positions, metadata_dict)

    Raises:
        FileNotFoundError: If cached result doesn't exist
    """
    result_path = get_result_path(results_directory, grid_resolution)

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


def list_cached_resolutions(results_directory: Path) -> list:
    """List all cached resolutions in results directory.

    Args:
        results_directory: Path to results folder

    Returns:
        Sorted list of grid resolutions with cached results
    """
    if not results_directory.exists():
        return []

    resolutions = []
    for path in results_directory.glob("solution_res*.npz"):
        # Extract resolution from filename like 'solution_res032.npz'
        try:
            resolution_str = path.stem.replace("solution_res", "")
            resolution = int(resolution_str)
            resolutions.append(resolution)
        except ValueError:
            continue

    return sorted(resolutions)


def solve_with_cache(
    solver_function,
    results_directory: Path,
    grid_resolution: int,
    force_recompute: bool = False,
    quiet: bool = True,
    **solver_kwargs,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Run solver with caching - skip computation if result exists.

    Args:
        solver_function: Function that takes grid_resolution and returns (solution, positions)
        results_directory: Path to results folder
        grid_resolution: Number of cells in each dimension
        force_recompute: If True, recompute even if cached result exists
        quiet: If True, suppress solver output
        **solver_kwargs: Additional arguments to pass to solver

    Returns:
        Tuple of (solution_values, node_positions, was_cached)
        was_cached is True if result was loaded from cache
    """
    if not force_recompute and result_exists(results_directory, grid_resolution):
        solution_values, node_positions, _ = load_result(results_directory, grid_resolution)
        return solution_values, node_positions, True

    # Run solver
    solution_values, node_positions = solver_function(
        grid_resolution=grid_resolution,
        quiet=quiet,
        **solver_kwargs,
    )

    # Save result
    save_result(
        results_directory=results_directory,
        grid_resolution=grid_resolution,
        solution_values=solution_values,
        node_positions=node_positions,
    )

    return solution_values, node_positions, False
