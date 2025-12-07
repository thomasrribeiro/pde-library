"""Timing utilities with GPU synchronization for accurate benchmarking."""

import time
from typing import List, Dict, Any

import warp as wp


def synchronize_gpu_if_needed(should_synchronize: bool) -> None:
    """Synchronize GPU to ensure all operations are complete."""
    if should_synchronize:
        wp.synchronize()


def measure_execution_time_milliseconds(
    function_to_time,
    *args,
    synchronize_gpu: bool = True,
    **kwargs
) -> tuple:
    """Execute a function and measure its execution time in milliseconds.

    Args:
        function_to_time: The function to execute and time
        *args: Positional arguments to pass to the function
        synchronize_gpu: If True, synchronize GPU before/after for accurate timing
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Tuple of (function_result, elapsed_time_in_milliseconds)
    """
    synchronize_gpu_if_needed(synchronize_gpu)
    start_time = time.perf_counter()

    result = function_to_time(*args, **kwargs)

    synchronize_gpu_if_needed(synchronize_gpu)
    end_time = time.perf_counter()

    elapsed_milliseconds = (end_time - start_time) * 1000.0
    return result, elapsed_milliseconds


def run_benchmark_with_warmup(
    function_to_benchmark,
    *args,
    number_of_timed_runs: int = 5,
    number_of_warmup_runs: int = 1,
    synchronize_gpu: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Run a function multiple times and collect timing statistics.

    Args:
        function_to_benchmark: Function to benchmark
        *args: Positional arguments to pass to function
        number_of_timed_runs: Number of timed runs (after warmup)
        number_of_warmup_runs: Number of warmup runs (not timed)
        synchronize_gpu: If True, synchronize GPU before/after each run
        **kwargs: Keyword arguments to pass to function

    Returns:
        Dictionary with timing statistics:
            - mean_time_milliseconds
            - standard_deviation_milliseconds
            - minimum_time_milliseconds
            - maximum_time_milliseconds
            - all_times_milliseconds (list)
    """
    import numpy as np

    # Warmup runs - not timed
    for _ in range(number_of_warmup_runs):
        function_to_benchmark(*args, **kwargs)

    # Timed runs
    all_elapsed_times_milliseconds = []
    for _ in range(number_of_timed_runs):
        _, elapsed_ms = measure_execution_time_milliseconds(
            function_to_benchmark,
            *args,
            synchronize_gpu=synchronize_gpu,
            **kwargs
        )
        all_elapsed_times_milliseconds.append(elapsed_ms)

    times_array = np.array(all_elapsed_times_milliseconds)

    return {
        "mean_time_milliseconds": float(np.mean(times_array)),
        "standard_deviation_milliseconds": float(np.std(times_array)),
        "minimum_time_milliseconds": float(np.min(times_array)),
        "maximum_time_milliseconds": float(np.max(times_array)),
        "all_times_milliseconds": all_elapsed_times_milliseconds,
    }


def create_timing_context(synchronize_gpu: bool = True) -> Dict[str, Any]:
    """Create a timing context dictionary for manual timing.

    Usage:
        context = create_timing_context()
        start_timing(context)
        # ... do work ...
        stop_timing(context)
        print(f"Elapsed: {context['elapsed_milliseconds']:.2f} ms")

    Args:
        synchronize_gpu: If True, synchronize GPU at start/stop

    Returns:
        Dictionary to track timing state
    """
    return {
        "synchronize_gpu": synchronize_gpu,
        "start_time": None,
        "elapsed_milliseconds": 0.0,
    }


def start_timing(timing_context: Dict[str, Any]) -> None:
    """Start the timer in a timing context."""
    synchronize_gpu_if_needed(timing_context["synchronize_gpu"])
    timing_context["start_time"] = time.perf_counter()


def stop_timing(timing_context: Dict[str, Any]) -> float:
    """Stop the timer and return elapsed milliseconds."""
    synchronize_gpu_if_needed(timing_context["synchronize_gpu"])
    end_time = time.perf_counter()
    timing_context["elapsed_milliseconds"] = (end_time - timing_context["start_time"]) * 1000.0
    return timing_context["elapsed_milliseconds"]
