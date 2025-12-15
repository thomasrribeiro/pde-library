/**
 * Data loader for pre-computed solver results.
 * Loads .npz files directly from benchmarks directory.
 */

import { load_solver_data_from_npz } from './npz-loader.js';

const cache = new Map();

/**
 * Load pre-computed solver data from .npz file in benchmarks directory.
 *
 * @param {string} equation - Equation name (e.g., 'laplace')
 * @param {string} boundary_condition - BC name (e.g., 'dirichlet')
 * @param {string} dimension - Dimension (e.g., '2d')
 * @param {string} solver - Solver name (e.g., 'warp', 'dolfinx')
 * @param {number} resolution - Grid resolution
 * @returns {Promise<Object>} Loaded data with x, y, values arrays
 */
export async function load_solver_data(equation, boundary_condition, dimension, solver, resolution) {
    const cache_key = `${equation}/${boundary_condition}/${dimension}/${solver}_res${String(resolution).padStart(3, '0')}`;

    if (cache.has(cache_key)) {
        return cache.get(cache_key);
    }

    try {
        const data = await load_solver_data_from_npz(equation, boundary_condition, dimension, solver, resolution);
        cache.set(cache_key, data);
        return data;
    } catch (error) {
        console.error(`Error loading solver data: ${error.message}`);
        return null;
    }
}

/**
 * Clear the data cache.
 */
export function clear_cache() {
    cache.clear();
}
