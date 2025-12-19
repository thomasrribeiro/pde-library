/**
 * Data loader for pre-computed solver results.
 * Loads .npz files from GitHub raw URLs (production) or local files (development).
 */

import { load_solver_data_from_npz } from './npz-loader.js';

const cache = new Map();

// GitHub repository configuration
const GITHUB_RAW_BASE_URL = 'https://raw.githubusercontent.com/thomasrribeiro/pde-library/master';

/**
 * Check if running in development mode (localhost or file://).
 *
 * @returns {boolean} True if in development mode
 */
function is_development_mode() {
    const hostname = window.location.hostname;
    return hostname === 'localhost' || hostname === '127.0.0.1' || window.location.protocol === 'file:';
}

/**
 * Load pre-computed solver data from .npz file.
 * In production, fetches from GitHub raw URLs.
 * In development, fetches from local benchmarks directory.
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
        const use_github = !is_development_mode();
        const data = await load_solver_data_from_npz(equation, boundary_condition, dimension, solver, resolution, use_github);
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

/**
 * Load problem description from description.txt file in benchmark directory.
 *
 * @param {string} equation - Equation name (e.g., 'laplace')
 * @param {string} boundary_condition - BC name (e.g., 'dirichlet')
 * @param {string} dimension - Dimension (e.g., '2d')
 * @returns {Promise<string|null>} Description text or null if not found
 */
export async function load_problem_description(equation, boundary_condition, dimension) {
    const cache_key = `description_${equation}/${boundary_condition}/${dimension}`;

    if (cache.has(cache_key)) {
        return cache.get(cache_key);
    }

    // Use GitHub raw URL in production, local path in development
    const relative_path = `benchmarks/${equation}/_${dimension}/${boundary_condition}/description.txt`;
    const url = is_development_mode()
        ? `../${relative_path}`
        : `${GITHUB_RAW_BASE_URL}/${relative_path}`;

    try {
        const response = await fetch(url);
        if (!response.ok) {
            console.log(`[loader] No description found at ${url}`);
            return null;
        }
        const description = await response.text();
        cache.set(cache_key, description.trim());
        return description.trim();
    } catch (error) {
        console.log(`[loader] Failed to load description: ${error.message}`);
        return null;
    }
}
