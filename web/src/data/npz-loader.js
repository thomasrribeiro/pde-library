/**
 * NPZ file loader for browser.
 * Loads NumPy .npz files directly without server-side conversion.
 * Supports loading from GitHub raw URLs (production) or local files (development).
 */

import JSZip from 'jszip';
import { load as load_npy } from 'npyjs';

// GitHub repository configuration
const GITHUB_RAW_BASE_URL = 'https://raw.githubusercontent.com/thomasrribeiro/pde-library/master';

/**
 * Load and parse an .npz file from a URL.
 *
 * @param {string} url - URL to the .npz file
 * @returns {Promise<Object>} - Object with array names as keys
 */
export async function load_npz_file(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to fetch ${url}: ${response.status}`);
    }

    const array_buffer = await response.arrayBuffer();
    const zip = await JSZip.loadAsync(array_buffer);

    const result = {};

    for (const [filename, file] of Object.entries(zip.files)) {
        if (filename.endsWith('.npy')) {
            const array_name = filename.replace('.npy', '');
            const npy_buffer = await file.async('arraybuffer');
            try {
                // load_npy can parse an ArrayBuffer directly
                const parsed = await load_npy(npy_buffer);
                result[array_name] = parsed;
            } catch (e) {
                // Skip arrays with unsupported dtypes (e.g., Unicode strings)
                console.log(`[npz-loader] Skipping ${array_name}: ${e.message}`);
            }
        }
    }

    return result;
}

/**
 * Load solver data from .npz file.
 * Supports both 2D and 3D data.
 * In production, fetches from GitHub raw URLs (results/ directory).
 * In development, fetches from local benchmarks directory (results/ directory).
 *
 * @param {string} equation - Equation name (e.g., 'laplace')
 * @param {string} boundary_condition - BC name (e.g., 'dirichlet')
 * @param {string} dimension - Dimension (e.g., '2d' or '3d')
 * @param {string} solver - Solver name (e.g., 'warp', 'analytical', 'dolfinx')
 * @param {number} resolution - Grid resolution
 * @param {boolean} use_github - If true, fetch from GitHub raw URLs; otherwise local
 * @returns {Promise<Object>} Data object with coordinates and values
 *   For 2D: {x, y, values, resolution, is_3d: false}
 *   For 3D: {x, y, z, values, resolution, is_3d: true}
 */
export async function load_solver_data_from_npz(equation, boundary_condition, dimension, solver, resolution, use_github = false) {
    // Try both naming conventions: with and without _solver suffix
    const solver_names = [
        `${solver}_solver`,
        solver
    ];

    let npz_data = null;
    let last_error = null;

    for (const solver_name of solver_names) {
        // Always load from results/ directory (committed data)
        const relative_path = `benchmarks/${equation}/_${dimension}/${boundary_condition}/results/${solver_name}_res${String(resolution).padStart(3, '0')}.npz`;
        const url = use_github
            ? `${GITHUB_RAW_BASE_URL}/${relative_path}`
            : `../${relative_path}`;

        console.log(`[npz-loader] Trying to load: ${url}`);

        try {
            npz_data = await load_npz_file(url);
            console.log(`[npz-loader] Successfully loaded: ${url}`);
            break;
        } catch (error) {
            console.log(`[npz-loader] Failed to load ${url}: ${error.message}`);
            last_error = error;
        }
    }

    if (!npz_data) {
        throw last_error || new Error(`Failed to load solver data for ${solver}`);
    }

    // Extract arrays from npz data
    const solution_values = npz_data.solution_values;
    const node_positions = npz_data.node_positions;

    const num_points = node_positions.shape[0];
    const num_dims = node_positions.shape[1];  // 2 for 2D, 3 for 3D
    const is_3d = num_dims === 3;

    const x_values = [];
    const y_values = [];
    const z_values = is_3d ? [] : null;

    for (let i = 0; i < num_points; i++) {
        x_values.push(node_positions.data[i * num_dims]);
        y_values.push(node_positions.data[i * num_dims + 1]);
        if (is_3d) {
            z_values.push(node_positions.data[i * num_dims + 2]);
        }
    }

    const result = {
        x: x_values,
        y: y_values,
        values: Array.from(solution_values.data),
        resolution: resolution,
        is_3d: is_3d
    };

    if (is_3d) {
        result.z = z_values;
    }

    return result;
}
