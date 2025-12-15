/**
 * Laplace equation analytical solver.
 *
 * Solves ∇²u = 0 on [0,1]² with Dirichlet boundary conditions:
 * - Top (y=1): u = sin(πx)
 * - Bottom, Left, Right: u = 0
 *
 * Analytical solution: u(x,y) = sin(πx) · sinh(πy) / sinh(π)
 */

const SINH_PI = Math.sinh(Math.PI);

/**
 * Compute Laplace analytical solution at given coordinates.
 *
 * @param {Float32Array} x_coordinates - X coordinates of evaluation points
 * @param {Float32Array} y_coordinates - Y coordinates of evaluation points
 * @returns {Float32Array} Solution values at each point
 */
export function compute_laplace_analytical_solution(x_coordinates, y_coordinates) {
    const num_points = x_coordinates.length;
    const solution_values = new Float32Array(num_points);

    for (let i = 0; i < num_points; i++) {
        const x = x_coordinates[i];
        const y = y_coordinates[i];
        solution_values[i] = Math.sin(Math.PI * x) * Math.sinh(Math.PI * y) / SINH_PI;
    }

    return solution_values;
}

/**
 * Generate a regular grid and compute the Laplace solution.
 *
 * @param {number} grid_resolution - Number of cells in each dimension
 * @returns {Object} Solution data with x, y coordinates and values
 */
export function generate_laplace_solution_on_grid(grid_resolution) {
    const nodes_per_dimension = grid_resolution + 1;
    const num_nodes = nodes_per_dimension * nodes_per_dimension;

    const x_coordinates = new Float32Array(num_nodes);
    const y_coordinates = new Float32Array(num_nodes);

    // Generate grid points (row-major order: x varies fastest)
    let index = 0;
    for (let iy = 0; iy < nodes_per_dimension; iy++) {
        const y = iy / grid_resolution;
        for (let ix = 0; ix < nodes_per_dimension; ix++) {
            const x = ix / grid_resolution;
            x_coordinates[index] = x;
            y_coordinates[index] = y;
            index++;
        }
    }

    const solution_values = compute_laplace_analytical_solution(x_coordinates, y_coordinates);

    return {
        x: x_coordinates,
        y: y_coordinates,
        values: solution_values,
        resolution: grid_resolution
    };
}

/**
 * Reshape flat solution array to 2D grid for Plotly heatmap.
 * Assumes row-major order (y varies slowest, x varies fastest).
 *
 * @param {Float32Array} values - Flat array of solution values
 * @param {number} grid_resolution - Number of cells in each dimension
 * @returns {Array<Array<number>>} 2D array for Plotly z data
 */
export function reshape_to_grid(values, grid_resolution) {
    const nodes_per_dimension = grid_resolution + 1;
    const grid = [];

    for (let iy = 0; iy < nodes_per_dimension; iy++) {
        const row = [];
        for (let ix = 0; ix < nodes_per_dimension; ix++) {
            const index = iy * nodes_per_dimension + ix;
            row.push(values[index]);
        }
        grid.push(row);
    }

    return grid;
}

/**
 * Reshape flat solution array to 2D grid for Plotly heatmap.
 * Assumes column-major order (x varies slowest, y varies fastest).
 * This is the order used by Warp and DOLFINx solvers.
 *
 * @param {Float32Array} values - Flat array of solution values
 * @param {number} grid_resolution - Number of cells in each dimension
 * @returns {Array<Array<number>>} 2D array for Plotly z data
 */
export function reshape_to_grid_column_major(values, grid_resolution) {
    const nodes_per_dimension = grid_resolution + 1;
    const grid = [];

    // Build grid row by row (each row is constant y)
    for (let iy = 0; iy < nodes_per_dimension; iy++) {
        const row = [];
        for (let ix = 0; ix < nodes_per_dimension; ix++) {
            // In column-major: index = ix * nodes_per_dimension + iy
            const index = ix * nodes_per_dimension + iy;
            row.push(values[index]);
        }
        grid.push(row);
    }

    return grid;
}

/**
 * Get unique coordinate values for Plotly axes.
 *
 * @param {number} grid_resolution - Number of cells in each dimension
 * @returns {Array<number>} Array of coordinate values from 0 to 1
 */
export function get_axis_values(grid_resolution) {
    const nodes_per_dimension = grid_resolution + 1;
    const values = [];
    for (let i = 0; i < nodes_per_dimension; i++) {
        values.push(i / grid_resolution);
    }
    return values;
}
