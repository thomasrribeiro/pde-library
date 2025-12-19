/**
 * Plotly.js heatmap visualization utilities.
 *
 * Uses custom colorscales matching VTK.js 3D visualization for consistency.
 */

const PLOT_SIZE = 320; // Match sidebar width

// Custom Viridis colorscale matching VTK.js (white at 0 -> purple -> teal -> yellow)
// White at zero for consistency with 3D volume rendering (transparent at 0)
const VIRIDIS_COLORSCALE = [
    [0.0, 'rgb(255, 255, 255)'],   // White at 0
    [0.1, 'rgb(68, 1, 84)'],       // Dark purple
    [0.25, 'rgb(72, 36, 117)'],    // Purple-blue
    [0.5, 'rgb(32, 144, 140)'],    // Teal
    [0.75, 'rgb(189, 223, 38)'],   // Yellow-green
    [1.0, 'rgb(253, 231, 37)']     // Yellow
];

// Custom Reds colorscale matching VTK.js (white -> red)
const REDS_COLORSCALE = [
    [0.0, 'rgb(255, 255, 255)'],   // White at 0
    [0.1, 'rgb(252, 230, 217)'],   // Very light pink
    [0.25, 'rgb(252, 199, 181)'],  // Light pink
    [0.5, 'rgb(250, 130, 112)'],   // Salmon
    [0.75, 'rgb(214, 48, 38)'],    // Red
    [1.0, 'rgb(102, 0, 13)']       // Dark red
];

/**
 * Create a 2D heatmap plot using Plotly.
 *
 * @param {string} element_id - DOM element ID for the plot
 * @param {Array<Array<number>>} z_data - 2D array of values
 * @param {Array<number>} x_values - X axis values
 * @param {Array<number>} y_values - Y axis values
 * @param {Object} options - Optional configuration
 */
export function create_heatmap(element_id, z_data, x_values, y_values, options = {}) {
    const {
        colorscale = 'viridis',
        zmin = null,
        zmax = null,
        showscale = true
    } = options;

    // Select colorscale based on option (use custom scales matching VTK.js)
    const colorscale_lower = colorscale.toLowerCase();
    const selected_colorscale = colorscale_lower === 'reds'
        ? REDS_COLORSCALE
        : VIRIDIS_COLORSCALE;

    const data = [{
        type: 'heatmap',
        z: z_data,
        x: x_values,
        y: y_values,
        colorscale: selected_colorscale,
        zmin: zmin,
        zmax: zmax,
        showscale: showscale,
        zsmooth: 'best',  // Bilinear interpolation for smooth appearance
        hoverongaps: false,
        hoverinfo: 'none',
        colorbar: {
            thickness: 12,
            lenmode: 'fraction',
            len: 0.6,
            yref: 'paper',
            y: 0.5,
            yanchor: 'middle',
            tickfont: { size: 10, family: 'monospace' },
            xpad: 8,
            outlinewidth: 1,
            outlinecolor: '#333',
            borderwidth: 0
        }
    }];

    // Generate matching tick values for x and y
    const tick_values = [0, 0.25, 0.5, 0.75, 1.0];

    const layout = {
        margin: { t: 5, r: showscale ? 45 : 5, b: 35, l: 45 },
        xaxis: {
            title: { text: 'x', font: { size: 11 }, standoff: 5 },
            tickfont: { size: 9 },
            tickvals: tick_values,
            ticktext: tick_values.map(v => v.toString()),
            constrain: 'domain'
        },
        yaxis: {
            title: { text: 'y', font: { size: 11 }, standoff: 10 },
            tickfont: { size: 9 },
            tickvals: tick_values,
            ticktext: tick_values.map(v => v.toString()),
            scaleanchor: 'x',
            scaleratio: 1,
            constrain: 'domain'
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        width: PLOT_SIZE,
        height: PLOT_SIZE
    };

    const config = {
        responsive: false,
        displayModeBar: false,
        staticPlot: true  // Disable all interaction
    };

    Plotly.newPlot(element_id, data, layout, config);
}

/**
 * Update an existing heatmap with new data.
 *
 * @param {string} element_id - DOM element ID for the plot
 * @param {Array<Array<number>>} z_data - 2D array of values
 * @param {Object} options - Optional configuration for zmin/zmax
 */
export function update_heatmap(element_id, z_data, options = {}) {
    const { zmin = null, zmax = null } = options;

    const update = { z: [z_data] };
    if (zmin !== null) update.zmin = zmin;
    if (zmax !== null) update.zmax = zmax;

    Plotly.restyle(element_id, update);
}

/**
 * Compute pointwise absolute error between two solution grids.
 *
 * @param {Array<Array<number>>} grid1 - First solution grid
 * @param {Array<Array<number>>} grid2 - Second solution grid
 * @returns {Array<Array<number>>} Absolute error grid
 */
export function compute_error_grid(grid1, grid2) {
    const error_grid = [];
    for (let i = 0; i < grid1.length; i++) {
        const row = [];
        for (let j = 0; j < grid1[i].length; j++) {
            row.push(Math.abs(grid1[i][j] - grid2[i][j]));
        }
        error_grid.push(row);
    }
    return error_grid;
}

/**
 * Find the min and max values in a 2D grid.
 *
 * @param {Array<Array<number>>} grid - 2D array of values
 * @returns {Object} Object with min and max properties
 */
export function find_grid_range(grid) {
    let min_val = Infinity;
    let max_val = -Infinity;

    for (const row of grid) {
        for (const val of row) {
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
    }

    return { min: min_val, max: max_val };
}

/**
 * Compute L2 and L-infinity error norms.
 *
 * @param {Array<Array<number>>} error_grid - Absolute error grid
 * @returns {Object} Object with l2 and linf error values
 */
export function compute_error_norms(error_grid) {
    let sum_squared = 0;
    let max_error = 0;
    let count = 0;

    for (const row of error_grid) {
        for (const val of row) {
            sum_squared += val * val;
            if (val > max_error) max_error = val;
            count++;
        }
    }

    return {
        l2: Math.sqrt(sum_squared / count),
        linf: max_error
    };
}
