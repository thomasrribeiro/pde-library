/**
 * VTK.js 2D heatmap visualization for PDE solutions.
 *
 * Uses vtkPlaneSource with scalar coloring for reliable 2D rendering.
 */

import '@kitware/vtk.js/Rendering/Profiles/Geometry';

import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
import vtkPlaneSource from '@kitware/vtk.js/Filters/Sources/PlaneSource';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';

// Cache for VTK contexts per plot
const vtk_contexts = {};

/**
 * Create Viridis color transfer function.
 *
 * @param {number} min_val - Minimum scalar value
 * @param {number} max_val - Maximum scalar value
 * @returns {vtkColorTransferFunction} Color transfer function
 */
function create_viridis_color_function(min_val, max_val) {
    const range = max_val - min_val;
    const ctf = vtkColorTransferFunction.newInstance();

    // Viridis colormap (purple -> teal -> yellow)
    ctf.addRGBPoint(min_val, 0.267, 0.004, 0.329);                      // Dark purple
    ctf.addRGBPoint(min_val + 0.25 * range, 0.282, 0.140, 0.458);       // Purple-blue
    ctf.addRGBPoint(min_val + 0.5 * range, 0.127, 0.566, 0.550);        // Teal
    ctf.addRGBPoint(min_val + 0.75 * range, 0.741, 0.873, 0.150);       // Yellow-green
    ctf.addRGBPoint(max_val, 0.993, 0.906, 0.144);                      // Yellow

    return ctf;
}

/**
 * Create Reds color transfer function for error visualization.
 *
 * @param {number} min_val - Minimum scalar value
 * @param {number} max_val - Maximum scalar value
 * @returns {vtkColorTransferFunction} Color transfer function
 */
function create_reds_color_function(min_val, max_val) {
    const range = max_val - min_val;
    const ctf = vtkColorTransferFunction.newInstance();

    // White to red colormap
    ctf.addRGBPoint(min_val, 1.0, 1.0, 1.0);                            // White at 0
    ctf.addRGBPoint(min_val + 0.1 * range, 0.99, 0.90, 0.85);           // Very light pink
    ctf.addRGBPoint(min_val + 0.25 * range, 0.99, 0.78, 0.71);          // Light pink
    ctf.addRGBPoint(min_val + 0.5 * range, 0.98, 0.51, 0.44);           // Salmon
    ctf.addRGBPoint(min_val + 0.75 * range, 0.84, 0.19, 0.15);          // Red
    ctf.addRGBPoint(max_val, 0.40, 0.0, 0.05);                          // Dark red

    return ctf;
}

/**
 * Create an HTML colorbar element overlay.
 *
 * @param {HTMLElement} container - Parent container element
 * @param {number} min_val - Minimum scalar value
 * @param {number} max_val - Maximum scalar value
 * @param {string} colorscale - Colorscale name ('viridis' or 'reds')
 * @returns {HTMLElement} The colorbar element
 */
function create_colorbar_element(container, min_val, max_val, colorscale) {
    // Remove existing colorbar if any
    const existing = container.querySelector('.vtk-colorbar');
    if (existing) {
        existing.remove();
    }

    // Create colorbar container
    const colorbar = document.createElement('div');
    colorbar.className = 'vtk-colorbar';
    colorbar.style.cssText = `
        position: absolute;
        right: 8px;
        top: 50%;
        transform: translateY(-50%);
        width: 18px;
        height: 55%;
        display: flex;
        flex-direction: column;
        align-items: center;
        pointer-events: none;
        z-index: 10;
    `;

    // Create gradient bar
    const gradientBar = document.createElement('div');
    gradientBar.className = 'colorbar-gradient';

    // Build CSS gradient based on colorscale
    let gradientStops;
    if (colorscale === 'reds') {
        gradientStops = `
            rgb(102, 0, 13) 0%,
            rgb(214, 48, 38) 25%,
            rgb(250, 130, 112) 50%,
            rgb(252, 199, 181) 75%,
            rgb(255, 255, 255) 100%
        `;
    } else {
        // Viridis (top to bottom: yellow -> teal -> purple)
        gradientStops = `
            rgb(253, 231, 37) 0%,
            rgb(189, 223, 38) 25%,
            rgb(32, 144, 140) 50%,
            rgb(72, 36, 117) 75%,
            rgb(68, 1, 84) 100%
        `;
    }

    gradientBar.style.cssText = `
        width: 10px;
        flex: 1;
        background: linear-gradient(to bottom, ${gradientStops});
        border: 1px solid #333;
        border-radius: 2px;
    `;

    // Create max label (top)
    const maxLabel = document.createElement('div');
    maxLabel.className = 'colorbar-label';
    maxLabel.textContent = format_colorbar_value(max_val);
    maxLabel.style.cssText = `
        font-size: 9px;
        color: #333;
        margin-bottom: 3px;
        font-family: monospace;
    `;

    // Create min label (bottom)
    const minLabel = document.createElement('div');
    minLabel.className = 'colorbar-label';
    minLabel.textContent = format_colorbar_value(min_val);
    minLabel.style.cssText = `
        font-size: 9px;
        color: #333;
        margin-top: 3px;
        font-family: monospace;
    `;

    // Assemble colorbar
    colorbar.appendChild(maxLabel);
    colorbar.appendChild(gradientBar);
    colorbar.appendChild(minLabel);

    // Make sure container has relative positioning for absolute child
    if (getComputedStyle(container).position === 'static') {
        container.style.position = 'relative';
    }

    container.appendChild(colorbar);
    return colorbar;
}

/**
 * Create axis labels overlay for 2D plot.
 *
 * @param {HTMLElement} container - Parent container element
 * @returns {HTMLElement} The axes labels element
 */
function create_axis_labels(container) {
    // Remove existing labels if any
    const existing = container.querySelector('.vtk-axis-labels');
    if (existing) {
        existing.remove();
    }

    const labelsContainer = document.createElement('div');
    labelsContainer.className = 'vtk-axis-labels';
    labelsContainer.style.cssText = `
        position: absolute;
        inset: 0;
        pointer-events: none;
        z-index: 10;
    `;

    // X-axis label
    const xLabel = document.createElement('div');
    xLabel.textContent = 'x';
    xLabel.style.cssText = `
        position: absolute;
        bottom: 5px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 11px;
        font-family: sans-serif;
        color: #333;
    `;

    // Y-axis label
    const yLabel = document.createElement('div');
    yLabel.textContent = 'y';
    yLabel.style.cssText = `
        position: absolute;
        left: 5px;
        top: 50%;
        transform: translateY(-50%) rotate(-90deg);
        font-size: 11px;
        font-family: sans-serif;
        color: #333;
    `;

    // Corner tick labels
    const tickStyle = `
        position: absolute;
        font-size: 9px;
        font-family: monospace;
        color: #666;
    `;

    // Origin (0,0)
    const originLabel = document.createElement('div');
    originLabel.textContent = '0';
    originLabel.style.cssText = tickStyle + `bottom: 20px; left: 20px;`;

    // X max (1,0)
    const xMaxLabel = document.createElement('div');
    xMaxLabel.textContent = '1';
    xMaxLabel.style.cssText = tickStyle + `bottom: 20px; right: 35px;`;

    // Y max (0,1)
    const yMaxLabel = document.createElement('div');
    yMaxLabel.textContent = '1';
    yMaxLabel.style.cssText = tickStyle + `top: 10px; left: 20px;`;

    labelsContainer.appendChild(xLabel);
    labelsContainer.appendChild(yLabel);
    labelsContainer.appendChild(originLabel);
    labelsContainer.appendChild(xMaxLabel);
    labelsContainer.appendChild(yMaxLabel);

    container.appendChild(labelsContainer);
    return labelsContainer;
}

/**
 * Format a value for colorbar display.
 *
 * @param {number} value - Value to format
 * @returns {string} Formatted string
 */
function format_colorbar_value(value) {
    const abs_val = Math.abs(value);
    if (abs_val === 0) {
        return '0';
    } else if (abs_val >= 1000 || abs_val < 0.01) {
        return value.toExponential(1);
    } else if (abs_val >= 1) {
        return value.toFixed(1);
    } else {
        return value.toFixed(2);
    }
}

/**
 * Create a 2D heatmap plot using VTK.js with a plane source.
 *
 * @param {string} element_id - DOM element ID for the plot
 * @param {Array<Array<number>>} z_data - 2D array of values [y][x]
 * @param {Array<number>} x_values - X axis values (unused, kept for API compatibility)
 * @param {Array<number>} y_values - Y axis values (unused, kept for API compatibility)
 * @param {Object} options - Optional configuration
 */
export function create_heatmap(element_id, z_data, x_values, y_values, options = {}) {
    const {
        colorscale = 'Viridis',
        zmin = null,
        zmax = null,
        showscale = true
    } = options;

    // Dispose existing context if any
    dispose_heatmap(element_id);

    // Get container element
    const container = document.getElementById(element_id);
    if (!container) {
        console.error(`Container element not found: ${element_id}`);
        return;
    }

    // Find value range
    let computed_zmin = zmin;
    let computed_zmax = zmax;
    if (computed_zmin === null || computed_zmax === null) {
        const range = find_grid_range(z_data);
        if (computed_zmin === null) computed_zmin = range.min;
        if (computed_zmax === null) computed_zmax = range.max;
    }

    // Ensure container has explicit dimensions
    const rect = container.getBoundingClientRect();
    const width = rect.width > 0 ? rect.width : 320;
    const height = rect.height > 0 ? rect.height : 320;

    // Create render window
    const genericRenderWindow = vtkGenericRenderWindow.newInstance();
    genericRenderWindow.setContainer(container);

    // Explicitly set size before any rendering
    const openGLRenderWindow = genericRenderWindow.getApiSpecificRenderWindow();
    openGLRenderWindow.setSize(width, height);

    const renderer = genericRenderWindow.getRenderer();
    const renderWindow = genericRenderWindow.getRenderWindow();
    const interactor = genericRenderWindow.getInteractor();

    // Disable interaction - this is a static 2D plot
    interactor.setInteractorStyle(null);
    interactor.disable();

    // Set white background
    renderer.setBackground(1, 1, 1);

    // Get grid dimensions
    const ny = z_data.length;
    const nx = z_data[0].length;

    // Create a plane source with resolution matching the data
    const planeSource = vtkPlaneSource.newInstance();
    planeSource.setOrigin(0, 0, 0);
    planeSource.setPoint1(1, 0, 0);  // X direction
    planeSource.setPoint2(0, 1, 0);  // Y direction
    planeSource.setXResolution(nx - 1);
    planeSource.setYResolution(ny - 1);
    planeSource.update();

    // Get the output polydata
    const polyData = planeSource.getOutputData();

    // Create scalar array for coloring - one value per point
    const numPoints = polyData.getNumberOfPoints();
    const scalars = new Float32Array(numPoints);

    // Map grid values to plane points
    // PlaneSource creates points in row-major order (y varies slower)
    for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
            const pointIndex = j * nx + i;
            scalars[pointIndex] = z_data[j][i];
        }
    }

    // Add scalars to polydata
    const scalarArray = vtkDataArray.newInstance({
        values: scalars,
        numberOfComponents: 1,
        name: 'solution'
    });
    polyData.getPointData().setScalars(scalarArray);

    // Create color transfer function
    const colorscale_lower = colorscale.toLowerCase();
    const colorTransferFunction = colorscale_lower === 'reds'
        ? create_reds_color_function(computed_zmin, computed_zmax)
        : create_viridis_color_function(computed_zmin, computed_zmax);

    // Create mapper
    const mapper = vtkMapper.newInstance();
    mapper.setInputData(polyData);
    mapper.setLookupTable(colorTransferFunction);
    mapper.setScalarRange(computed_zmin, computed_zmax);
    mapper.setScalarModeToUsePointData();
    mapper.setColorModeToMapScalars();
    mapper.setInterpolateScalarsBeforeMapping(true);

    // Create actor
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);

    renderer.addActor(actor);

    // Set up camera for 2D view (looking down Z axis)
    const camera = renderer.getActiveCamera();
    camera.setParallelProjection(true);
    camera.setPosition(0.5, 0.5, 10);
    camera.setFocalPoint(0.5, 0.5, 0);
    camera.setViewUp(0, 1, 0);

    // Adjust parallel scale to fit the domain with some padding
    const aspectRatio = width / height;
    if (aspectRatio > 1) {
        camera.setParallelScale(0.55);
    } else {
        camera.setParallelScale(0.55 / aspectRatio);
    }

    renderer.resetCameraClippingRange();

    // Create colorbar and axis labels
    let colorbarElement = null;
    if (showscale) {
        colorbarElement = create_colorbar_element(
            container,
            computed_zmin,
            computed_zmax,
            colorscale_lower
        );
    }
    const axisLabels = create_axis_labels(container);

    // Force initial render
    renderWindow.render();

    // Store context for later updates
    vtk_contexts[element_id] = {
        genericRenderWindow,
        openGLRenderWindow,
        renderer,
        renderWindow,
        planeSource,
        polyData,
        scalarArray,
        mapper,
        actor,
        colorTransferFunction,
        colorbarElement,
        axisLabels,
        nx,
        ny,
        zmin: computed_zmin,
        zmax: computed_zmax
    };
}

/**
 * Update an existing heatmap with new data.
 *
 * @param {string} element_id - DOM element ID for the plot
 * @param {Array<Array<number>>} z_data - 2D array of values
 * @param {Object} options - Optional configuration for zmin/zmax
 */
export function update_heatmap(element_id, z_data, options = {}) {
    const context = vtk_contexts[element_id];
    if (!context) {
        console.error(`No VTK context found for ${element_id}`);
        return;
    }

    const { zmin = null, zmax = null } = options;
    const { nx, ny, scalarArray, polyData, mapper, renderWindow } = context;

    // Find value range if not provided
    let computed_zmin = zmin !== null ? zmin : context.zmin;
    let computed_zmax = zmax !== null ? zmax : context.zmax;

    if (zmin === null || zmax === null) {
        const range = find_grid_range(z_data);
        if (zmin === null) computed_zmin = range.min;
        if (zmax === null) computed_zmax = range.max;
    }

    // Update scalar values
    const scalars = scalarArray.getData();
    for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
            const pointIndex = j * nx + i;
            scalars[pointIndex] = z_data[j][i];
        }
    }
    scalarArray.modified();
    polyData.modified();

    // Update scalar range if changed
    if (computed_zmin !== context.zmin || computed_zmax !== context.zmax) {
        mapper.setScalarRange(computed_zmin, computed_zmax);
        context.zmin = computed_zmin;
        context.zmax = computed_zmax;
    }

    // Trigger render
    renderWindow.render();
}

/**
 * Dispose of a heatmap and free resources.
 *
 * @param {string} element_id - DOM element ID for the plot
 */
export function dispose_heatmap(element_id) {
    const context = vtk_contexts[element_id];
    if (!context) return;

    // Remove HTML overlays
    if (context.colorbarElement && context.colorbarElement.parentNode) {
        context.colorbarElement.remove();
    }
    if (context.axisLabels && context.axisLabels.parentNode) {
        context.axisLabels.remove();
    }

    // Clean up VTK objects
    if (context.colorTransferFunction) {
        context.colorTransferFunction.delete();
    }
    if (context.actor) {
        context.actor.delete();
    }
    if (context.mapper) {
        context.mapper.delete();
    }
    if (context.planeSource) {
        context.planeSource.delete();
    }
    if (context.genericRenderWindow) {
        context.genericRenderWindow.delete();
    }

    // Remove from cache
    delete vtk_contexts[element_id];
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
