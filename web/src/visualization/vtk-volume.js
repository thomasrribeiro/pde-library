/**
 * VTK.js 3D visualization with bisecting slice plane and volume rendering below.
 *
 * Architecture:
 * - vtkImageResliceMapper + vtkImageSlice for the 2D bisecting plane (always shows solution)
 * - vtkVolumeMapper + vtkVolume for translucent volume rendering below the plane
 * - vtkPlane with addClippingPlane() clips volume ABOVE the slice
 * - vtkOrientationMarkerWidget + vtkAxesActor for axis indicator
 *
 * Coordinate system: X=Red, Y=Blue, Z=Green (up)
 * Domain: [0,1]³ with Z as vertical axis
 */

import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import '@kitware/vtk.js/Rendering/Profiles/Volume';

import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkPlane from '@kitware/vtk.js/Common/DataModel/Plane';
import vtkImageSlice from '@kitware/vtk.js/Rendering/Core/ImageSlice';
import vtkImageResliceMapper from '@kitware/vtk.js/Rendering/Core/ImageResliceMapper';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
import vtkPiecewiseFunction from '@kitware/vtk.js/Common/DataModel/PiecewiseFunction';
import vtkVolumeMapper from '@kitware/vtk.js/Rendering/Core/VolumeMapper';
import vtkVolume from '@kitware/vtk.js/Rendering/Core/Volume';
import vtkOrientationMarkerWidget from '@kitware/vtk.js/Interaction/Widgets/OrientationMarkerWidget';
import vtkAxesActor from '@kitware/vtk.js/Rendering/Core/AxesActor';

// Cache for VTK contexts per plot
const vtk_contexts = {};

// Flag to prevent recursive camera sync updates
let is_syncing_cameras = false;

/**
 * Synchronize camera state from source to all other VTK plots.
 * Called when user interacts with any plot's camera.
 *
 * @param {string} source_container_id - The container ID that triggered the sync
 */
function sync_cameras_from(source_container_id) {
    if (is_syncing_cameras) return;

    const source_context = vtk_contexts[source_container_id];
    if (!source_context) return;

    is_syncing_cameras = true;

    const source_camera = source_context.renderer.getActiveCamera();
    const position = source_camera.getPosition();
    const focal_point = source_camera.getFocalPoint();
    const view_up = source_camera.getViewUp();
    const parallel_scale = source_camera.getParallelScale();

    // Apply to all other contexts
    for (const [container_id, context] of Object.entries(vtk_contexts)) {
        if (container_id === source_container_id) continue;

        const camera = context.renderer.getActiveCamera();
        camera.setPosition(...position);
        camera.setFocalPoint(...focal_point);
        camera.setViewUp(...view_up);
        camera.setParallelScale(parallel_scale);

        context.renderer.resetCameraClippingRange();
        context.renderWindow.render();
    }

    is_syncing_cameras = false;
}

/**
 * Set up camera synchronization listener for a VTK context.
 * Uses interactor's EndInteractionEvent to sync after user finishes dragging.
 *
 * @param {string} container_id - The container ID to set up sync for
 */
function setup_camera_sync(container_id) {
    const context = vtk_contexts[container_id];
    if (!context) return;

    const { interactor } = context;

    // Sync on end of interaction (mouse release after rotate/pan/zoom)
    interactor.onEndInteractionEvent(() => {
        sync_cameras_from(container_id);
    });

    // Also sync on mouse wheel (zoom) - these don't trigger EndInteraction
    interactor.onMouseWheel(() => {
        // Use requestAnimationFrame to let the camera update first
        requestAnimationFrame(() => {
            sync_cameras_from(container_id);
        });
    });

    // If there are existing plots, sync new plot's camera to match them
    const existing_ids = Object.keys(vtk_contexts).filter(id => id !== container_id);
    if (existing_ids.length > 0) {
        // Sync from first existing plot to the new one
        const source_context = vtk_contexts[existing_ids[0]];
        const source_camera = source_context.renderer.getActiveCamera();

        const camera = context.renderer.getActiveCamera();
        camera.setPosition(...source_camera.getPosition());
        camera.setFocalPoint(...source_camera.getFocalPoint());
        camera.setViewUp(...source_camera.getViewUp());
        camera.setParallelScale(source_camera.getParallelScale());

        context.renderer.resetCameraClippingRange();
        context.renderWindow.render();
    }
}

/**
 * Create Viridis color transfer function for slice coloring.
 * 0 values map to white, higher values follow Viridis.
 *
 * @param {number} min_val - Minimum scalar value
 * @param {number} max_val - Maximum scalar value
 * @returns {vtkColorTransferFunction} Color transfer function
 */
function create_viridis_color_function(min_val, max_val) {
    const range = max_val - min_val;
    const ctf = vtkColorTransferFunction.newInstance();

    // Start with white at min (0), then transition to Viridis
    ctf.addRGBPoint(min_val, 1.0, 1.0, 1.0);                          // White at 0
    ctf.addRGBPoint(min_val + 0.1 * range, 0.267, 0.004, 0.329);      // Dark purple
    ctf.addRGBPoint(min_val + 0.25 * range, 0.282, 0.140, 0.458);     // Purple-blue
    ctf.addRGBPoint(min_val + 0.5 * range, 0.127, 0.566, 0.550);      // Teal
    ctf.addRGBPoint(min_val + 0.75 * range, 0.741, 0.873, 0.150);     // Yellow-green
    ctf.addRGBPoint(max_val, 0.993, 0.906, 0.144);                    // Yellow

    ctf.setMappingRange(min_val, max_val);
    ctf.updateRange();

    return ctf;
}

/**
 * Create Reds color transfer function for error visualization.
 * 0 values map to white (transparent in volume), higher values to red.
 *
 * @param {number} min_val - Minimum scalar value
 * @param {number} max_val - Maximum scalar value
 * @returns {vtkColorTransferFunction} Color transfer function
 */
function create_reds_color_function(min_val, max_val) {
    const range = max_val - min_val;
    const ctf = vtkColorTransferFunction.newInstance();

    // White at 0, then transition to reds
    ctf.addRGBPoint(min_val, 1.0, 1.0, 1.0);                          // White at 0
    ctf.addRGBPoint(min_val + 0.1 * range, 0.99, 0.90, 0.85);         // Very light pink
    ctf.addRGBPoint(min_val + 0.25 * range, 0.99, 0.78, 0.71);        // Light pink
    ctf.addRGBPoint(min_val + 0.5 * range, 0.98, 0.51, 0.44);         // Salmon
    ctf.addRGBPoint(min_val + 0.75 * range, 0.84, 0.19, 0.15);        // Red
    ctf.addRGBPoint(max_val, 0.40, 0.0, 0.05);                        // Dark red

    ctf.setMappingRange(min_val, max_val);
    ctf.updateRange();

    return ctf;
}

/**
 * Create opacity function for volume rendering.
 * 0 values are transparent, higher values become more opaque.
 *
 * @param {number} min_val - Minimum scalar value
 * @param {number} max_val - Maximum scalar value
 * @returns {vtkPiecewiseFunction} Opacity transfer function
 */
function create_volume_opacity_function(min_val, max_val) {
    const range = max_val - min_val;
    const ofun = vtkPiecewiseFunction.newInstance();

    // Transparent at 0, then more opaque for visible values
    ofun.addPoint(min_val, 0.0);                    // Fully transparent at 0
    ofun.addPoint(min_val + 0.02 * range, 0.0);    // Still transparent very near 0
    ofun.addPoint(min_val + 0.05 * range, 0.3);    // Start becoming visible
    ofun.addPoint(min_val + 0.2 * range, 0.5);     // More visible
    ofun.addPoint(min_val + 0.5 * range, 0.7);     // Fairly opaque
    ofun.addPoint(max_val, 0.85);                   // Nearly opaque at max

    return ofun;
}

/**
 * Create an HTML colorbar element overlay for VTK visualization.
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
        right: 10px;
        top: 50%;
        transform: translateY(-50%);
        width: 20px;
        height: 60%;
        display: flex;
        flex-direction: column;
        align-items: center;
        pointer-events: none;
        z-index: 10;
    `;

    // Create gradient bar
    const gradientBar = document.createElement('div');
    gradientBar.className = 'colorbar-gradient';

    // Build CSS gradient based on colorscale (matching Plotly exactly)
    // Gradient goes top to bottom: max value (top) -> min value (bottom)
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
        // Viridis (top to bottom: yellow -> teal -> purple -> white at 0)
        gradientStops = `
            rgb(253, 231, 37) 0%,
            rgb(189, 223, 38) 25%,
            rgb(32, 144, 140) 50%,
            rgb(72, 36, 117) 75%,
            rgb(68, 1, 84) 90%,
            rgb(255, 255, 255) 100%
        `;
    }

    gradientBar.style.cssText = `
        width: 12px;
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
        font-size: 10px;
        color: #333;
        margin-bottom: 4px;
        font-family: monospace;
    `;

    // Create min label (bottom)
    const minLabel = document.createElement('div');
    minLabel.className = 'colorbar-label';
    minLabel.textContent = format_colorbar_value(min_val);
    minLabel.style.cssText = `
        font-size: 10px;
        color: #333;
        margin-top: 4px;
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
 * Convert 3D grid array to vtkImageData.
 *
 * @param {Array<Array<Array<number>>>} grid_3d - 3D nested array [x][y][z]
 * @param {number} nodes_per_dim - Number of nodes in each dimension
 * @returns {vtkImageData} VTK image data object
 */
function create_vtk_image_data(grid_3d, nodes_per_dim) {
    const imageData = vtkImageData.newInstance();
    imageData.setDimensions(nodes_per_dim, nodes_per_dim, nodes_per_dim);
    imageData.setSpacing(
        1.0 / (nodes_per_dim - 1),
        1.0 / (nodes_per_dim - 1),
        1.0 / (nodes_per_dim - 1)
    );
    imageData.setOrigin(0, 0, 0);

    // Flatten 3D grid to 1D array for VTK (VTK uses x-fastest ordering)
    const flat_values = new Float32Array(nodes_per_dim ** 3);
    let idx = 0;
    for (let k = 0; k < nodes_per_dim; k++) {
        for (let j = 0; j < nodes_per_dim; j++) {
            for (let i = 0; i < nodes_per_dim; i++) {
                flat_values[idx++] = grid_3d[i][j][k];
            }
        }
    }

    const scalars = vtkDataArray.newInstance({
        values: flat_values,
        numberOfComponents: 1,
        name: 'solution'
    });
    imageData.getPointData().setScalars(scalars);

    return imageData;
}

/**
 * Find min and max values in a 3D grid.
 *
 * @param {Array<Array<Array<number>>>} grid_3d - 3D nested array
 * @returns {Object} Object with min and max properties
 */
function find_grid_range_3d(grid_3d) {
    let min_val = Infinity;
    let max_val = -Infinity;

    for (const plane of grid_3d) {
        for (const row of plane) {
            for (const val of row) {
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
        }
    }

    return { min: min_val, max: max_val };
}

/**
 * Create a 3D visualization with bisecting slice plane and volume rendering below.
 *
 * @param {string} container_id - DOM element ID for the plot container
 * @param {Object} data_3d - Object with values (3D array) and nodes_per_dim
 * @param {Object} options - Optional configuration
 */
export function create_volume_plot(container_id, data_3d, options = {}) {
    const {
        colorscale = 'viridis',
        slice_z = 1.0,  // Default: show full volume (plane at top)
        showscale = true,
        zmin = null,
        zmax = null
    } = options;

    const { values, nodes_per_dim } = data_3d;

    // Find value range
    let computed_zmin = zmin;
    let computed_zmax = zmax;
    if (computed_zmin === null || computed_zmax === null) {
        const range = find_grid_range_3d(values);
        if (computed_zmin === null) computed_zmin = range.min;
        if (computed_zmax === null) computed_zmax = range.max;
    }

    // Dispose existing context if any
    dispose_volume_plot(container_id);

    // Get container element
    const container = document.getElementById(container_id);
    if (!container) {
        console.error(`Container element not found: ${container_id}`);
        return;
    }

    // Ensure container has explicit dimensions (VTK requires non-zero size)
    const rect = container.getBoundingClientRect();
    const width = rect.width > 0 ? rect.width : 360;
    const height = rect.height > 0 ? rect.height : 360;

    // Create render window
    const genericRenderWindow = vtkGenericRenderWindow.newInstance();
    genericRenderWindow.setContainer(container);

    // Explicitly set size before any rendering
    const openGLRenderWindow = genericRenderWindow.getApiSpecificRenderWindow();
    openGLRenderWindow.setSize(width, height);

    const renderer = genericRenderWindow.getRenderer();
    const renderWindow = genericRenderWindow.getRenderWindow();
    const interactor = genericRenderWindow.getInteractor();

    // Set white background
    renderer.setBackground(1, 1, 1);

    // Create vtkImageData from grid
    const imageData = create_vtk_image_data(values, nodes_per_dim);

    // Create color transfer function
    const colorscale_lower = colorscale.toLowerCase();
    const colorTransferFunction = colorscale_lower === 'reds'
        ? create_reds_color_function(computed_zmin, computed_zmax)
        : create_viridis_color_function(computed_zmin, computed_zmax);

    console.log(`[VTK] Creating plot for ${container_id}, scalar range: [${computed_zmin}, ${computed_zmax}]`);

    // ===== SLICE PLANE (always shows solution at current z) =====

    // Create slice plane perpendicular to Z axis
    const slicePlane = vtkPlane.newInstance();
    slicePlane.setNormal(0, 0, 1);  // Perpendicular to Z axis
    slicePlane.setOrigin(0, 0, slice_z);

    // Create image slice actor using reslice mapper
    const imageMapper = vtkImageResliceMapper.newInstance();
    imageMapper.setInputData(imageData);
    imageMapper.setSlicePlane(slicePlane);
    imageMapper.setSlabThickness(0.0);  // Single slice, no thickness

    const imageSliceActor = vtkImageSlice.newInstance();
    imageSliceActor.setMapper(imageMapper);

    // Configure slice properties - transparent at zero, opaque for other values
    const range = computed_zmax - computed_zmin;
    const sliceOpacityFunction = vtkPiecewiseFunction.newInstance();
    sliceOpacityFunction.addPoint(computed_zmin, 0.0);                    // Transparent at 0
    sliceOpacityFunction.addPoint(computed_zmin + 0.02 * range, 0.0);    // Still transparent near 0
    sliceOpacityFunction.addPoint(computed_zmin + 0.05 * range, 1.0);    // Fully opaque
    sliceOpacityFunction.addPoint(computed_zmax, 1.0);                    // Fully opaque at max

    const sliceProperty = imageSliceActor.getProperty();
    sliceProperty.setRGBTransferFunction(colorTransferFunction);
    sliceProperty.setPiecewiseFunction(sliceOpacityFunction);
    sliceProperty.setColorWindow(computed_zmax - computed_zmin);
    sliceProperty.setColorLevel((computed_zmax + computed_zmin) / 2);
    sliceProperty.setUseLookupTableScalarRange(true);
    sliceProperty.setInterpolationTypeToLinear();

    renderer.addActor(imageSliceActor);

    // ===== VOLUME RENDERING (shows translucent volume BELOW the slice) =====

    // Create clipping plane for volume (clips ABOVE the slice plane)
    // VTK clips geometry on the POSITIVE side of the plane normal
    // Normal points DOWN (-Z), so geometry ABOVE the origin is clipped/hidden
    const clippingPlane = vtkPlane.newInstance();
    clippingPlane.setNormal(0, 0, -1);  // Normal points down = clips above (keeps below)
    clippingPlane.setOrigin(0, 0, slice_z);

    // Create volume opacity function (0 = transparent)
    const volumeOpacityFunction = create_volume_opacity_function(computed_zmin, computed_zmax);

    // Create volume mapper with clipping
    const volumeMapper = vtkVolumeMapper.newInstance();
    volumeMapper.setInputData(imageData);
    volumeMapper.addClippingPlane(clippingPlane);
    volumeMapper.setSampleDistance(0.5 / nodes_per_dim);  // Quality vs performance

    // Create volume actor
    const volumeActor = vtkVolume.newInstance();
    volumeActor.setMapper(volumeMapper);

    // Configure volume properties
    const volumeProperty = volumeActor.getProperty();
    volumeProperty.setRGBTransferFunction(0, colorTransferFunction);
    volumeProperty.setScalarOpacity(0, volumeOpacityFunction);
    volumeProperty.setInterpolationTypeToLinear();
    volumeProperty.setShade(false);  // No shading for scientific data
    volumeProperty.setAmbient(1.0);
    volumeProperty.setDiffuse(0.0);

    renderer.addVolume(volumeActor);

    // ===== CAMERA SETUP =====
    // Set up camera with Z as vertical axis (looking down at XY plane from above-front)
    renderer.resetCamera();
    const camera = renderer.getActiveCamera();

    // Position camera to look at the volume with Z up
    // View from front-right-above, pulled back to avoid clipping
    camera.setPosition(2.5, 2.5, 2.0);  // Camera position (further back)
    camera.setFocalPoint(0.5, 0.5, 0.5);  // Look at center of domain
    camera.setViewUp(0, 0, 1);  // Z is up
    camera.zoom(0.85);  // Good zoom level for viewing
    renderer.resetCameraClippingRange();

    // ===== ORIENTATION AXES with X, Y, Z labels =====
    // vtkAxesActor shows colored axes (X=Red, Y=Green, Z=Blue) with labels
    const axes = vtkAxesActor.newInstance();
    const orientationWidget = vtkOrientationMarkerWidget.newInstance({
        actor: axes,
        interactor: interactor,
    });
    orientationWidget.setEnabled(true);
    orientationWidget.setViewportCorner(
        vtkOrientationMarkerWidget.Corners.BOTTOM_LEFT
    );
    orientationWidget.setViewportSize(0.12);  // Compact size
    orientationWidget.setMinPixelSize(50);
    orientationWidget.setMaxPixelSize(80);

    // ===== COLORBAR =====
    // Create HTML colorbar overlay (VTK.js doesn't have a built-in colorbar widget)
    const colorbarElement = create_colorbar_element(
        container,
        computed_zmin,
        computed_zmax,
        colorscale_lower
    );

    // Initial render
    renderWindow.render();

    // Store context for later updates
    vtk_contexts[container_id] = {
        genericRenderWindow,
        openGLRenderWindow,
        renderer,
        renderWindow,
        interactor,
        slicePlane,
        clippingPlane,
        imageMapper,
        imageSliceActor,
        volumeMapper,
        volumeActor,
        imageData,
        orientationWidget,
        axes,
        colorTransferFunction,
        sliceOpacityFunction,
        volumeOpacityFunction,
        colorbarElement,
        zmin: computed_zmin,
        zmax: computed_zmax
    };

    // Set up camera synchronization with other plots
    setup_camera_sync(container_id);
}

/**
 * Reset the camera to default orientation and zoom for a volume plot.
 *
 * @param {string} container_id - DOM element ID for the plot
 */
export function reset_camera(container_id) {
    const context = vtk_contexts[container_id];
    if (!context) {
        console.error(`No VTK context found for ${container_id}`);
        return;
    }

    const { renderer, renderWindow } = context;
    const camera = renderer.getActiveCamera();

    // Reset to default camera position (same as initial setup)
    camera.setPosition(2.5, 2.5, 2.0);
    camera.setFocalPoint(0.5, 0.5, 0.5);
    camera.setViewUp(0, 0, 1);

    renderer.resetCamera();
    camera.zoom(0.85);  // Match initial zoom level
    renderer.resetCameraClippingRange();

    renderWindow.render();

    // Sync the reset camera state to all other plots
    sync_cameras_from(container_id);
}

/**
 * Update the slice plane position for an existing volume plot.
 * This is the fast operation - only updates plane origins and re-renders.
 *
 * @param {string} container_id - DOM element ID for the plot
 * @param {number} z_position - Z position of the slice plane (0 to 1)
 */
export function update_slice_position(container_id, z_position) {
    const context = vtk_contexts[container_id];
    if (!context) {
        console.error(`No VTK context found for ${container_id}`);
        return;
    }

    const { slicePlane, clippingPlane, imageMapper, renderWindow } = context;

    // Update slice plane origin (bisecting plane position)
    slicePlane.setOrigin(0, 0, z_position);

    // Update clipping plane origin (clips volume ABOVE this z)
    clippingPlane.setOrigin(0, 0, z_position);

    // Mark mapper as modified to trigger update
    imageMapper.modified();

    // Trigger render (efficient - no data re-upload)
    renderWindow.render();
}

/**
 * Dispose of a volume plot and free resources.
 *
 * @param {string} container_id - DOM element ID for the plot
 */
export function dispose_volume_plot(container_id) {
    const context = vtk_contexts[container_id];
    if (!context) return;

    // Remove colorbar element from DOM
    if (context.colorbarElement && context.colorbarElement.parentNode) {
        context.colorbarElement.remove();
    }

    // Disable orientation widget first
    if (context.orientationWidget) {
        context.orientationWidget.setEnabled(false);
        context.orientationWidget.delete();
    }
    if (context.axes) {
        context.axes.delete();
    }

    // Clean up transfer functions
    if (context.colorTransferFunction) {
        context.colorTransferFunction.delete();
    }
    if (context.sliceOpacityFunction) {
        context.sliceOpacityFunction.delete();
    }
    if (context.volumeOpacityFunction) {
        context.volumeOpacityFunction.delete();
    }

    // Clean up volume objects
    if (context.volumeActor) {
        context.volumeActor.delete();
    }
    if (context.volumeMapper) {
        context.volumeMapper.delete();
    }

    // Clean up slice objects
    context.imageSliceActor.delete();
    context.imageMapper.delete();
    context.slicePlane.delete();
    context.clippingPlane.delete();
    context.imageData.delete();
    context.genericRenderWindow.delete();

    // Remove from cache
    delete vtk_contexts[container_id];
}

/**
 * Reshape flat 1D values array to 3D grid structure.
 * Supports two orderings:
 * - 'fortran': x varies fastest (analytical solver uses np.meshgrid with order='F')
 * - 'warp': z varies fastest (Warp's native node ordering)
 *
 * @param {Array<number>} flat_values - 1D array of solution values
 * @param {number} nodes_per_dim - Number of nodes in each dimension
 * @param {string} order - Data ordering: 'fortran' or 'warp' (default: 'fortran')
 * @returns {Array<Array<Array<number>>>} 3D nested array [x][y][z]
 */
export function reshape_to_3d_grid(flat_values, nodes_per_dim, order = 'fortran') {
    const grid_3d = [];
    const n = nodes_per_dim;

    for (let i = 0; i < n; i++) {
        grid_3d[i] = [];
        for (let j = 0; j < n; j++) {
            grid_3d[i][j] = [];
            for (let k = 0; k < n; k++) {
                let index;
                if (order === 'warp') {
                    // Warp order: z varies fastest, then y, then x
                    index = k + j * n + i * n * n;
                } else {
                    // Fortran/column-major order: x varies fastest, then y, then z
                    index = i + j * n + k * n * n;
                }
                grid_3d[i][j][k] = flat_values[index];
            }
        }
    }

    return grid_3d;
}

/**
 * Compute pointwise absolute error between two 3D solution grids.
 *
 * @param {Array<Array<Array<number>>>} grid1_3d - First solution grid
 * @param {Array<Array<Array<number>>>} grid2_3d - Second solution grid
 * @returns {Array<Array<Array<number>>>} Absolute error grid
 */
export function compute_error_grid_3d(grid1_3d, grid2_3d) {
    const nx = grid1_3d.length;
    const error_3d = [];

    for (let i = 0; i < nx; i++) {
        error_3d[i] = [];
        const ny = grid1_3d[i].length;
        for (let j = 0; j < ny; j++) {
            error_3d[i][j] = [];
            const nz = grid1_3d[i][j].length;
            for (let k = 0; k < nz; k++) {
                error_3d[i][j][k] = Math.abs(grid1_3d[i][j][k] - grid2_3d[i][j][k]);
            }
        }
    }

    return error_3d;
}

/**
 * Compute L2 and L-infinity error norms for 3D grids.
 *
 * @param {Array<Array<Array<number>>>} error_grid_3d - Absolute error grid
 * @returns {Object} Object with l2 and linf error values
 */
export function compute_error_norms_3d(error_grid_3d) {
    let sum_squared = 0;
    let max_error = 0;
    let count = 0;

    for (const plane of error_grid_3d) {
        for (const row of plane) {
            for (const val of row) {
                sum_squared += val * val;
                if (val > max_error) max_error = val;
                count++;
            }
        }
    }

    return {
        l2: Math.sqrt(sum_squared / count),
        linf: max_error
    };
}

/**
 * Find the min and max values in a 3D grid.
 *
 * @param {Array<Array<Array<number>>>} grid_3d - 3D nested array of values
 * @returns {Object} Object with min and max properties
 */
export { find_grid_range_3d };
