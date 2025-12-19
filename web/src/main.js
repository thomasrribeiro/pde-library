/**
 * PDE Library Web Visualization - Main Entry Point
 */

import {
    generate_laplace_solution_on_grid,
    reshape_to_grid,
    reshape_to_grid_column_major,
    get_axis_values
} from './solvers/laplace.js';

import { load_solver_data, load_problem_description } from './data/loader.js';

import {
    create_heatmap,
    compute_error_grid,
    compute_error_norms,
    find_grid_range
} from './visualization/plotly-heatmap.js';

import {
    create_volume_plot,
    update_slice_position,
    reset_camera,
    dispose_volume_plot,
    reshape_to_3d_grid,
    compute_error_grid_3d,
    compute_error_norms_3d,
    find_grid_range_3d
} from './visualization/vtk-volume.js';

// Available solvers (order matters for default selection)
const AVAILABLE_SOLVERS = [
    { id: 'analytical', label: 'Analytical' },
    { id: 'warp', label: 'Warp' },
    { id: 'dolfinx', label: 'DOLFINx' }
];

// Application state
const state = {
    equation: null,
    boundary_condition: null,
    dimension: null,
    resolution: 32,
    // Dynamic solver management
    active_solvers: ['analytical'],  // Array of active solver IDs
    solver_data: {},                  // Map: solver_id -> data object
    solver_grids: {},                 // Map: solver_id -> 2D/3D grid array
    solver_grids_3d: {},              // Map: solver_id -> 3D grid array (for 3D problems)
    global_slice_z: 1.0,              // Global slice position for all 3D plots (default to top)
    plots_visible: false
};

/**
 * Check if current problem is 3D.
 */
function is_3d_problem() {
    return state.dimension === '3d';
}

/**
 * Get the default solver (analytical if available, otherwise first solver).
 */
function get_default_solver() {
    return 'analytical';
}

/**
 * Get list of solvers available to add (not already active).
 */
function get_available_solvers_to_add() {
    return AVAILABLE_SOLVERS.filter(s => !state.active_solvers.includes(s.id));
}

/**
 * Get all unique pairs of active solvers for error computation.
 * Returns array of [solver1_id, solver2_id] pairs.
 */
function get_solver_pairs() {
    const pairs = [];
    const solvers = state.active_solvers;
    for (let i = 0; i < solvers.length; i++) {
        for (let j = i + 1; j < solvers.length; j++) {
            pairs.push([solvers[i], solvers[j]]);
        }
    }
    return pairs;
}

/**
 * Get solver label by ID.
 */
function get_solver_label(solver_id) {
    const solver = AVAILABLE_SOLVERS.find(s => s.id === solver_id);
    return solver ? solver.label : solver_id;
}

/**
 * Create a solver plot wrapper element with dropdown and delete button.
 */
function create_solver_plot_element(solver_id) {
    const is_3d = is_3d_problem();

    const wrapper = document.createElement('div');
    wrapper.className = is_3d ? 'plot-wrapper plot-wrapper-3d' : 'plot-wrapper';
    wrapper.dataset.solverId = solver_id;

    const header = document.createElement('div');
    header.className = 'plot-header';

    // Create dropdown
    const select = document.createElement('select');
    select.className = 'solver-select';
    select.dataset.solverId = solver_id;

    for (const solver of AVAILABLE_SOLVERS) {
        const option = document.createElement('option');
        option.value = solver.id;
        option.textContent = solver.label;
        option.selected = solver.id === solver_id;
        select.appendChild(option);
    }

    // Create delete button (hidden if only one solver)
    const delete_btn = document.createElement('button');
    delete_btn.className = 'delete-solver-btn';
    delete_btn.innerHTML = '×';
    delete_btn.title = 'Remove solver';
    delete_btn.dataset.solverId = solver_id;
    if (state.active_solvers.length <= 1) {
        delete_btn.style.visibility = 'hidden';
    }

    header.appendChild(select);
    header.appendChild(delete_btn);

    wrapper.appendChild(header);

    // Create the plot element (same for 2D and 3D, just different class)
    const plot = document.createElement('div');
    plot.className = is_3d ? 'plot plot-3d' : 'plot';
    plot.id = `plot-${solver_id}`;
    wrapper.appendChild(plot);

    return wrapper;
}

/**
 * Create an add solver button element.
 */
function create_add_solver_button() {
    const wrapper = document.createElement('div');
    wrapper.className = 'plot-wrapper add-solver-wrapper';
    wrapper.id = 'add-solver-wrapper';

    const header = document.createElement('div');
    header.className = 'plot-header';

    // Dropdown for selecting which solver to add
    const select = document.createElement('select');
    select.className = 'add-solver-select';
    select.id = 'add-solver-select';

    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = '+ Add solver';
    placeholder.disabled = true;
    placeholder.selected = true;
    select.appendChild(placeholder);

    const available = get_available_solvers_to_add();
    for (const solver of available) {
        const option = document.createElement('option');
        option.value = solver.id;
        option.textContent = solver.label;
        select.appendChild(option);
    }

    header.appendChild(select);
    wrapper.appendChild(header);

    return wrapper;
}

/**
 * Create an error plot wrapper element.
 */
function create_error_plot_element(solver1_id, solver2_id) {
    const pair_id = `${solver1_id}-${solver2_id}`;
    const is_3d = is_3d_problem();

    const wrapper = document.createElement('div');
    wrapper.className = is_3d ? 'plot-wrapper error-plot-wrapper plot-wrapper-3d' : 'plot-wrapper error-plot-wrapper';
    wrapper.dataset.pairId = pair_id;

    const header = document.createElement('div');
    header.className = 'plot-header';

    const label = document.createElement('div');
    label.className = 'error-label';
    label.id = `error-label-${pair_id}`;

    const title = document.createElement('span');
    title.className = 'error-label-title';
    title.textContent = `${get_solver_label(solver1_id)} vs ${get_solver_label(solver2_id)}`;

    const metrics = document.createElement('span');
    metrics.className = 'error-label-metrics';
    metrics.id = `error-metrics-${pair_id}`;
    metrics.textContent = '';  // Will be filled after rendering

    label.appendChild(title);
    label.appendChild(metrics);
    header.appendChild(label);

    wrapper.appendChild(header);

    // Create the plot element (same for 2D and 3D, just different class)
    const plot = document.createElement('div');
    plot.className = is_3d ? 'plot plot-3d' : 'plot';
    plot.id = `error-plot-${pair_id}`;
    wrapper.appendChild(plot);

    return wrapper;
}

/**
 * Update visibility of all delete buttons based on solver count.
 */
function update_delete_button_visibility() {
    const delete_btns = document.querySelectorAll('.delete-solver-btn');
    const show = state.active_solvers.length > 1;
    delete_btns.forEach(btn => {
        btn.style.visibility = show ? 'visible' : 'hidden';
    });
}

/**
 * Update the add solver dropdown options.
 */
function update_add_solver_dropdown() {
    const select = document.getElementById('add-solver-select');
    if (!select) return;

    // Clear all options except placeholder
    while (select.options.length > 1) {
        select.remove(1);
    }

    const available = get_available_solvers_to_add();
    for (const solver of available) {
        const option = document.createElement('option');
        option.value = solver.id;
        option.textContent = solver.label;
        select.appendChild(option);
    }

    // Hide the add button wrapper if no solvers available
    const wrapper = document.getElementById('add-solver-wrapper');
    if (wrapper) {
        wrapper.style.display = available.length > 0 ? '' : 'none';
    }

    // Reset selection to placeholder
    select.selectedIndex = 0;
}

// Hierarchical data structure for cascading dropdowns
const pde_data = {
    laplace: {
        label: "Laplace:",
        formula: "∇²u = 0",
        dimensions: {
            "2d": {
                label: "2D",
                boundary_conditions: {
                    dirichlet: {
                        label: "Dirichlet:",
                        conditions: [
                            "Top edge: u = sin(πx)",
                            "Bottom edge: u = 0",
                            "Left edge: u = 0",
                            "Right edge: u = 0"
                        ]
                    },
                    mixed: {
                        label: "Mixed:",
                        conditions: [
                            "Top edge: u = sin(πx)",
                            "Bottom edge: ∂u/∂y = 0",
                            "Left edge: u = 0",
                            "Right edge: u = 0"
                        ]
                    }
                }
            },
            "3d": {
                label: "3D",
                boundary_conditions: {
                    dirichlet: {
                        label: "Dirichlet:",
                        conditions: [
                            "Top face (z=1): u = sin(πx)sin(πy)",
                            "Bottom face (z=0): u = 0",
                            "Side faces: u = 0"
                        ]
                    }
                }
            }
        }
    },
    poisson: {
        label: "Poisson:",
        formula: "−∇²u = 2π²sin(πx)sin(πy)",
        dimensions: {
            "2d": {
                label: "2D",
                boundary_conditions: {
                    dirichlet: {
                        label: "Dirichlet:",
                        conditions: [
                            "All edges: u = 0"
                        ]
                    }
                }
            }
        }
    }
};

/**
 * Get solution data for a solver.
 * Returns data with a `column_major` flag indicating data ordering.
 */
async function get_solver_data(solver) {
    if (solver === 'analytical') {
        // Only use JS implementation for 2D laplace with dirichlet BC
        // 3D problems and other BCs need to load from npz files
        if (state.equation === 'laplace' && state.boundary_condition === 'dirichlet' && state.dimension === '2d') {
            const solution = generate_laplace_solution_on_grid(state.resolution);
            return {
                x: Array.from(solution.x),
                y: Array.from(solution.y),
                values: Array.from(solution.values),
                resolution: solution.resolution,
                column_major: false  // JS solver uses row-major order
            };
        }
    }

    const data = await load_solver_data(
        state.equation,
        state.boundary_condition,
        state.dimension,
        solver,
        state.resolution
    );
    // Data loaded from npz files is in column-major order
    if (data) {
        data.column_major = true;
    }
    return data;
}

/**
 * Render a single solver plot (2D heatmap or 3D volume).
 */
async function render_solver_plot(solver_id) {
    const data = await get_solver_data(solver_id);

    if (!data) {
        console.error(`Failed to load data for solver: ${solver_id}`);
        return null;
    }

    // Store data in state
    state.solver_data[solver_id] = data;

    const plot_id = `plot-${solver_id}`;

    if (is_3d_problem() && data.is_3d) {
        // 3D volume rendering
        const values = data.values instanceof Float64Array || data.values instanceof Float32Array
            ? Array.from(data.values)
            : data.values;

        const nodes_per_dim = state.resolution + 1;
        // Analytical solver uses Fortran order (x varies fastest)
        // Warp and DOLFINx use their native order (z varies fastest)
        const data_order = solver_id === 'analytical' ? 'fortran' : 'warp';
        const grid_3d = reshape_to_3d_grid(values, nodes_per_dim, data_order);
        state.solver_grids_3d[solver_id] = grid_3d;

        create_volume_plot(plot_id, {
            values: grid_3d,
            nodes_per_dim: nodes_per_dim
        }, {
            slice_z: state.global_slice_z
        });

        return grid_3d;
    } else {
        // 2D heatmap rendering
        const values = data.values instanceof Float32Array ? data.values : new Float32Array(data.values);
        const reshape_fn = data.column_major ? reshape_to_grid_column_major : reshape_to_grid;
        const grid = reshape_fn(values, state.resolution);
        state.solver_grids[solver_id] = grid;

        const axis_values = get_axis_values(state.resolution);

        create_heatmap(plot_id, grid, axis_values, axis_values, {
            showscale: true
        });

        return grid;
    }
}

/**
 * Render an error plot for a pair of solvers (2D or 3D).
 */
function render_error_plot_for_pair(solver1_id, solver2_id) {
    const pair_id = `${solver1_id}-${solver2_id}`;

    if (is_3d_problem()) {
        // 3D error visualization
        const grid1 = state.solver_grids_3d[solver1_id];
        const grid2 = state.solver_grids_3d[solver2_id];

        if (!grid1 || !grid2) return;

        const error_grid_3d = compute_error_grid_3d(grid1, grid2);
        state.error_grids_3d = state.error_grids_3d || {};
        state.error_grids_3d[pair_id] = error_grid_3d;

        const range = find_grid_range_3d(error_grid_3d);
        const norms = compute_error_norms_3d(error_grid_3d);
        const nodes_per_dim = state.resolution + 1;

        create_volume_plot(`error-plot-${pair_id}`, {
            values: error_grid_3d,
            nodes_per_dim: nodes_per_dim
        }, {
            colorscale: 'Reds',
            slice_z: state.global_slice_z,
            zmin: 0,
            zmax: range.max
        });

        const metrics_el = document.getElementById(`error-metrics-${pair_id}`);
        if (metrics_el) {
            metrics_el.textContent = `L² = ${norms.l2.toExponential(2)}, L∞ = ${norms.linf.toExponential(2)}`;
        }
    } else {
        // 2D error visualization
        const grid1 = state.solver_grids[solver1_id];
        const grid2 = state.solver_grids[solver2_id];

        if (!grid1 || !grid2) return;

        const error_grid = compute_error_grid(grid1, grid2);
        const axis_values = get_axis_values(state.resolution);
        const range = find_grid_range(error_grid);
        const norms = compute_error_norms(error_grid);

        create_heatmap(`error-plot-${pair_id}`, error_grid, axis_values, axis_values, {
            colorscale: 'Reds',
            showscale: true,
            zmin: 0,
            zmax: range.max
        });

        const metrics_el = document.getElementById(`error-metrics-${pair_id}`);
        if (metrics_el) {
            metrics_el.textContent = `L² = ${norms.l2.toExponential(2)}, L∞ = ${norms.linf.toExponential(2)}`;
        }
    }
}

/**
 * Update sidebar controls visibility based on problem type.
 * Only shows controls when a full problem is defined (equation, dimension, BC all selected).
 * Shows z-slice control for 3D problems, time control for time-dependent PDEs.
 */
function update_sidebar_controls() {
    const controls_section = document.getElementById('controls-section');
    const reset_view_control = document.getElementById('reset-view-control');
    const z_slice_control = document.getElementById('z-slice-control');
    const time_control = document.getElementById('time-control');

    // Only show controls if full problem is defined
    const problem_defined = state.equation && state.dimension && state.boundary_condition;

    // TODO: Add time-dependent detection later
    const is_time_dependent = false;

    // Show controls section only if problem is defined AND (3D or time-dependent)
    if (problem_defined && (is_3d_problem() || is_time_dependent)) {
        controls_section.style.display = 'block';
    } else {
        controls_section.style.display = 'none';
    }

    // Show reset view button and z-slice control for 3D problems
    if (problem_defined && is_3d_problem()) {
        reset_view_control.style.display = 'block';
        z_slice_control.style.display = 'block';
        // Reset slider value
        const slider = document.getElementById('z-slice-slider');
        const value_display = document.getElementById('z-slice-value');
        if (slider) {
            slider.value = String(state.global_slice_z * 100);
        }
        if (value_display) {
            value_display.textContent = state.global_slice_z.toFixed(2);
        }
    } else {
        reset_view_control.style.display = 'none';
        z_slice_control.style.display = 'none';
    }

    // Show time control for time-dependent problems
    if (problem_defined && is_time_dependent) {
        time_control.style.display = 'block';
    } else {
        time_control.style.display = 'none';
    }
}

/**
 * Build the plots container with all solver plots, add button, and error plots.
 */
function build_plots_container() {
    const container = document.querySelector('.plots-container');
    container.innerHTML = '';

    // Add solver plots
    for (const solver_id of state.active_solvers) {
        container.appendChild(create_solver_plot_element(solver_id));
    }

    // Add the "add solver" button if there are solvers available to add
    if (get_available_solvers_to_add().length > 0) {
        container.appendChild(create_add_solver_button());
    }

    // Add error plots for all pairs
    const pairs = get_solver_pairs();
    for (const [solver1, solver2] of pairs) {
        container.appendChild(create_error_plot_element(solver1, solver2));
    }

    // Setup event listeners for the new elements
    setup_dynamic_event_listeners();
}

/**
 * Setup event listeners for dynamically created elements.
 */
function setup_dynamic_event_listeners() {
    // Solver dropdown change
    document.querySelectorAll('.solver-select').forEach(select => {
        select.addEventListener('change', handle_solver_change);
    });

    // Delete buttons
    document.querySelectorAll('.delete-solver-btn').forEach(btn => {
        btn.addEventListener('click', handle_delete_solver);
    });

    // Add solver dropdown
    const add_select = document.getElementById('add-solver-select');
    if (add_select) {
        add_select.addEventListener('change', handle_add_solver);
    }

}

// Debounce state for z-slice slider
let pending_z_slice_update = null;

/**
 * Perform the actual z-slice update (called via requestAnimationFrame).
 */
function perform_z_slice_update(z_position) {
    // Update all solver plots with new clipping plane position
    for (const solver_id of state.active_solvers) {
        if (state.solver_grids_3d[solver_id]) {
            update_slice_position(`plot-${solver_id}`, z_position);
        }
    }

    // Update all error plots with new clipping plane position
    const pairs = get_solver_pairs();
    for (const [solver1, solver2] of pairs) {
        const pair_id = `${solver1}-${solver2}`;
        if (state.error_grids_3d?.[pair_id]) {
            update_slice_position(`error-plot-${pair_id}`, z_position);
        }
    }

    pending_z_slice_update = null;
}

/**
 * Handle reset view button click - resets all 3D plots to default orientation and zoom.
 */
function handle_reset_view() {
    if (!is_3d_problem()) return;

    // Reset all solver plots
    for (const solver_id of state.active_solvers) {
        if (state.solver_grids_3d[solver_id]) {
            reset_camera(`plot-${solver_id}`);
        }
    }

    // Reset all error plots
    const pairs = get_solver_pairs();
    for (const [solver1, solver2] of pairs) {
        const pair_id = `${solver1}-${solver2}`;
        if (state.error_grids_3d?.[pair_id]) {
            reset_camera(`error-plot-${pair_id}`);
        }
    }
}

/**
 * Handle z-slice slider change - updates all 3D plots.
 * Uses requestAnimationFrame debouncing to prevent excessive re-renders.
 */
function handle_z_slice_change(event) {
    const slider_value = parseInt(event.target.value, 10);
    const z_position = slider_value / 100;

    // Update state immediately
    state.global_slice_z = z_position;

    // Update value display immediately (lightweight)
    const value_display = document.getElementById('z-slice-value');
    if (value_display) {
        value_display.textContent = z_position.toFixed(2);
    }

    // Debounce the expensive plot updates via requestAnimationFrame
    if (pending_z_slice_update !== null) {
        cancelAnimationFrame(pending_z_slice_update);
    }
    pending_z_slice_update = requestAnimationFrame(() => perform_z_slice_update(z_position));
}

/**
 * Handle solver dropdown change.
 */
async function handle_solver_change(event) {
    const select = event.target;
    const old_solver_id = select.dataset.solverId;
    const new_solver_id = select.value;

    if (old_solver_id === new_solver_id) return;

    // Check if new solver is already active
    if (state.active_solvers.includes(new_solver_id)) {
        // Reset to old value
        select.value = old_solver_id;
        return;
    }

    // Update state
    const index = state.active_solvers.indexOf(old_solver_id);
    state.active_solvers[index] = new_solver_id;

    // Clean up old solver data
    delete state.solver_data[old_solver_id];
    delete state.solver_grids[old_solver_id];

    // Rebuild and re-render
    build_plots_container();
    await render_all_solver_plots();
    render_all_error_plots();
    update_add_solver_dropdown();
}

/**
 * Handle delete solver button click.
 */
async function handle_delete_solver(event) {
    const solver_id = event.target.dataset.solverId;

    // Don't delete if only one solver
    if (state.active_solvers.length <= 1) return;

    // Remove from state
    const index = state.active_solvers.indexOf(solver_id);
    if (index > -1) {
        state.active_solvers.splice(index, 1);
    }

    // Clean up data
    delete state.solver_data[solver_id];
    delete state.solver_grids[solver_id];

    // Rebuild and re-render
    build_plots_container();
    await render_all_solver_plots();
    render_all_error_plots();
    update_delete_button_visibility();
    update_add_solver_dropdown();
}

/**
 * Handle add solver dropdown selection.
 */
async function handle_add_solver(event) {
    const solver_id = event.target.value;
    if (!solver_id) return;

    // Add to active solvers
    state.active_solvers.push(solver_id);

    // Rebuild and re-render
    build_plots_container();
    await render_all_solver_plots();
    render_all_error_plots();
    update_delete_button_visibility();
    update_add_solver_dropdown();
}

/**
 * Render all active solver plots.
 */
async function render_all_solver_plots() {
    const render_promises = state.active_solvers.map(solver_id => render_solver_plot(solver_id));
    await Promise.all(render_promises);
}

/**
 * Render all error plots for solver pairs.
 */
function render_all_error_plots() {
    const pairs = get_solver_pairs();
    for (const [solver1, solver2] of pairs) {
        render_error_plot_for_pair(solver1, solver2);
    }
}

/**
 * Dispose all VTK 3D volume plots to free GPU memory.
 * (2D plots use Plotly which handles its own cleanup)
 */
function dispose_all_vtk_plots() {
    // Only dispose 3D VTK plots
    if (!is_3d_problem()) return;

    // Dispose solver plots
    for (const solver_id of state.active_solvers) {
        dispose_volume_plot(`plot-${solver_id}`);
    }

    // Dispose error plots
    const pairs = get_solver_pairs();
    for (const [solver1, solver2] of pairs) {
        const pair_id = `${solver1}-${solver2}`;
        dispose_volume_plot(`error-plot-${pair_id}`);
    }
}

/**
 * Render all plots (entry point after selections are made).
 */
async function render_all_plots() {
    if (!state.equation || !state.boundary_condition || !state.dimension) {
        return;
    }

    // Dispose existing VTK plots before clearing data (frees GPU memory)
    dispose_all_vtk_plots();

    // Clear old data
    state.solver_data = {};
    state.solver_grids = {};
    state.solver_grids_3d = {};
    state.error_grids_3d = {};
    state.global_slice_z = 1.0;

    // Update sidebar controls visibility
    update_sidebar_controls();

    // Build DOM structure
    build_plots_container();

    // Render all plots
    await render_all_solver_plots();
    render_all_error_plots();

    if (!state.plots_visible) {
        document.querySelector('.plots-container').style.display = 'flex';
        state.plots_visible = true;
    }
}

/**
 * Update the interpretation in the sidebar.
 * Only shows interpretation after all selections (equation, BC, dimension) are made.
 * Loads description from description.txt file in benchmark directory.
 */
async function update_interpretation() {
    const interp_section = document.getElementById('interpretation-section');
    const interp_el = document.getElementById('physics-interpretation');

    if (state.equation && state.boundary_condition && state.dimension) {
        const description = await load_problem_description(
            state.equation,
            state.boundary_condition,
            state.dimension
        );
        if (description) {
            interp_el.textContent = description;
            interp_section.style.display = 'block';
        } else {
            interp_el.textContent = '';
            interp_section.style.display = 'none';
        }
    } else {
        interp_el.textContent = '';
        interp_section.style.display = 'none';
    }
}

/**
 * Populate the dimension dropdown based on selected equation.
 */
function populate_dim_dropdown() {
    const dim_menu = document.getElementById('dim-menu');
    const dim_dropdown = document.getElementById('dim-dropdown');
    const dim_selected = document.getElementById('dim-selected');

    dim_menu.innerHTML = '';
    dim_selected.innerHTML = '<span class="dropdown-placeholder">Select dimensionality</span>';
    state.dimension = null;

    if (!state.equation || !pde_data[state.equation]) {
        dim_dropdown.classList.add('disabled');
        return;
    }

    // Add label (non-clickable)
    const label = document.createElement('div');
    label.className = 'dropdown-label';
    label.textContent = 'Select dimension';
    dim_menu.appendChild(label);

    const dims = pde_data[state.equation].dimensions;
    for (const [key, dim] of Object.entries(dims)) {
        const item = document.createElement('div');
        item.className = 'dropdown-item';
        item.dataset.value = key;

        const title = document.createElement('div');
        title.className = 'dropdown-item-title';
        title.textContent = dim.label;
        item.appendChild(title);

        dim_menu.appendChild(item);
    }

    dim_dropdown.classList.remove('disabled');
    setup_dim_dropdown_listeners();
}

/**
 * Populate the BC dropdown based on selected dimension.
 */
function populate_bc_dropdown() {
    const bc_menu = document.getElementById('bc-menu');
    const bc_dropdown = document.getElementById('bc-dropdown');
    const bc_selected = document.getElementById('bc-selected');

    bc_menu.innerHTML = '';
    bc_selected.innerHTML = '<span class="dropdown-placeholder">Select boundary conditions</span>';
    state.boundary_condition = null;

    if (!state.equation || !state.dimension) {
        bc_dropdown.classList.add('disabled');
        return;
    }

    // Add label (non-clickable)
    const label = document.createElement('div');
    label.className = 'dropdown-label';
    label.textContent = 'Select boundary conditions';
    bc_menu.appendChild(label);

    const bcs = pde_data[state.equation].dimensions[state.dimension].boundary_conditions;
    for (const [key, bc] of Object.entries(bcs)) {
        const item = document.createElement('div');
        item.className = 'dropdown-item';
        item.dataset.value = key;

        const title = document.createElement('div');
        title.className = 'dropdown-item-title';
        title.textContent = bc.label;
        item.appendChild(title);

        const detail = document.createElement('div');
        detail.className = 'dropdown-item-detail';
        const ul = document.createElement('ul');
        for (const cond of bc.conditions) {
            const li = document.createElement('li');
            li.textContent = cond;
            ul.appendChild(li);
        }
        detail.appendChild(ul);
        item.appendChild(detail);

        bc_menu.appendChild(item);
    }

    bc_dropdown.classList.remove('disabled');
    setup_bc_dropdown_listeners();
}

/**
 * Setup equation dropdown listeners.
 */
function setup_equation_dropdown() {
    const dropdown = document.getElementById('equation-dropdown');
    const menu = document.getElementById('equation-menu');
    const selected = document.getElementById('equation-selected');

    dropdown.addEventListener('click', (e) => {
        e.stopPropagation();
        document.querySelectorAll('.dropdown-menu.open').forEach(m => {
            if (m !== menu) m.classList.remove('open');
        });
        menu.classList.toggle('open');
    });

    menu.querySelectorAll('.dropdown-item').forEach(item => {
        item.addEventListener('click', async (e) => {
            e.stopPropagation();
            const value = item.dataset.value;
            state.equation = value;

            // Update selected display
            const eq = pde_data[value];
            selected.innerHTML = `
                <div class="selected-title">${eq.label}</div>
                <div class="selected-detail">${eq.formula}</div>
            `;

            menu.querySelectorAll('.dropdown-item').forEach(i => i.classList.remove('selected'));
            item.classList.add('selected');
            menu.classList.remove('open');

            // Reset and populate dimension dropdown
            populate_dim_dropdown();
            // Reset BC dropdown
            document.getElementById('bc-dropdown').classList.add('disabled');
            document.getElementById('bc-selected').innerHTML = '<span class="dropdown-placeholder">Select boundary conditions</span>';
            state.boundary_condition = null;

            // Hide plots until full selection
            document.querySelector('.plots-container').style.display = 'none';
            state.plots_visible = false;

            // Hide interpretation and controls until full selection
            update_interpretation();
            update_sidebar_controls();
        });
    });
}

/**
 * Setup BC dropdown listeners (called after populating).
 */
function setup_bc_dropdown_listeners() {
    const dropdown = document.getElementById('bc-dropdown');
    const menu = document.getElementById('bc-menu');
    const selected = document.getElementById('bc-selected');

    // Remove old listener by cloning
    const new_dropdown = dropdown.cloneNode(true);
    dropdown.parentNode.replaceChild(new_dropdown, dropdown);

    const new_menu = new_dropdown.querySelector('.dropdown-menu');
    const new_selected = new_dropdown.querySelector('.dropdown-selected');

    new_dropdown.addEventListener('click', (e) => {
        if (new_dropdown.classList.contains('disabled')) return;
        e.stopPropagation();
        document.querySelectorAll('.dropdown-menu.open').forEach(m => {
            if (m !== new_menu) m.classList.remove('open');
        });
        new_menu.classList.toggle('open');
    });

    new_menu.querySelectorAll('.dropdown-item').forEach(item => {
        item.addEventListener('click', async (e) => {
            e.stopPropagation();
            const value = item.dataset.value;
            state.boundary_condition = value;

            const bc = pde_data[state.equation].dimensions[state.dimension].boundary_conditions[value];
            let conditions_html = '<ul>';
            for (const cond of bc.conditions) {
                conditions_html += `<li>${cond}</li>`;
            }
            conditions_html += '</ul>';

            new_selected.innerHTML = `
                <div class="selected-title">${bc.label}</div>
                <div class="selected-detail">${conditions_html}</div>
            `;

            new_menu.querySelectorAll('.dropdown-item').forEach(i => i.classList.remove('selected'));
            item.classList.add('selected');
            new_menu.classList.remove('open');

            // All selections made - render plots and show interpretation
            await render_all_plots();
            update_interpretation();
        });
    });
}

/**
 * Setup dimension dropdown listeners (called after populating).
 */
function setup_dim_dropdown_listeners() {
    const dropdown = document.getElementById('dim-dropdown');
    const menu = document.getElementById('dim-menu');
    const selected = document.getElementById('dim-selected');

    // Remove old listener by cloning
    const new_dropdown = dropdown.cloneNode(true);
    dropdown.parentNode.replaceChild(new_dropdown, dropdown);

    const new_menu = new_dropdown.querySelector('.dropdown-menu');
    const new_selected = new_dropdown.querySelector('.dropdown-selected');

    new_dropdown.addEventListener('click', (e) => {
        if (new_dropdown.classList.contains('disabled')) return;
        e.stopPropagation();
        document.querySelectorAll('.dropdown-menu.open').forEach(m => {
            if (m !== new_menu) m.classList.remove('open');
        });
        new_menu.classList.toggle('open');
    });

    new_menu.querySelectorAll('.dropdown-item').forEach(item => {
        item.addEventListener('click', async (e) => {
            e.stopPropagation();
            const value = item.dataset.value;
            state.dimension = value;

            const dim = pde_data[state.equation].dimensions[value];
            new_selected.innerHTML = `
                <div class="selected-title">${dim.label}</div>
            `;

            new_menu.querySelectorAll('.dropdown-item').forEach(i => i.classList.remove('selected'));
            item.classList.add('selected');
            new_menu.classList.remove('open');

            // Populate BC dropdown
            populate_bc_dropdown();

            // Hide plots until full selection
            document.querySelector('.plots-container').style.display = 'none';
            state.plots_visible = false;

            // Update controls visibility (hides 3D controls when switching to 2D)
            update_sidebar_controls();
        });
    });
}

/**
 * Initialize the application.
 */
async function initialize() {
    console.log('PDE Library Web Visualization initializing...');

    // Hide plots initially
    document.querySelector('.plots-container').style.display = 'none';

    // Close dropdowns when clicking outside
    document.addEventListener('click', () => {
        document.querySelectorAll('.dropdown-menu.open').forEach(m => m.classList.remove('open'));
    });

    // Setup equation dropdown (solver dropdowns are now dynamic)
    setup_equation_dropdown();

    // Setup sidebar z-slice slider (exists in HTML, just needs event listener)
    const z_slice_slider = document.getElementById('z-slice-slider');
    if (z_slice_slider) {
        z_slice_slider.addEventListener('input', handle_z_slice_change);
    }

    // Setup reset view button
    const reset_view_btn = document.getElementById('reset-view-btn');
    if (reset_view_btn) {
        reset_view_btn.addEventListener('click', handle_reset_view);
    }

    console.log('Initialization complete. Select equation, boundary condition, and dimensionality to view plots.');
}

document.addEventListener('DOMContentLoaded', initialize);
