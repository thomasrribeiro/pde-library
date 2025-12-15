/**
 * PDE Library Web Visualization - Main Entry Point
 */

import {
    generate_laplace_solution_on_grid,
    reshape_to_grid,
    reshape_to_grid_column_major,
    get_axis_values
} from './solvers/laplace.js';

import { load_solver_data } from './data/loader.js';

import {
    create_heatmap,
    compute_error_grid,
    compute_error_norms,
    find_grid_range
} from './visualization/plotly-heatmap.js';

// Application state
const state = {
    equation: null,
    boundary_condition: null,
    dimension: null,
    resolution: 32,
    plot1_solver: 'analytical',
    plot2_solver: 'warp',
    plot1_data: null,
    plot2_data: null,
    plots_visible: false
};

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
                        ],
                        interpretation: "A square whose top edge is heated according to sin(πx) and whose remaining sides are held at a temperature of 0."
                    },
                    mixed: {
                        label: "Mixed:",
                        conditions: [
                            "Top edge: u = sin(πx)",
                            "Bottom edge: ∂u/∂y = 0",
                            "Left edge: u = 0",
                            "Right edge: u = 0"
                        ],
                        interpretation: "A square whose top edge is heated according to sin(πx), whose left and right edges are held at a temperature of 0, and whose bottom edge is thermally insulated."
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
        },
        interpretation: "Picture a rubber sheet stretched over a square frame and pushed down by an invisible hand that presses hardest at the center. The colors show how much each point is displaced downward."
    }
};

/**
 * Get solution data for a solver.
 * Returns data with a `column_major` flag indicating data ordering.
 */
async function get_solver_data(solver) {
    if (solver === 'analytical') {
        // Only use JS implementation for laplace with dirichlet BC
        // Other BCs (like mixed) need to load from npz files
        if (state.equation === 'laplace' && state.boundary_condition === 'dirichlet') {
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
 * Render a single plot.
 */
async function render_plot(plot_id, solver, data_key) {
    const data = await get_solver_data(solver);

    if (!data) {
        console.error(`Failed to load data for solver: ${solver}`);
        return null;
    }

    state[data_key] = data;

    const values = data.values instanceof Float32Array ? data.values : new Float32Array(data.values);
    const reshape_fn = data.column_major ? reshape_to_grid_column_major : reshape_to_grid;
    const grid = reshape_fn(values, state.resolution);
    const axis_values = get_axis_values(state.resolution);

    create_heatmap(plot_id, grid, axis_values, axis_values, {
        showscale: true
    });

    return grid;
}

/**
 * Render the error plot.
 */
function render_error_plot(grid1, grid2) {
    if (!grid1 || !grid2) return;

    const error_grid = compute_error_grid(grid1, grid2);
    const axis_values = get_axis_values(state.resolution);
    const range = find_grid_range(error_grid);
    const norms = compute_error_norms(error_grid);

    create_heatmap('plot3', error_grid, axis_values, axis_values, {
        colorscale: 'Reds',
        showscale: true,
        zmin: 0,
        zmax: range.max
    });

    const error_label = document.querySelector('.error-label');
    if (error_label) {
        error_label.textContent = `Error: L² = ${norms.l2.toExponential(2)}, L∞ = ${norms.linf.toExponential(2)}`;
    }
}

/**
 * Render all three plots.
 */
async function render_all_plots() {
    if (!state.equation || !state.boundary_condition || !state.dimension) {
        return;
    }

    const [grid1, grid2] = await Promise.all([
        render_plot('plot1', state.plot1_solver, 'plot1_data'),
        render_plot('plot2', state.plot2_solver, 'plot2_data')
    ]);

    render_error_plot(grid1, grid2);

    if (!state.plots_visible) {
        document.querySelector('.plots-container').style.display = 'flex';
        state.plots_visible = true;
    }
}

/**
 * Update the interpretation in the sidebar.
 * Only shows interpretation after all selections (equation, BC, dimension) are made.
 * Uses BC-level interpretation if available, otherwise falls back to equation-level.
 */
function update_interpretation() {
    const interp_section = document.getElementById('interpretation-section');
    const interp_el = document.getElementById('physics-interpretation');

    if (state.equation && state.boundary_condition && state.dimension && pde_data[state.equation]) {
        // Try BC-level interpretation first
        const bc_data = pde_data[state.equation]?.dimensions?.[state.dimension]?.boundary_conditions?.[state.boundary_condition];
        if (bc_data?.interpretation) {
            interp_el.textContent = bc_data.interpretation;
        } else {
            // Fall back to equation-level interpretation
            interp_el.textContent = pde_data[state.equation].interpretation || '';
        }
        interp_section.style.display = 'block';
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

            // Hide interpretation until full selection
            update_interpretation();
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
        });
    });
}

/**
 * Set up solver dropdown listeners.
 */
function setup_solver_dropdowns() {
    document.getElementById('plot1-solver').addEventListener('change', async (e) => {
        state.plot1_solver = e.target.value;
        if (!state.plots_visible) return;

        const grid1 = await render_plot('plot1', state.plot1_solver, 'plot1_data');
        const reshape_fn2 = state.plot2_data.column_major ? reshape_to_grid_column_major : reshape_to_grid;
        const grid2 = reshape_fn2(
            state.plot2_data.values instanceof Float32Array ?
                state.plot2_data.values : new Float32Array(state.plot2_data.values),
            state.resolution
        );
        render_error_plot(grid1, grid2);
    });

    document.getElementById('plot2-solver').addEventListener('change', async (e) => {
        state.plot2_solver = e.target.value;
        if (!state.plots_visible) return;

        const reshape_fn1 = state.plot1_data.column_major ? reshape_to_grid_column_major : reshape_to_grid;
        const grid1 = reshape_fn1(
            state.plot1_data.values instanceof Float32Array ?
                state.plot1_data.values : new Float32Array(state.plot1_data.values),
            state.resolution
        );
        const grid2 = await render_plot('plot2', state.plot2_solver, 'plot2_data');
        render_error_plot(grid1, grid2);
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

    // Setup dropdowns
    setup_equation_dropdown();
    setup_solver_dropdowns();

    console.log('Initialization complete. Select equation, boundary condition, and dimensionality to view plots.');
}

document.addEventListener('DOMContentLoaded', initialize);
