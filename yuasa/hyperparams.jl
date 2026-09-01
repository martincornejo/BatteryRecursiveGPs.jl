using DataFrames, Dates, Intervals
using JSON

using YuasaAnalysis
using BatteryRecursiveGPs

using Distributed
nprocs() == 1 && addprocs(Sys.CPU_THREADS ÷ 2)
@everywhere using BatteryRecursiveGPs

using CairoMakie

# Two-stage distributed hyperparameter selection from a generic init.
# Stage 1: fit every unit at ϑ_init and build a clean composite reference from the units already within thresh_mV.
# Stage 2: re-fit the misfits over the ℓ grids and keep, per unit, the ℓ minimising the composite-OCV residual.


# === Data ===

datadir = "yuasa/data/cycles/"
paramdir = "yuasa/data/hyperparams/"

data = load_dataset(datadir; signals = (:cell_voltage, :module_voltage, :module_current, :battery_temperature))
ti = Interval(DateTime("2025-12-10T14:00:20"), DateTime("2025-12-11T02:30:20"))

cell_ids = [(; p, m, c) for p in 1:3, m in 1:9, c in 1:12] |> vec |> sort
module_ids = [(; p, m) for p in 1:3, m in 1:9] |> vec |> sort

zt = fit_zscore()
zt_mod = fit_zscore(12)
cell_data = Dict(id => cell_dataset(data, ti, id.p, id.m, id.c; zt) for id in cell_ids)
module_data = Dict(id => module_dataset(data, ti, id.p, id.m; zt = zt_mod) for id in module_ids)

pool = WorkerPool(workers())


# === Selection ===

ϑ_init = (; ocv = (; σ = 0.5, ℓ = 0.5), r1 = (; σ = 0.5, ℓ = 0.5))
ℓ_ocv_grid = [0.6, 0.85, 1.0, 1.5, 3.0, 7.0, 15.0]
ℓ_r1_grid = [0.3, 1.0]

@time cells = select_hyperparams(pool, cell_ids, cell_data, zt; ϑ = ϑ_init, ℓ_ocv_grid, ℓ_r1_grid, thresh_mV = 4.0);
@time modules = select_hyperparams(pool, module_ids, module_data, zt_mod; ϑ = ϑ_init, ℓ_ocv_grid, ℓ_r1_grid, thresh_mV = 50.0, n = 12);

df_selection = selection_counts(cells, modules)

fig_hyperparam_selection = plot_hyperparam_selection(
    calc_hyperparam_selection(cells), calc_hyperparam_selection(modules)
)
fig_hyperparam_scales = plot_hyperparam_scales(
    calc_scaled_hyperparams(cells, cell_data, zt),
    calc_scaled_hyperparams(modules, module_data, zt_mod),
)


# === Export ===
export_json = false
export_figs = false

if export_json
    cell_hyperparams = build_hyperparam_export(cells)
    module_hyperparams = build_hyperparam_export(modules)
    write(paramdir * "cell_hyperparams.json", JSON.json(cell_hyperparams, 2))
    write(paramdir * "module_hyperparams.json", JSON.json(module_hyperparams, 2))
end

if export_figs
    save("figs/hyperparam-selection.pdf", fig_hyperparam_selection)
    save("figs/hyperparam-scales.pdf", fig_hyperparam_scales)
end
