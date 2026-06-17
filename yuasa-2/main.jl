using DataFrames
using CSV
using JSON
using QuackIO
using Dates
using Intervals
using DataInterpolations
using StatsBase
using Measurements
using Printf

# using GLMakie
using CairoMakie
using ColorSchemes

using BatteryDigitalTwin
using StaticArrays
import ComponentArrays: ComponentVector, ComponentMatrix, getaxes

include("fit-model.jl")
include("analysis.jl")
include("plot.jl")
include("dataset.jl")
include("ocv.jl")


# === Data ===

# cycling dataset
datadir = "data/data-yuasa-cycles-2/"
files = Dict(
    :cell_voltage => datadir * "cell_voltages.csv",
    :cell_soc => datadir * "cell_soc.csv",
    :module_voltage => datadir * "module_voltage.csv",
    :module_current => datadir * "module_current_average.csv",
    :derating_current => datadir * "derating_currents.csv",
    :battery_temperature => datadir * "battery_temperature.csv",
)
dateformat = dateformat"y-m-d H:M:S+00:00"
data = Dict(id => CSV.File(file; dateformat) |> DataFrame for (id, file) in files)
ti = Interval(DateTime("2025-12-10T14:00:20"), DateTime("2025-12-11T02:30:20"))

# measured low-power OCV reference (rig module 7 ≡ P1M9; accurate oscilloscope probe)
df_dch = read_parquet(DataFrame, "data/yuasa-ocv-test/combined_log_20260131_200948.parquet")
df_chg = read_parquet(DataFrame, "data/yuasa-ocv-test/combined_log_20260201_181000.parquet")


# === Model fits ===
# parametrize ECM and OCV + R1 GPs for every cell and module

cell_ids = [(; p, m, c) for p in 1:3, m in 1:9, c in 1:12] |> vec |> sort
cell_ϑ = load_hyperparams(datadir * "cell_hyperparams.json", cell_ids, id -> "$(id.p)_$(id.m)_$(id.c)")
@time (; cell_models, cell_sols) = fit_cells(data, cell_ϑ, ti, cell_ids);

module_ids = [(; p, m) for p in 1:3, m in 1:9] |> vec |> sort
module_ϑ = load_hyperparams(datadir * "module_hyperparams.json", module_ids, id -> "$(id.p)_$(id.m)")
@time (; module_models, module_sols) = fit_modules(data, module_ϑ, ti, module_ids);

# open-loop run: frozen parameters, voltage prediction without correction step
cell_sols_ol = eval_models(cell_models, cell_sols, cell_ids);
module_sols_ol = eval_models(module_models, module_sols, module_ids);

# closed-loop run: frozen ECM parameters, 2-state EKF estimates charge + RC voltage
θ_soc = (; q = (; σ0 = 1.0e-3, σ1 = 0.5e-5), rc = (; σ0 = 1.0e-4, σ1 = 1.0e-4))
cell_soc = fit_soc_models(cell_models, cell_sols, cell_ids; θ = θ_soc);
module_soc = fit_soc_models(module_models, module_sols, module_ids; θ = θ_soc);

# reconstructed OCV posteriors
cell_ocvs = [gp_ocv(cell_models[id], cell_sols[id]) for id in cell_ids]
module_ocvs = [gp_ocv(module_models[id], module_sols[id]) for id in module_ids]


# === Voltage accuracy ===
# open-/closed-loop voltage prediction; RMSE/MAE/quantiles over cells & modules.
# Three runs:
# fit — innovations during training, inflated by the GP warm-up transient
# ol  — pure simulation, no feedback: errors accumulate (predictive performance)
# soc — one-step-ahead with SOC feedback: model-shape mismatch only

v_runs = (
    fit = (; cell = (; models = cell_models, sols = cell_sols), mod = (; models = module_models, sols = module_sols)),
    ol = (; cell = (; models = cell_models, sols = cell_sols_ol), mod = (; models = module_models, sols = module_sols_ol)),
    soc = (; cell = cell_soc, mod = module_soc),
);
df_v_runs = calc_v_run_summary(v_runs, cell_ids, module_ids)

# figures show the open-loop run: predictive performance is what we want to measure
df_v_cell = calc_v_summary(v_runs.ol.cell.models, v_runs.ol.cell.sols, cell_ids)
df_v_module = calc_module_v_summary(v_runs.ol.cell, v_runs.ol.mod, module_ids)


# === SOH estimation ===
# composite OCV curve from all cells / all modules → SOH and initial SOC from the fit.
# inhomogeneity: cell-to-cell spread within each module; unusable capacity split into
# irreversible (SOH spread) and reversible (SOC imbalance).

cell_fit = fit_composite_ocv(cell_ocvs, (v_low = 3.45, soc_low = 0.05, v_high = 4.05, soc_high = 0.85))
module_fit = fit_composite_ocv(module_ocvs, (v_low = 44.0, soc_low = 0.1, v_high = 48.6, soc_high = 0.85))
df_soh = calc_module_soh_summary(cell_ids, cell_fit, module_ids, module_fit)


# === OCV reconstruction validation ===
# Validate the reconstructed cell OCVs against the oscilloscope-measured OCV from the
# low-power cycling experiment (yuasa-ocv-test). Reference module is P1M9 here ≡ rig module 7
# there (the only module with the accurate current probe). Each dataset is decomposed into a
# shared shape + per-cell (SOH, SOC) and compared on one global gauge (see ocv.jl); per-cell
# affine alignment is NOT used (it manufactures artifacts).

p1m9_ids = [(; p = 1, m = 9, c) for c in 1:12]
reconstructed = [gp_ocv(cell_models[id], cell_sols[id]) for id in p1m9_ids];

# measured midline OCV: low-power charge/discharge, oscilloscope current, rig module 7
fcs = [clean_ocv(df_chg, (; m = 7, c); dch = false, current_col = "tek") for c in 1:12];
fds = [clean_ocv(df_dch, (; m = 7, c); dch = true, current_col = "tek") for c in 1:12];
measured = [average_charge_discharge(fcs[c], fds[c]) for c in 1:12];  # (; q, μ)

ocv_val = validate_cell_ocvs(measured, reconstructed)
eval_cell_ocv_validation(ocv_val)

# usable SOC window + reference-OCV curve from the measured composite (SOC → V)
meas_composite = LinearInterpolation(ocv_val.meas_comp.v_grid, ocv_val.meas_comp.soc_grid; extrapolation = ExtrapolationType.Constant)
eval_soc_range(meas_composite)
ref_curve = rescale_composite_ocv(ocv_val.meas_comp, (v_low = 3.45, soc_low = 0.15, v_high = 4.05, soc_high = 0.95))
# write_table("data/yuasa-ocv-test/reference_ocv.csv", DataFrame(v = ref_curve.v_grid, soc = ref_curve.soc_grid))


# === SOC estimation ===
# soc(t) = s0 + q(t) / Q from the closed-loop runs, with (Q, s0) from the composite-OCV fit.
# discrepancy between module SOC (module model) and aggregated cell SOC.

tg = 0:60:45000  # common time grid / s

soc_cell = calc_soc_trajectories(cell_soc.models, cell_soc.sols, cell_fit, cell_ids; tg)
soc_module = calc_soc_trajectories(module_soc.models, module_soc.sols, module_fit, module_ids; tg)

soc_err = calc_soc_error(soc_cell, soc_module, cell_fit, cell_ids, module_ids)
df_soc = calc_module_soc_summary(soc_err, module_ids)


# === Figures ===
# Semantic names; trailing comment is the paper figure number.

# dataset
fig_dataset = plot_dataset_overview(data; id_norm = (3, 7), id_out = (3, 5))  # Fig 1
fig_cell_voltages = plot_cell_voltage_system(data)                            # Fig S1
fig_data_resolution = plot_data_resolution(data)                             # Fig S2

# ECM parameters
fig_ecms = plot_ecms_comparison(cell_models, cell_sols, module_models, module_sols; n_mod = 12)  # Fig 2

# voltage accuracy
fig_sim_example = let id = (; p = 1, m = 1, c = 1)  # Fig 7
    fig = plot_sim(cell_models[id], cell_sols[id])
    fig.content[1].title = "Cell p=$(id.p) m=$(id.m) c=$(id.c)"
    fig
end
fig_cell_rmse = plot_cell_v_rmse(df_v_cell)        # Fig 8
fig_module_rmse = plot_module_v_rmse(df_v_module)  # Fig 9

# SOH
fig_cell_soh = plot_cell_soh(cell_fit, cell_ocvs)  # Fig 3
fig_module_soh = plot_module_summary(df_soh)        # Fig 4

# OCV reconstruction validation
fig_current_sources = compare_current_sources(df_dch)  # Fig S4
fig_ocv_cleaning = plot_ocv_cleaning(df_chg, (; m = 7, c = 1); dch = false)  # diagnostic
fig_ocv_extrap = plot_ocv_extrapolation(meas_composite)                             # Fig 12
fig_ocv_cells = plot_cell_ocv_validation(ocv_val, measured, reconstructed)          # Fig 10

# SOC estimation
fig_soc_comparison = let  # Fig 5
    df_norm = calc_module_soc((; p = 3, m = 7), tg, soc_cell, soc_module, cell_fit, cell_ids, module_ids)
    df_out = calc_module_soc((; p = 3, m = 5), tg, soc_cell, soc_module, cell_fit, cell_ids, module_ids)
    plot_soc_comparison(df_norm, df_out; titles = ("p=3 m=7", "p=3 m=5"))
end
fig_soc_discrepancy = plot_soc_discrepancy(soc_err)            # Fig 6
fig_soc_heatmap = plot_soc_discrepancy_heatmap(tg, soc_err)    # Fig S3
