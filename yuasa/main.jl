using YuasaAnalysis        # the analysis module (src/): fitting, OCV validation, metrics, figures
using BatteryRecursiveGPs   # gp_ocv, fit_composite_ocv, rescale_composite_ocv

using CSV, DataFrames, Dates, Intervals
using QuackIO              # read_parquet
using DataInterpolations   # LinearInterpolation, ExtrapolationType
using StatsBase            # reconstruct, mean
using Measurements         # Measurements.value
using CairoMakie           # Makie backend + figure display/manipulation


# === Data ===

# cycling dataset
datadir = "yuasa/data/cycles/"
paramdir = "yuasa/data/hyperparams/"
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
let df = CSV.File(datadir * "oscilloscope_p1_m9.csv"; dateformat = dateformat"y-m-dTH:M:S.sss+00:00") |> DataFrame
    df.timestamp_utc = floor.(df.timestamp_utc, Second(1))
    data[:oscilloscope_current] = combine(groupby(df, :timestamp_utc), :MEAS1 => mean => :MEAS1)
end
ti = Interval(DateTime("2025-12-10T14:00:20"), DateTime("2025-12-11T02:30:20"))

# measured low-power OCV reference (rig module 7 ≡ P1M9; accurate oscilloscope probe)
df_dch = read_parquet(DataFrame, "yuasa/data/ocv-test/combined_log_20260131_200948.parquet")
df_chg = read_parquet(DataFrame, "yuasa/data/ocv-test/combined_log_20260201_181000.parquet")


# === Model fits ===
# parametrize ECM and OCV + R1 GPs for every cell and module

cell_ids = [(; p, m, c) for p in 1:3, m in 1:9, c in 1:12] |> vec |> sort
cell_ϑ = load_hyperparams(paramdir * "cell_hyperparams.json", cell_ids, id -> "$(id.p)_$(id.m)_$(id.c)")
@time (; cell_models, cell_sols) = fit_cells(data, cell_ϑ, ti, cell_ids);

module_ids = [(; p, m) for p in 1:3, m in 1:9] |> vec |> sort
module_ϑ = load_hyperparams(paramdir * "module_hyperparams.json", module_ids, id -> "$(id.p)_$(id.m)")
@time (; module_models, module_sols) = fit_modules(data, module_ϑ, ti, module_ids);

# open-loop run: frozen parameters, voltage prediction without correction step
cell_sols_ol = eval_models(cell_models, cell_sols, cell_ids);
module_sols_ol = eval_models(module_models, module_sols, module_ids);

# closed-loop run: frozen ECM parameters, 2-state EKF estimates charge + RC voltage
# tuning in physical units: q in Ah, rc in V. Charge noise is scope-invariant (cells and
# module see the same series current); the RC voltage noise is module-scale (12 cells).
θ_soc_cell = (; q = (; σ0 = 0.3, σ1 = 1.5e-3), rc = (; σ0 = 25.0e-3, σ1 = 50.0e-6))
θ_soc_module = (; q = (; σ0 = 0.3, σ1 = 1.5e-3), rc = (; σ0 = 12 * 25.0e-3, σ1 = 12 * 50.0e-6))
cell_soc = fit_soc_models(cell_models, cell_sols, cell_ids; θ = θ_soc_cell);
module_soc = fit_soc_models(module_models, module_sols, module_ids; θ = θ_soc_module);

# reconstructed OCV posteriors
cell_ocvs = [gp_ocv(cell_models[id], cell_sols[id]) for id in cell_ids]
module_ocvs = [gp_ocv(module_models[id], module_sols[id]) for id in module_ids]


# === Voltage accuracy ===
# Predictive voltage accuracy is reported on the open-loop run: frozen parameters, no
# correction step, errors accumulate. (The fit run's innovations are inflated by the GP
# warm-up transient; the soc run corrects state every step and only shows model-shape
# mismatch — neither measures prediction.)
ol_cell = (; models = cell_models, sols = cell_sols_ol)
ol_mod = (; models = module_models, sols = module_sols_ol)
df_v_cell = calc_v_summary(ol_cell.models, ol_cell.sols, cell_ids)
df_v_module = calc_module_v_summary(ol_cell, ol_mod, module_ids)


# === measured low-power OCV reference ===
# Cleaned midline OCV from the rig (module 7 ≡ P1M9, accurate oscilloscope probe). Its
# composite, extrapolated to the full voltage window, fixes the absolute SOC gauge used by
# the SOH fit below and by the OCV-validation section.
fcs = [clean_ocv(df_chg, (; m = 7, c); dch = false, current_col = "tek") for c in 1:12];
fds = [clean_ocv(df_dch, (; m = 7, c); dch = true, current_col = "tek") for c in 1:12];
measured = [average_charge_discharge(fcs[c], fds[c]) for c in 1:12];  # (; q, μ)

V_window = (2.9, 4.1)
meas_comp = fit_composite_ocv(measured; uq = false)
meas_composite = LinearInterpolation(meas_comp.v_grid, meas_comp.soc_grid; extrapolation = ExtrapolationType.Constant)
soc_range = eval_soc_range(meas_composite; V_min = V_window[1], V_max = V_window[2])
# absolute SOC (0 at V_min, 1 at V_max) of the measured OCV at a voltage, linearly
# extrapolated beyond the measured range — the reference anchors for the composite-OCV gauge.
ref_soc(v) = (soc_range.soc_of_v(v) - soc_range.soc_at_Vmin) / soc_range.soc_full

V_window = (2.75, 4.1)
cell_fit = fit_composite_ocv(cell_ocvs)
cell_composite = LinearInterpolation(cell_fit.v_grid, cell_fit.soc_grid)
soc_range = eval_soc_range(cell_composite; V_min = V_window[1], V_max = V_window[2])
ref_soc(v) = (soc_range.soc_of_v(v) - soc_range.soc_at_Vmin) / soc_range.soc_full


# === SOH estimation ===
# composite OCV curve from all cells / all modules → SOH and initial SOC from the fit.
# inhomogeneity: cell-to-cell spread within each module; unusable capacity split into
# irreversible (SOH spread) and reversible (SOC imbalance).

# cell_fit = fit_composite_ocv(cell_ocvs, (v_low = 3.45, soc_low = 0.06, v_high = 4.05, soc_high = 0.9))
cell_fit = fit_composite_ocv(cell_ocvs, (v_low = 3.62, soc_low = 0.1, v_high = 4.05, soc_high = 0.9))
module_fit = fit_composite_ocv(module_ocvs, (v_low = 43.44, soc_low = 0.1, v_high = 48.6, soc_high = 0.9))
df_soh = calc_module_soh_summary(cell_ids, cell_fit, module_ids, module_fit)

df_soh_cell = map(cell_fit.Q_cell, cell_ids) do Q, id
    (; p, m, c) = id
    soh = Q
    (; p, m, c, soh)
end |> DataFrame

combine(
    groupby(df_soh_cell, [:p, :m]),
    :soh => mean => :soh_mean,
    :soh => std => :soh_std,
    :soh => (x -> minimum(x)) => :soh_min,
    :soh => (x -> maximum(x)) => :soh_max,
    nrow => :n,
)

df_soc_cell = map(cell_fit.s0, cell_ids) do soc, id
    (; p, m, c) = id
    (; p, m, c, soc = soc * 100)
end |> DataFrame

combine(
    groupby(df_soc_cell, [:p, :m]),
    :soc => mean => :soc_mean,
    :soc => std => :soc_std,
    :soc => (x -> minimum(x)) => :soc_min,
    :soc => (x -> maximum(x)) => :soc_max,
    :soc => (x -> maximum(x) - minimum(x)) => :soc_delta,
    nrow => :n,
)

# per-cell voltage misfit to the composite consensus OCV (mV) — fit quality of the alignment
df_comp_rmse = calc_composite_rmse(cell_fit, cell_ocvs, cell_ids)


# === OCV reconstruction validation ===
# Validate the reconstructed cell OCVs against the oscilloscope-measured OCV from the
# low-power cycling experiment (data/ocv-test). Reference module is P1M9 here ≡ rig module 7
# there (the only module with the accurate current probe). Each dataset is decomposed into a
# shared shape + per-cell (SOH, SOC) and compared on one global gauge (see ocv.jl); per-cell
# affine alignment is NOT used (it manufactures artifacts).

p1m9_ids = [(; p = 1, m = 9, c) for c in 1:12]
reconstructed = [gp_ocv(cell_models[id], cell_sols[id]) for id in p1m9_ids];

ocv_val = validate_cell_ocvs(measured, reconstructed)
eval_cell_ocv_validation(ocv_val)

ref_curve = rescale_composite_ocv(meas_comp, (v_low = 3.45, soc_low = 0.15, v_high = 4.05, soc_high = 0.95))
# write_table("yuasa/data/ocv-test/reference_ocv.csv", DataFrame(v = ref_curve.v_grid, soc = ref_curve.soc_grid))


# === SOC estimation ===
# soc(t) = s0 + q(t) / Q from the closed-loop runs, with (Q, s0) from the composite-OCV fit.
# discrepancy between module SOC (module model) and aggregated cell SOC.

tg = 0:60:45000  # common time grid / s

soc_cell = calc_soc_trajectories(cell_soc.models, cell_soc.sols, cell_fit, cell_ids; tg)
soc_module = calc_soc_trajectories(module_soc.models, module_soc.sols, module_fit, module_ids; tg)

soc_err = calc_soc_error(soc_cell, soc_module, cell_fit, cell_ids, module_ids)
df_soc = calc_module_soc_summary(soc_err, module_ids)

# charge-estimation accuracy against the oscilloscope ground truth — reference module P1M9 only
# (the sole module with a precision current probe). Empirical validation of the estimated charge:
# the EKF tracks the reference to a median SOC RMSE <1% (~2% peak), consistent across the 12 cells.
cell_idx = Dict(cell_ids .=> eachindex(cell_ids))
Q_p1m9 = [Measurements.value(cell_fit.Q_cell[cell_idx[id]]) for id in p1m9_ids]
df_q_acc = calc_charge_accuracy(cell_soc.models, cell_soc.sols, data, ti, p1m9_ids, Q_p1m9)
df_q_err = calc_charge_error(cell_soc.models, cell_soc.sols, data, ti, p1m9_ids, Q_p1m9; tg)
# write_table("yuasa/data/cycles/charge_accuracy.csv", df_q_acc)

# fault-rejection diagnostic (single reference cell): the EKF rejects a wrong initial SOC and a
# constant current-sensor bias that Coulomb counting integrates without bound; the no-fault panel
# shows CC is near-optimal otherwise. Faults in Ah/A so the CC error reads directly off the axis.
id_diag = (p = 1, m = 9, c = 2)
soc_scenarios = ((; offset = 0.0, bias = 0.0), (; offset = 3.0, bias = 0.2))  # Ah, A
soc_diag = calc_soc_diagnostic(cell_models[id_diag], cell_sols[id_diag], data, ti, id_diag.c, soc_scenarios; θ = θ_soc_cell)


# === Figures ===
# Semantic names; trailing comment is the paper figure number.

# dataset
fig_dataset = plot_dataset_overview(data; id_norm = (3, 7), id_out = (3, 5))  # Fig 1
fig_cell_voltages = plot_cell_voltage_system(data)                            # Fig S1
df_completeness = calc_data_completeness(data, ti)                           # data availability per signal
fig_data_resolution = plot_data_resolution(data; completeness = df_completeness)  # Fig S2

# ECM parameters
fig_ecms = plot_ecms_comparison(cell_models, cell_sols, module_models, module_sols; n_mod = 12)  # Fig 2

# voltage accuracy
fig_v_overview = let id = (; p = 1, m = 1, c = 1)  # Fig 7: example fit (A) + fleet accuracy (B)
    plot_v_accuracy_overview(cell_models[id], cell_sols_ol[id], df_v_cell, df_v_module; title = "Cell P$(id.p)M$(id.m)C$(id.c)")
end

# SOH
fig_cell_soh = plot_cell_soh(cell_fit, cell_ocvs)  # Fig 3
fig_module_soh = plot_module_summary(df_soh)        # Fig 4
fig_cell_soh_hist = plot_cell_soh_hist(cell_fit)

# OCV reconstruction validation
# fig_ocv_extrap = plot_ocv_extrapolation(meas_composite)                             # Fig 12
fig_ocv_cells = plot_cell_ocv_validation(ocv_val, measured, reconstructed)          # Fig 10

# SOC estimation
fig_soc_diag = plot_soc_diagnostic(soc_diag)                 # fault-rejection diagnostic (EKF vs CC)
fig_charge_error = plot_charge_error(df_q_err)                # charge accuracy vs oscilloscope (SOC %)
fig_soc_overview = let
    df_norm = calc_module_soc((; p = 3, m = 7), tg, soc_cell, soc_module, cell_fit, cell_ids, module_ids)
    df_out = calc_module_soc((; p = 3, m = 5), tg, soc_cell, soc_module, cell_fit, cell_ids, module_ids)
    plot_soc_overview(df_norm, df_out, soc_err; titles = ("Module P3M7", "Module P3M5"))
end
fig_soc_heatmap = plot_soc_discrepancy_heatmap(tg, soc_err)    # Fig S3
