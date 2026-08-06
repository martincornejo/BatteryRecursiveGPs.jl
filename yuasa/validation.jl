using YuasaAnalysis
using BatteryRecursiveGPs

using DataFrames, Dates, Intervals
using QuackIO
using DataInterpolations
using CairoMakie


# === Data ===

datadir = "yuasa/data/cycles/"
paramdir = "yuasa/data/hyperparams/"
data = load_dataset(datadir; signals = (:cell_voltage, :module_current, :battery_temperature))
ti = Interval(DateTime("2025-12-10T14:00:20"), DateTime("2025-12-11T02:30:20"))

# measured low-power OCV reference (rig module 7 ≡ P1M9; accurate oscilloscope probe)
df_dch = read_parquet(DataFrame, "yuasa/data/ocv-test/combined_log_20260131_200948.parquet")
df_chg = read_parquet(DataFrame, "yuasa/data/ocv-test/combined_log_20260201_181000.parquet")


# === Model fits ===
# reference module only — the validation compares against P1M9, so the other 312 cells
# would be fitted and discarded.

p1m9_ids = [(; p = 1, m = 9, c) for c in 1:12]
cell_ϑ = load_hyperparams(paramdir * "cell_hyperparams.json", p1m9_ids, id -> "$(id.p)_$(id.m)_$(id.c)")
@time (; cell_models, cell_sols) = fit_cells(data, cell_ϑ, ti, p1m9_ids);

reconstructed = [gp_ocv(cell_models[id], cell_sols[id]) for id in p1m9_ids];


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


# === OCV reconstruction validation ===
# per-cell SOH (Q_cell), initial SOC (s0) and OCV residual on the common gauge

ocv_val = validate_cell_ocvs(measured, reconstructed)
df_ocv_val = calc_cell_ocv_validation(ocv_val)          # per-cell SOH %, s0, residual (mV)
ocv_summary = calc_ocv_validation_summary(ocv_val)      # SOH corr, SOC corr, median residual

ref_curve = rescale_composite_ocv(meas_comp, (v_low = 3.45, soc_low = 0.15, v_high = 4.05, soc_high = 0.95))


# === Figures ===

fig_ocv_extrap = plot_ocv_extrapolation(meas_composite)                    # Fig 12
fig_ocv_cells = plot_cell_ocv_validation(ocv_val, measured, reconstructed)  # Fig 10


# === Export ===
export_csv = false
export_figs = false

if export_csv
    df_ocv = DataFrame(v = ref_curve.v_grid, soc = ref_curve.soc_grid)
    write_table("yuasa/data/ocv-test/reference_ocv.csv", df_ocv)
end

if export_figs
    save("figs/ocv-validation.pdf", fig_ocv_cells)
end
