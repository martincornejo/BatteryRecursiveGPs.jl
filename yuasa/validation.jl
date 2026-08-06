using YuasaAnalysis
using BatteryRecursiveGPs

using DataFrames, Dates, Intervals
using QuackIO
using DataInterpolations
using CairoMakie


# === Data ===
# The RGP side is NOT refitted here: `main.jl` exports the per-cell (Q_cell, s0) and OCV curves
# from its 324-cell composite fit, so the validation compares the numbers the paper reports.
# Run `main.jl` with `export_json = true` once to (re)generate the file.

valdir = "yuasa/data/validation/"
rgp = load_validation_export(valdir * "validation_p1m9.json")
reconstructed = rgp.curves

# measured low-power OCV reference (rig module 7 ≡ P1M9; accurate oscilloscope probe)
df_dch = read_parquet(DataFrame, valdir * "combined_log_20260131_200948.parquet")
df_chg = read_parquet(DataFrame, valdir * "combined_log_20260201_181000.parquet")


# === measured low-power OCV reference ===
# Cleaned midline OCV from the rig. `fcs`/`fds` (the charge and discharge branches) are kept:
# their separation is the reference's own ambiguity, which the shape validation compares against.
# NOTE the rig has ONE module-level current sensor, so all 12 cells share a charge axis — the
# measured (Q_cell, s0) below are inferred under the shared-shape assumption, not measured.

fcs = [clean_ocv(df_chg, (; m = 7, c); dch = false, current_col = "tek") for c in 1:12];
fds = [clean_ocv(df_dch, (; m = 7, c); dch = true, current_col = "tek") for c in 1:12];
measured = [average_charge_discharge(fcs[c], fds[c]) for c in 1:12];  # (; q, μ)

# per-cell (Q_cell, s0) for the reference, on the SAME anchor convention main.jl used
meas_comp = fit_composite_ocv(measured; uq = true)
meas_abs = rescale_composite_ocv(meas_comp, rgp.refs)


# === Validation ===
# Goal 1 — shape: does the reconstruction give the right OCV curve, cell by cell?
# Goal 2 — SOH/SOC: does that curve carry the information the extraction needs?

df_shape = calc_ocv_shape_validation(measured, reconstructed, fcs, fds, meas_abs.Q_cell, rgp.Q)
curves = calc_ocv_curves(measured, reconstructed, df_shape, meas_abs.Q_cell, meas_abs.s0)
df_soh = calc_soh_validation(meas_abs.Q_cell, meas_abs.s0, rgp.Q, rgp.s0)
summary = calc_validation_summary(df_shape, df_soh)


# === Reference-quality diagnostics ===
# How much of the full SOC window the measured experiment actually covers (it stops well short
# of both voltage limits, so the composite is extrapolated at both ends).

V_window = (2.9, 4.1)
meas_composite = LinearInterpolation(meas_comp.v_grid, meas_comp.soc_grid; extrapolation = ExtrapolationType.Constant)
soc_range = eval_soc_range(meas_composite; V_min = V_window[1], V_max = V_window[2])

ref_curve = rescale_composite_ocv(meas_comp, (v_low = 3.45, soc_low = 0.15, v_high = 4.05, soc_high = 0.95))


# === Figures ===

fig_ocv_cells = plot_cell_ocv_validation(df_shape, curves, df_soh)  # Fig 10
fig_ocv_extrap = plot_ocv_extrapolation(meas_composite)                                        # Fig 12


# === Export ===
export_csv = false
export_figs = false

if export_csv
    df_ocv = DataFrame(v = ref_curve.v_grid, soc = ref_curve.soc_grid)
    write_table(valdir * "reference_ocv.csv", df_ocv)
end

if export_figs
    save("figs/ocv-validation.pdf", fig_ocv_cells)
    save("figs/ocv-extrapolation.pdf", fig_ocv_extrap)
end
