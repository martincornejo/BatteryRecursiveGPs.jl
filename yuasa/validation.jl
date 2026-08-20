using YuasaAnalysis
using BatteryRecursiveGPs

using DataFrames
using QuackIO
using CairoMakie


# Validation of the reconstructed cell OCV curves against the low-power rig reference. The rig
# carries cycling phase 3 (P3) in module order (rig module m ≡ P3Mm, cell index identical; see
# reference.jl), so all nine rig modules are compared with their cycling counterparts and the
# results pooled over the eight comparable ones.


# === Data ===

# measured reference: the same build as reference.jl (all 108 rig cells, one composite)
valdir = "yuasa/data/validation/"
df_dch = read_parquet(DataFrame, valdir * "combined_log_20260131_200948.parquet")
df_chg = read_parquet(DataFrame, valdir * "combined_log_20260201_181000.parquet")

rig_ids = [(; m, c) for m in 1:9 for c in 1:12]
(; fcs, fds, measured) = build_reference_curves(df_chg, df_dch, rig_ids)
meas_comp = fit_composite_ocv(measured; uq = true)

# The model side is NOT refitted here: `main.jl` exports every cell's (Q_cell, s0) and OCV curve
# from its 324-cell composite fit, so the validation compares the numbers the paper reports. Run
# `main.jl` with `export_json = true` once to (re)generate the file.
rgp = load_validation_export(valdir * "validation_cells.json")

# per-cell (Q_cell, s0) for the reference, on the SAME anchor convention main.jl used
meas_abs = rescale_composite_ocv(meas_comp, rgp.refs)


# === Validation ===
# Goal 1 — shape: does the reconstruction give the right OCV curve, cell by cell?
# Goal 2 — SOH/SOC: does that curve carry the information the extraction needs?
# Per module (rig m ↔ P3Mm), then pooled. P3M5 is run but not comparable: its cell 12 was
# replaced between the two experiments (38.6 Ah in cycling, 74.5 Ah on the rig).

modules = 1:9
labels = ["P3M$m" for m in modules]
vals = map(modules) do m
    ir = findall(id -> id.m == m, rig_ids)                        # rig cells of module m, c = 1:12
    ic = [findfirst(==("3_$(m)_$(c)"), rgp.ids) for c in 1:12]   # their cycling counterparts
    validate_module(measured[ir], rgp.curves[ic], fcs[ir], fds[ir], meas_abs.Q_cell[ir], meas_abs.s0[ir], rgp.Q[ic], rgp.s0[ic])
end

comparable = [1, 2, 3, 4, 6, 7, 8, 9]   # all but P3M5
table = calc_validation_table(vals, labels)
pooled = calc_pooled_validation(vals[comparable])


# === Figures ===

fig_ocv_validation = plot_validation(vals[comparable], labels[comparable])  # Fig 10
fig_ocv_rmse = plot_validation_rmse(vals[comparable], labels[comparable])  # supplementary


# === Export ===
export_figs = false

if export_figs
    save("figs/ocv-validation.pdf", fig_ocv_validation)
    save("figs/ocv-validation-rmse.pdf", fig_ocv_rmse)
end
