using YuasaAnalysis
using BatteryRecursiveGPs

using DataFrames
using QuackIO
using DataInterpolations
using CairoMakie
using JSON


# The low-power rig reference: the measured OCV every absolute capacity claim is scaled against.
# It is upstream of everything — `main.jl` reads the anchor points from here, `validation.jl`
# rebuilds the same measured curves — so it depends on nothing but the rig data.
#
# The rig carries cycling PHASE 3 (P3) in module order: rig module m ≡ P3Mm, cell index c
# identical (established by fingerprinting per-cell (Q, s0) patterns; NOT P1M9 — the arrangement
# changed between the experiments). P3M5 is not comparable: its cell 12 was replaced in between.


# === Data ===

valdir = "yuasa/data/validation/"
df_dch = read_parquet(DataFrame, valdir * "combined_log_20260131_200948.parquet")
df_chg = read_parquet(DataFrame, valdir * "combined_log_20260201_181000.parquet")


# === Measured low-power OCV ===
# Cleaned midline OCV, all nine modules × 12 cells, from the BMS current (quieter and better
# zeroed than the tek probe). Each module has ONE current sensor, so its 12 cells share a charge
# axis and the per-cell (Q_cell, s0) are inferred under the shared-shape assumption, not measured.
rig_ids = [(; m, c) for m in 1:9 for c in 1:12]
(; fcs, fds, measured) = build_reference_curves(df_chg, df_dch, rig_ids)  # measured: midline (; q, μ)

# one composite over all 108 cells (per-module composites differ from it by ≤0.3 % in Q_cell)
meas_comp = fit_composite_ocv(measured; uq = true)


# === Absolute SOC gauge ===
# `V_window` is the whole convention: the datasheet defines 4.1 V as 100 % SOC, the composite is
# linearly extrapolated to both limits, and the two anchor points that set the absolute capacity
# scale are READ OFF that gauge rather than asserted. They act as one global gain in
# `rescale_composite_ocv`, so every relative SOH statement is invariant to them and every
# absolute one is theirs.

V_window = (2.9, 4.1)  # datasheet: 4.1 V is 100 % SOC
meas_composite = LinearInterpolation(meas_comp.v_grid, meas_comp.soc_grid; extrapolation = ExtrapolationType.Constant)
soc_range = eval_soc_range(meas_composite; V_min = V_window[1], V_max = V_window[2])

gauge = soc_gauge(meas_composite; V_min = V_window[1], V_max = V_window[2])
refs_ref = (; v_low = 3.62, soc_low = gauge.soc_of_v(3.62), v_high = 4.05, soc_high = gauge.soc_of_v(4.05))

ref_curve = rescale_composite_ocv(meas_comp, refs_ref)


# === Figures ===

fig_ocv_extrap = plot_ocv_extrapolation(meas_composite)  # Fig 12


# === Export ===
# `reference_refs.json` is what `main.jl` consumes; regenerate it whenever the reference build
# or `V_window` changes, then re-run `main.jl` (moves every absolute capacity by a common
# factor, no relative result).

export_json = false
export_csv = false
export_figs = false

if export_json
    write(valdir * "reference_refs.json", JSON.json(refs_ref, 2))
end

if export_csv
    write_table(valdir * "reference_ocv.csv", DataFrame(v = ref_curve.v_grid, soc = ref_curve.soc_grid))
end

if export_figs
    save("figs/ocv-extrapolation.pdf", fig_ocv_extrap)
end
