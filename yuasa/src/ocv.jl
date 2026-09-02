# Validation of the reconstructed cell OCV curves against the measured OCV from the low-power
# rig experiment (data/validation): measured-OCV builders, the per-module validation, and its
# plots. The rig carries cycling phase 3 in module order (rig m ≡ P3Mm; see reference.jl).


# === measured OCV builders (low-power charge/discharge, oscilloscope current) ===

function integrate_current(df; current_col, negate = true, offset = 0.0)
    transform = negate ? (x -> -(x .+ offset)) : (x -> x .+ offset)
    df_i = select(df, :timestamp_utc, current_col => transform => :i)
    dropmissing!(df_i)
    Δt = [Dates.value.(diff(df_i.timestamp_utc)); 0] * 1.0e-3  # in seconds
    df_i[!, :q] = cumsum(df_i.i .* Δt) ./ 3600  # in Ah
    return df_i
end

function find_tek_offset(df; m = 7, bms_thresh = 0.1)
    df_both = select(df, :timestamp_utc, "m$(m)_current" => :bms, :tek_m_cur_ref => :tek)
    dropmissing!(df_both)
    # Only use rest periods where BMS reads ~0; offset so that tek + offset ≈ 0 at rest
    df_rest = subset(df_both, :bms => ByRow(x -> abs(x) < bms_thresh))
    return -mean(df_rest.tek)
end

function invert_ocv(f; n_samples = 500, extrapolation = ExtrapolationType.Constant)
    q = collect(range(first(f.t), last(f.t); length = n_samples))
    v = f.(q)
    # strict monotonicity via a running maximum (comparing only against the immediate
    # predecessor leaves dips in place and the interpolation rejects the unsorted result)
    keep = Int[]
    vmax = -Inf
    for k in eachindex(v)
        v[k] > vmax && (push!(keep, k); vmax = v[k])
    end
    return LinearInterpolation(q[keep], v[keep]; extrapolation)
end

# Midline OCV: at each voltage, the mean of the two branches' charge, q̄(v) = (q_chg + q_dch)/2.
# Averaging in the voltage frame (not at fixed charge) keeps the high-SOC end and is insensitive
# to the unknown charge offset between the two experiments. Returns (; q, μ = v grid).
function average_charge_discharge(fc, fd; n_samples = 500)
    gc = invert_ocv(fc)  # q(v), charge branch
    gd = invert_ocv(fd)  # q(v), discharge branch
    v_lo = max(first(gc.t), first(gd.t))
    v_hi = min(last(gc.t), last(gd.t))
    v = collect(range(v_lo, v_hi; length = n_samples))
    q = (gc.(v) .+ gd.(v)) ./ 2
    return (; q, μ = v)
end

function clean_ocv(df, id; dch::Bool, i_thresh = 0.5, current_col = "bms")
    (; m, c) = id

    cell = @sprintf("%02d", c)
    df_cell = select(df, :timestamp_utc, "m$(m)_cell$cell" => :v)
    dropmissing!(df_cell)

    if current_col == "tek"
        df_i = integrate_current(df; current_col = "tek_m_cur_ref", negate = true, offset = find_tek_offset(df; m))
    else
        df_i = integrate_current(df; current_col = "m$(m)_current", negate = true)
    end

    q0 = minimum(df_i.q)
    df_i[!, :q] = df_i.q .- q0

    df1 = innerjoin(df_cell, df_i, on = :timestamp_utc, makeunique = true)
    sort!(df1, :timestamp_utc)

    df2 = subset(df1, :i => ByRow(<=(i_thresh) ∘ abs))
    df2[!, :q_floor] = floor.(df2.q; digits = 1)

    df3 = combine(groupby(df2, :q_floor)) do subdf
        sort(subdf, :v; rev = dch) |> first
    end
    sort!(df3, :q)

    return LinearInterpolation(df3.v, df3.q; extrapolation = ExtrapolationType.Constant)
end

"""
    build_reference_curves(df_chg, df_dch, ids; current_col = "bms") -> (; fcs, fds, measured)

The measured reference for the rig cells `ids` (`(; m, c)` each): charge/discharge branch
interpolants from `clean_ocv` and their midline `(; q, μ)` from `average_charge_discharge`. Used
by both `reference.jl` and `validation.jl`, which build the reference independently rather than
sharing state.
"""
function build_reference_curves(df_chg, df_dch, ids; current_col = "bms")
    fcs = [clean_ocv(df_chg, id; dch = false, current_col) for id in ids]
    fds = [clean_ocv(df_dch, id; dch = true, current_col) for id in ids]
    measured = [average_charge_discharge(fcs[i], fds[i]) for i in eachindex(ids)]
    return (; fcs, fds, measured)
end


# === individual-cell OCV validation (model OCV vs measured low-power OCV) ===
# `measured`/`reconstructed` are monotone v(q) curves `(; q, μ)` (Ah, V), one per cell in
# matching order. Two separable claims: SHAPE — per-cell curve comparison over each cell's own
# voltage overlap, with only the (unidentifiable) charge origin removed (`a = 1`); note in the
# plateau 1 % of capacity ≈ 2.8 mV of OCV, so shape and capacity are different tests. SOH/SOC —
# per-cell `(Q_cell, s0)` from each side's own composite decomposition (model: the 324-cell fit
# the paper reports; reference: one composite over all 108 rig cells), on a shared anchor
# convention; absolute capacity is that convention, not a measurement.


# q(v) for a monotone (; q, μ) curve — the voltage frame both datasets are compared in
function _q_of_v(c)
    o = sortperm(collect(c.μ))
    v = collect(c.μ)[o]
    q = collect(c.q)[o]
    mask = [true; diff(v) .> 1.0e-9]
    return LinearInterpolation(q[mask], v[mask]; extrapolation = ExtrapolationType.Linear)
end

# Local dV/dq (V/Ah) of a q(v) interpolant, on a coarse grid so no difference is zero. Used to
# express charge-domain quantities in mV — in the plateau this rate is ~6 mV/Ah, which is why a
# percent-level capacity difference is only a few mV of voltage.
function _dv_dq(g, v; n = 40)
    vg = collect(range(minimum(v), maximum(v); length = n))
    s = diff(vg) ./ diff(g.(vg))
    return LinearInterpolation([s; s[end]], vg; extrapolation = ExtrapolationType.Constant).(v)
end

"""
    calc_ocv_shape_validation(measured, reconstructed, fcs, fds) -> DataFrame

Per-cell OCV shape validation over each cell's own measured∩RGP voltage overlap. `fcs`/`fds`
are that cell's charge/discharge branch interpolants from `clean_ocv` (needed for the reference
floor).

Columns: `v_lo`/`v_hi` (the cell's own window), `ocv_rmse` (mV, `a = 1`: both charge axes in
physical Ah, only the origin `b` removed), `ocv_rmse_free` (mV, a scale `a_fit` refitted as
well), `floor` (mV, the reference's own ambiguity — see `calc_reference_floor`), `ratio` =
`ocv_rmse`/`floor` (the cross-cell-comparable number; the bare mV is not, since the windows
differ in width and steepness), `a_fit`, and the fitted offset `b` the figure needs to place
the curves.
"""
function calc_ocv_shape_validation(measured, reconstructed, fcs, fds; n_v = 200)
    floors = calc_reference_floor(measured, reconstructed, fcs, fds; n_v)

    return map(eachindex(measured)) do c
        gm = _q_of_v(measured[c])
        gr = _q_of_v(reconstructed[c])
        v_lo = max(minimum(gm.t), minimum(gr.t)) + 1.0e-3
        v_hi = min(maximum(gm.t), maximum(gr.t)) - 1.0e-3
        v = collect(range(v_lo, v_hi; length = n_v))
        slope = _dv_dq(gr, v)

        b = mean(gr.(v) .- gm.(v))                       # only the charge origin is removed
        a_fit, b_fit = hcat(gm.(v), ones(length(v))) \ gr.(v)

        rmse(a, b) = sqrt(mean(abs2, (gr.(v) .- (a .* gm.(v) .+ b)) .* slope .* 1000))
        r = rmse(1.0, b)
        (;
            cell = c, v_lo, v_hi,
            ocv_rmse = r, ocv_rmse_free = rmse(a_fit, b_fit),
            floor = floors[c], ratio = r / floors[c],
            a_fit, b,
        )
    end |> DataFrame
end

"""
    calc_ocv_curves(measured, reconstructed, df_shape, Q_meas, s0_meas) -> Vector

Plot-ready per-cell curves on the common SOC axis, plus the residual behind the `ocv_rmse`
summary. Reuses the charge offset `b` from `df_shape`, so there is one alignment, not two: the
RGP curve is shifted into the measured charge frame (`a = 1`) and both are then divided by that
cell's own `(Q_meas, s0_meas)` — the reference is never moved.

Each entry is `(; soc_meas, v_meas, soc_rgp, v_rgp, soc, r)` — SOC in %, voltages in V, residual
in mV over that cell's own window.

SOC rather than charge because the residual is a function of electrode state, not of absolute
charge: across cells it collapses 18 % tighter on this axis (2.66 → 2.18 mV), and plotting
against charge instead smears it by each cell's own `s0` offset (~2.6 pp spread). Note the SOC
values sit on the composite gauge — they are a labelling convention, not an absolute state of
charge (see `calc_soh_validation`).
"""
function calc_ocv_curves(measured, reconstructed, df_shape, Q_meas, s0_meas; n_v = 200)
    Q = Measurements.value.(Q_meas)
    s0 = Measurements.value.(s0_meas)
    return map(df_shape.cell) do c
        tosoc(q) = (s0[c] .+ q ./ Q[c]) .* 100
        gm = _q_of_v(measured[c])
        gr = _q_of_v(reconstructed[c])
        v = collect(range(df_shape.v_lo[c], df_shape.v_hi[c]; length = n_v))
        r = (gr.(v) .- (gm.(v) .+ df_shape.b[c])) .* _dv_dq(gr, v) .* 1000
        (;
            soc_meas = tosoc(collect(measured[c].q)), v_meas = collect(measured[c].μ),
            # RGP shifted into the measured charge frame by this cell's offset, then to SOC
            soc_rgp = tosoc(collect(reconstructed[c].q) .- df_shape.b[c]),
            v_rgp = collect(reconstructed[c].μ),
            soc = tosoc(gm.(v)), r,
        )
    end
end

"""
    calc_reference_floor(measured, reconstructed, fcs, fds) -> Vector

The measured reference's own ambiguity per cell, in mV, on the same window and axis as the OCV
residual so the two are directly comparable.

The reference OCV is the midline between a slow charge and a slow discharge branch, so its
uncertainty is HALF their separation `Δq(v) = q_dch(v) - q_chg(v)`, converted to mV by the local
dV/dq. The median of `Δq` is removed first: the two branches come from separate rig experiments
with an unknown relative charge origin, which is unidentifiable and absorbed by `s0` downstream —
leaving it in would inflate the floor (~20 mV rather than ~11 mV). What remains is the genuine
hysteresis + unrelaxed polarisation: rests are ~58 s against a slow-RC τ ≈ 800 s, so each branch
carries residual polarisation and the midline cancels only the symmetric part.

A residual below this floor is indistinguishable from a perfect reconstruction GIVEN this
reference. CAVEAT: this is a lower bound on the reference's ambiguity, not a full uncertainty
budget — it captures the dominant term but not voltage-sensor calibration or `clean_ocv`'s 0.1 Ah
binning. It reads ~11 mV where earlier notes say ~8 mV; same phenomenon, but averaged over each
cell's own comparison window rather than the full curve.
"""
function calc_reference_floor(measured, reconstructed, fcs, fds; n_v = 200)
    return map(eachindex(measured)) do c
        gm = _q_of_v(measured[c])
        gr = _q_of_v(reconstructed[c])
        gc = invert_ocv(fcs[c])
        gd = invert_ocv(fds[c])
        lo = max(minimum(gc.t), minimum(gd.t), minimum(gm.t), minimum(gr.t)) + 1.0e-3
        hi = min(maximum(gc.t), maximum(gd.t), maximum(gm.t), maximum(gr.t)) - 1.0e-3
        v = collect(range(lo, hi; length = n_v))
        Δq = gd.(v) .- gc.(v)
        mean(abs.(Δq .- median(Δq)) ./ 2 .* _dv_dq(gc, v)) * 1000
    end
end

"""
    calc_soh_validation(Q_meas, s0_meas, Q_rgp, s0_rgp) -> DataFrame

Per-cell SOH and initial-SOC comparison. Inputs are `Measurement` vectors on a shared anchor
convention (the same `refs` main.jl used), so `Q_meas`/`Q_rgp` are full-cell capacities in Ah and
are reported as such — full SOH is the quantity of interest, and the extrapolation it needs is a
limitation of the experiment's voltage coverage, not a reason to report something else.

`ΔQ` is the per-cell error after removing the common scale `k = mean(Q_rgp)/mean(Q_meas)` — a
scaling, not a shift, since the nuisances act as gains on capacity. Only `ΔQ` is a per-cell claim;
`k` itself is bounded by the instrumentation rather than measured (two current sensors on this
module disagree by 2.6 % at the same instant). Note what the anchors do and do not move: the SOC
values assigned to them scale BOTH datasets identically, so they change the absolute Ah by ~2.5 %
per 2 pp but leave `k` exactly invariant; only the anchor VOLTAGES move `k`, and only by ≤0.4 pp
per 10 mV.

`s0` is reported as a deviation from each dataset's own module mean — absolute initial SOC is not
comparable, since the two datasets are different experiments started at different charge states.
"""
function calc_soh_validation(Q_meas, s0_meas, Q_rgp, s0_rgp)
    qm = Measurements.value.(Q_meas)
    qr = Measurements.value.(Q_rgp)
    sm = Measurements.value.(s0_meas)
    sr = Measurements.value.(s0_rgp)
    k = mean(qr) / mean(qm)
    return DataFrame(
        cell = collect(eachindex(qm)),
        Q_meas = qm,
        Q_rgp = qr,
        ΔQ = qr ./ k .- qm,
        Q_meas_σ = Measurements.uncertainty.(Q_meas),
        Q_rgp_σ = Measurements.uncertainty.(Q_rgp),
        soh_meas = (qm ./ mean(qm) .- 1) .* 100,
        soh_rgp = (qr ./ mean(qr) .- 1) .* 100,
        soh_meas_σ = Measurements.uncertainty.(Q_meas) ./ mean(qm) .* 100,
        soh_rgp_σ = Measurements.uncertainty.(Q_rgp) ./ mean(qr) .* 100,
        s0_meas = (sm .- mean(sm)) .* 100,
        s0_rgp = (sr .- mean(sr)) .* 100,
        s0_meas_σ = Measurements.uncertainty.(s0_meas) .* 100,
        s0_rgp_σ = Measurements.uncertainty.(s0_rgp) .* 100,
        Δs0 = ((sr .- mean(sr)) .- (sm .- mean(sm))) .* 100,
    )
end

# The headline validation numbers. SOH/SOC are correlations of the per-cell deviations (absolute
# scale is unidentifiable); the shape claim is `n_below_floor` — how many cells disagree with the
# reference by less than the reference's own ambiguity.
function calc_validation_summary(df_shape, df_soh)
    return (;
        ocv_rmse = median(df_shape.ocv_rmse),
        ocv_rmse_free = median(df_shape.ocv_rmse_free),
        ocv_floor = median(df_shape.floor),
        n_below_floor = count(df_shape.ocv_rmse .< df_shape.floor),
        n_cells = nrow(df_shape),
        soh_cor = cor(df_soh.soh_rgp, df_soh.soh_meas),
        soh_err = sqrt(mean(abs2, df_soh.ΔQ)),
        soh_spread = std(df_soh.Q_meas),
        soc_cor = cor(df_soh.s0_rgp, df_soh.s0_meas),
        soc_err = sqrt(mean(abs2, df_soh.Δs0)),
    )
end

"""
    validate_module(measured, reconstructed, fcs, fds, Q_meas, s0_meas, Q_rgp, s0_rgp)

The per-module validation: shape table, plot-ready curves, SOH/SOC table and their summary, for
one module's cells (same order on both sides). Returns `(; df_shape, curves, df_soh, summary)`.
"""
function validate_module(measured, reconstructed, fcs, fds, Q_meas, s0_meas, Q_rgp, s0_rgp)
    df_shape = calc_ocv_shape_validation(measured, reconstructed, fcs, fds)
    curves = calc_ocv_curves(measured, reconstructed, df_shape, Q_meas, s0_meas)
    df_soh = calc_soh_validation(Q_meas, s0_meas, Q_rgp, s0_rgp)
    return (; df_shape, curves, df_soh, summary = calc_validation_summary(df_shape, df_soh))
end

# One row per module from `validate_module` results: the shape numbers, the SOH/SOC numbers, the
# module-mean capacities on both sides and their ratio `k` (the module-wide scale the per-cell
# comparison removes).
function calc_validation_table(vals, labels)
    return DataFrame(
        id = labels,
        ocv_rmse = [v.summary.ocv_rmse for v in vals],
        ocv_rmse_free = [v.summary.ocv_rmse_free for v in vals],
        ocv_floor = [v.summary.ocv_floor for v in vals],
        n_below_floor = [v.summary.n_below_floor for v in vals],
        soh_cor = [v.summary.soh_cor for v in vals],
        soh_err = [v.summary.soh_err for v in vals],
        soh_spread = [v.summary.soh_spread for v in vals],
        soc_cor = [v.summary.soc_cor for v in vals],
        soc_err = [v.summary.soc_err for v in vals],
        Q_meas = [mean(v.df_soh.Q_meas) for v in vals],
        Q_rgp = [mean(v.df_soh.Q_rgp) for v in vals],
        k = [mean(v.df_soh.Q_rgp) / mean(v.df_soh.Q_meas) for v in vals],
    )
end

# The headline numbers pooled over the comparable modules: the per-cell shape RMSE (median over
# all cells and how many lie below their own reference floor), the pooled SOC and SOH
# correlations of the module-centred deviations, and the module-level capacity agreement (mean
# offset and spread of the per-module ratio `k`, correlation of the module means).
function calc_pooled_validation(vals)
    rmse = vcat([v.df_shape.ocv_rmse for v in vals]...)
    floors = vcat([v.df_shape.floor for v in vals]...)
    ks = [mean(v.df_soh.Q_rgp) / mean(v.df_soh.Q_meas) for v in vals]
    return (;
        ocv_rmse = median(rmse), n_below_floor = count(rmse .< floors), n_cells = length(rmse),
        soc_cor = cor(vcat([v.df_soh.s0_meas for v in vals]...), vcat([v.df_soh.s0_rgp for v in vals]...)),
        soh_cor = cor(vcat([v.df_soh.soh_meas for v in vals]...), vcat([v.df_soh.soh_rgp for v in vals]...)),
        module_Q_cor = cor([mean(v.df_soh.Q_meas) for v in vals], [mean(v.df_soh.Q_rgp) for v in vals]),
        k_mean = mean(ks), k_sd = std(ks), n_modules = length(vals),
    )
end


# === validation export (main.jl → validation.jl) ===
# validation.jl must compare the SOH the paper REPORTS, which comes from the 324-cell joint fit
# in main.jl. It cannot rederive that from 12 cells, so main.jl hands over the per-cell
# parameters and curves directly.

# Per-cell `(Q_cell, s0)` + reconstructed OCV curves for `ref_ids`, taken from the full-fleet
# `comp_fit`/`ocvs` indexed by `ids`, plus the anchor convention they are expressed in.
function build_validation_export(comp_fit, ocvs, ids, ref_ids, refs)
    pos = Dict(id => i for (i, id) in enumerate(ids))
    cell(id, i) = Dict(
        "id" => id_key(id),
        "Q" => Measurements.value(comp_fit.Q_cell[i]),
        "Q_sigma" => Measurements.uncertainty(comp_fit.Q_cell[i]),
        "s0" => Measurements.value(comp_fit.s0[i]),
        "s0_sigma" => Measurements.uncertainty(comp_fit.s0[i]),
        "q" => collect(ocvs[i].q),
        "v" => collect(ocvs[i].μ),
    )
    return Dict(
        "refs" => Dict(String(k) => v for (k, v) in pairs(refs)),
        "cells" => [cell(id, pos[id]) for id in ref_ids],
    )
end

# Inverse of `build_validation_export` → (; curves, Q, s0, refs, ids)
function load_validation_export(file)
    d = JSON.parsefile(file)
    cells = d["cells"]
    return (;
        curves = [(; q = Float64.(c["q"]), μ = Float64.(c["v"])) for c in cells],
        Q = [c["Q"] ± c["Q_sigma"] for c in cells],
        s0 = [c["s0"] ± c["s0_sigma"] for c in cells],
        refs = (; (Symbol(k) => v for (k, v) in d["refs"])...),
        ids = [c["id"] for c in cells],
    )
end


# Build the linearly-extrapolated OCV interpolants from a composite SOC → V curve
# (`composite.t` = SOC grid, `composite.u` = V grid, e.g.
# `LinearInterpolation(fit.v_grid, fit.soc_grid)`). Extends the curve linearly beyond the
# observed data so any voltage maps to an SOC and vice versa. Returns `(; v_of_soc, soc_of_v)`,
# two interpolants that are mutual inverses: `v_of_soc` is SOC → V, `soc_of_v` is V → SOC.
# Single source of truth for both `eval_soc_range` and `plot_ocv_extrapolation`.
function extrapolate_ocv(composite; n_samples = 100)
    soc = collect(composite.t)
    v = collect(composite.u)
    v_of_soc = LinearInterpolation(v, soc; extrapolation = ExtrapolationType.Linear)        # SOC → V
    soc_of_v = invert_ocv(v_of_soc; n_samples, extrapolation = ExtrapolationType.Linear)    # V → SOC
    return (; v_of_soc, soc_of_v)
end

# The absolute SOC gauge a reference OCV measurement implies: extrapolate the composite linearly
# to the voltage limits, call `V_min` 0 % SOC and `V_max` 100 % (datasheet: 4.1 V is 100 %), and
# read the anchor points for `rescale_composite_ocv` off it:
#
#     g = soc_gauge(meas_composite)
#     refs = (; v_low = 3.62, soc_low = g.soc_of_v(3.62), v_high = 4.05, soc_high = g.soc_of_v(4.05))
function soc_gauge(composite; V_min = 2.9, V_max = 4.1)
    (; v_of_soc, soc_of_v) = extrapolate_ocv(composite)
    soc_min = soc_of_v(V_min)
    soc_span = soc_of_v(V_max) - soc_min
    return (;
        soc_of_v = v -> (soc_of_v(v) - soc_min) / soc_span,
        v_of_soc = s -> v_of_soc(soc_min + s * soc_span),
    )
end

# Estimate the usable SOC window of the measured composite by linearly extrapolating the
# OCV to the full voltage window [V_min, V_max]. Returns the extrapolated interpolants and
# the SOC bounds — `soc_of_v` maps any voltage to its SOC on the extrapolated composite and
# is exactly the `ref_soc_of_v` reference that `fit_cells_to_reference` consumes to scale the
# reconstructed cell OCVs onto this full-window gauge.
function eval_soc_range(composite; V_min = 2.9, V_max = 4.1)
    (; v_of_soc, soc_of_v) = extrapolate_ocv(composite)
    soc = collect(composite.t)
    v = collect(composite.u)

    soc_at_Vmin = soc_of_v(V_min)
    soc_at_Vmax = soc_of_v(V_max)
    soc_full = soc_at_Vmax - soc_at_Vmin

    soc_data_min = first(soc)
    soc_data_max = last(soc)
    soc_data = soc_data_max - soc_data_min

    V_data_min = first(v)
    V_data_max = last(v)

    return (;
        v_of_soc, soc_of_v, soc_at_Vmin, soc_at_Vmax, soc_full,
        # coverage of the measured data within the extrapolated [V_min, V_max] gauge
        soc_data_min, soc_data_max, soc_data, V_data_min, V_data_max,
        soc_used_lo = 100 * (soc_data_min - soc_at_Vmin) / soc_full,
        soc_used_hi = 100 * (soc_data_max - soc_at_Vmin) / soc_full,
    )
end
