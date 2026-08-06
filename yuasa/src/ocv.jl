# Validation of the reconstructed cell OCV curves (RGP-ECM, this dataset) against the
# oscilloscope-measured OCV from the low-power cycling experiment (data/validation).
#
# The validated module carries the accurate current probe: it is rig module 7 in the
# yuasa-ocv test and P1M9 (p=1, m=9) here. Cell index c refers to the same physical
# cell in both datasets.
#
# This file is self-contained: measured-OCV builders (parse the rig parquet),
# composite-decomposition validation, and the validation plots.


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
    q = range(first(f.t), last(f.t); length = n_samples)
    v = f.(q)
    # Ensure strict monotonicity: remove duplicate/decreasing V values
    mask = [true; diff(v) .> 0]
    return LinearInterpolation(q[mask], v[mask]; extrapolation)
end

# Midline OCV by averaging the charge and discharge branches in the VOLTAGE frame:
# at each voltage, the mean of the two branches' charge, q̄(v) = (q_chg(v)+q_dch(v))/2.
# Averaging at fixed voltage (not fixed charge) keeps the high-SOC end that q-frame
# averaging clips — the q-frame average is restricted to the charge range BOTH
# branches share, which (because the two experiments are offset in charge) truncates
# the top of the curve ~20 mV below either branch. The unknown charge offset between
# the experiments only shifts the (irrelevant) q origin here; it does not distort the
# shape. Returns (; q, μ) with μ the voltage grid and q the averaged charge.
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


# === individual-cell OCV validation (RGP-OCV vs measured low-power OCV) ===
# `measured`/`reconstructed` are vectors of monotone v(q) curves `(; q, μ)` (Ah, V), one per
# cell in matching order (rig module 7 ≡ P1M9; cell index identical).
#
# Two separable claims, deliberately NOT sharing machinery:
#
# SHAPE — does the reconstruction give the right OCV curve? Each RGP curve is aligned to its
#   measured counterpart over THAT cell's own voltage overlap (never a window common to all
#   cells: the shared window is pure plateau, and dropping the steep knee — 12 % of the charge
#   but 8× the dV/dq — costs most of the capacity information). The alignment is
#   `q_rgp(v) ≈ a·q_meas(v) + b`, and only `b` is fitted: the two datasets are separate
#   experiments so their charge origins are unrelated, but their capacities are not. `a` is
#   FIXED at the ratio the SOH estimates imply, so the residual tests shape AND capacity
#   together. `ocv_rmse_free` refits `a` as well; the gap between the two localises a failure
#   (equal ⇒ capacity is consistent, free much lower ⇒ capacity is the problem).
#
# SOH/SOC — per-cell `Q_cell` and `s0`. The RGP side comes from the 324-cell composite fit the
#   paper actually reports, NOT a 12-cell refit: refitting a composite from these 12 curves
#   alone shrinks the estimated spread ~40 % (0.74 → 0.45 %) because the shared shape adapts to
#   them, and it validates an estimator no result uses. The measured side has no such option —
#   only 12 rig curves exist. Both are relabelled onto one declared anchor convention and only
#   DEVIATIONS are compared: absolute capacity is a convention (mean Q moves 63–79 Ah with the
#   anchors), not a measurement.
#
# NOTE the rig measures VOLTAGE ONLY. There is one module-level current sensor, so all 12 cells
# share a charge axis and the measured `Q_cell`/`s0` are INFERRED under the shared-shape
# assumption, not independently measured. So this validates the OCV curve that feeds the
# extraction, not the extraction itself. The assumption is checkable and holds: normalising each
# measured curve by its own charge between two fixed voltages (no decomposition, no gauge) puts
# the 12 cells within ~1.4 mV of each other.


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
    calc_ocv_shape_validation(measured, reconstructed, fcs, fds, Q_meas, Q_rgp) -> DataFrame

Per-cell OCV shape validation over each cell's own measured∩RGP voltage overlap. `fcs`/`fds`
are that cell's charge/discharge branch interpolants from `clean_ocv` (needed for the reference
floor). `Q_meas`/`Q_rgp` set the fixed alignment scale `a = Q_rgp/Q_meas`.

Columns: `v_lo`/`v_hi` (the cell's own window), `ocv_rmse` (mV, `a` fixed at the SOH ratio),
`ocv_rmse_free` (mV, `a` refitted), `floor` (mV, the reference's own ambiguity — see
`calc_reference_floor`), `ratio` = `ocv_rmse`/`floor` (the cross-cell-comparable number; the
bare mV is not, since the windows differ in width and steepness), and the two scales `a_soh`,
`a_fit` plus the fitted offset `b_soh` the figure needs to place the curves.
"""
function calc_ocv_shape_validation(measured, reconstructed, fcs, fds, Q_meas, Q_rgp; n_v = 200)
    qm = Measurements.value.(Q_meas)
    qr = Measurements.value.(Q_rgp)
    floors = calc_reference_floor(measured, reconstructed, fcs, fds; n_v)

    return map(eachindex(measured)) do c
        gm = _q_of_v(measured[c])
        gr = _q_of_v(reconstructed[c])
        v_lo = max(minimum(gm.t), minimum(gr.t)) + 1.0e-3
        v_hi = min(maximum(gm.t), maximum(gr.t)) - 1.0e-3
        v = collect(range(v_lo, v_hi; length = n_v))
        slope = _dv_dq(gr, v)

        a_soh = qr[c] / qm[c]
        b_soh = mean(gr.(v) .- a_soh .* gm.(v))          # only the charge origin is fitted
        a_fit, _ = hcat(gm.(v), ones(length(v))) \ gr.(v)
        b_fit = mean(gr.(v) .- a_fit .* gm.(v))

        rmse(a, b) = sqrt(mean(abs2, (gr.(v) .- (a .* gm.(v) .+ b)) .* slope .* 1000))
        r = rmse(a_soh, b_soh)
        (;
            cell = c, v_lo, v_hi,
            ocv_rmse = r, ocv_rmse_free = rmse(a_fit, b_fit),
            floor = floors[c], ratio = r / floors[c],
            a_soh, a_fit, b_soh,
        )
    end |> DataFrame
end

"""
    calc_ocv_curves(measured, reconstructed, df_shape, Q_meas, s0_meas) -> Vector

Plot-ready per-cell curves on the common SOC axis, plus the residual behind the `ocv_rmse`
summary. Reuses the alignment `df_shape` already fitted (`a_soh`, `b_soh`), so there is one
alignment, not two: the RGP curve is mapped into the measured charge frame and both are then
divided by that cell's own `(Q_meas, s0_meas)`.

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
        r = (gr.(v) .- (df_shape.a_soh[c] .* gm.(v) .+ df_shape.b_soh[c])) .* _dv_dq(gr, v) .* 1000
        (;
            soc_meas = tosoc(collect(measured[c].q)), v_meas = collect(measured[c].μ),
            # RGP mapped into the measured charge frame by this cell's alignment, then to SOC
            soc_rgp = tosoc((collect(reconstructed[c].q) .- df_shape.b_soh[c]) ./ df_shape.a_soh[c]),
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
    qm = Measurements.value.(Q_meas); qr = Measurements.value.(Q_rgp)
    sm = Measurements.value.(s0_meas); sr = Measurements.value.(s0_rgp)
    k = mean(qr) / mean(qm)
    return DataFrame(
        cell = collect(eachindex(qm)),
        Q_meas = qm, Q_rgp = qr, ΔQ = qr ./ k .- qm,
        Q_meas_σ = Measurements.uncertainty.(Q_meas), Q_rgp_σ = Measurements.uncertainty.(Q_rgp),
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


# === validation export (main.jl → validation.jl) ===
# validation.jl must compare the SOH the paper REPORTS, which comes from the 324-cell joint fit
# in main.jl. It cannot rederive that from 12 cells, so main.jl hands over the per-cell
# parameters and curves directly.

# Per-cell `(Q_cell, s0)` + reconstructed OCV curves for `ref_ids`, taken from the full-fleet
# `comp_fit`/`ocvs` indexed by `ids`, plus the anchor convention they are expressed in.
function build_validation_export(comp_fit, ocvs, ids, ref_ids, refs)
    pos = Dict(id => i for (i, id) in enumerate(ids))
    return Dict(
        "refs" => Dict(String(k) => v for (k, v) in pairs(refs)),
        "cells" => [
            let i = pos[id]
                Dict(
                    "id" => "$(id.p)_$(id.m)_$(id.c)",
                    "Q" => Measurements.value(comp_fit.Q_cell[i]),
                    "Q_sigma" => Measurements.uncertainty(comp_fit.Q_cell[i]),
                    "s0" => Measurements.value(comp_fit.s0[i]),
                    "s0_sigma" => Measurements.uncertainty(comp_fit.s0[i]),
                    "q" => collect(ocvs[i].q), "v" => collect(ocvs[i].μ),
                )
            end for id in ref_ids
        ],
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


# === validation plots ===

# Consumes the prepared tables + the raw curves. Left column shares the charge axis:
# (A) every cell's measured OCV against its reconstruction, (B) the residual behind that
# overlay. Right column: (C) SOH and (D) initial-SOC deviations, with the bars both estimates
# carry. The measured charge/discharge branches are deliberately NOT drawn — they are binned
# pseudo-OCV (0.1 Ah bins, extreme voltage per bin), so their scatter reads as data when it is
# mostly extraction noise. Their separation is the reference floor, reported per cell in
# `df_shape` instead.
function plot_cell_ocv_validation(df_shape, curves, df_soh)
    wong = Makie.wong_colors()
    c_meas = :gray55
    c_rgp = wong[1]

    fig = Figure(size = (700, 500))

    # (A) OCV overlay on the common SOC axis — each cell placed by its own (Q, s0), so the
    # curves collapse and the shape comparison is what the reader sees
    # x fixed to the full 0-100 % SOC range: the gap at each end is the coverage this validation
    # actually has (the experiments span ~4-90 %), which an auto-scaled axis would hide
    axA = Axis(
        fig[1, 1]; xlabel = "SOC / %", ylabel = "OCV / V",
        limits = (0, 100, nothing, nothing), xgridvisible = false, ygridvisible = false,
    )
    for cv in curves
        lines!(axA, cv.soc_meas, cv.v_meas; color = c_meas)
        lines!(axA, cv.soc_rgp, cv.v_rgp; color = c_rgp, linestyle = :dash)
    end
    axislegend(
        axA,
        [LineElement(color = c_meas), LineElement(color = c_rgp, linestyle = :dash)],
        ["Reference", "RGP-ECM"];
        position = :rb, framevisible = false,
    )
    axA.yticks = 3.4:0.2:4.0

    # (B) the residual behind panel A, same SOC axis. Deliberately unannotated: no single summary
    # survives here, because the per-cell mV RMSE is not comparable between cells (each is scored
    # over its own window). Median 6.6, mean 10.7, pooled 13.1 mV — the spread is cells 1-3's wider
    # windows reaching into the knee, not worse agreement (on a matched window they are the BEST
    # three, 4.6-4.8 mV). The comparable statistic is `ratio` in `df_shape` — RMSE over each cell's
    # own reference floor — which is 0.49-0.79 for all twelve, median and mean both 0.60.
    # Clipped: the low-SOC knee of cells 1-3
    # spikes to ~-70 mV (steep dV/dq there amplifies a small charge error). It is real and
    # included in the RMS, but plotting it would flatten the ±10 mV structure that matters.
    axB = Axis(
        fig[2, 1]; xlabel = "SOC / %", ylabel = "ΔOCV / mV",
        xgridvisible = false, ygridvisible = false, limits = (0, 100, -25, 15),
    )
    for cv in curves
        lines!(axB, cv.soc, cv.r; color = (c_rgp, 0.6))
    end
    hlines!(axB, [0]; color = :black, linestyle = :dot)
    linkxaxes!(axA, axB)

    for ax in (axA, axB)
        ax.xticks = 0:20:100
        ax.xminorticks = IntervalsBetween(2)
        ax.xminorticksvisible = true
        ax.yminorticks = IntervalsBetween(2)
        ax.yminorticksvisible = true
    end

    # (C) SOH as absolute capacity. Only 1:1 is drawn — the displacement of the cloud from it IS
    # the offset, which is instrument-bounded rather than a per-cell claim, so it needs no fitted
    # line of its own. Square axes with equal ranges, so the 1:1 line means what it looks like;
    # that is why the data occupies only part of the panel.
    lo = minimum(vcat(df_soh.Q_meas .- df_soh.Q_meas_σ, df_soh.Q_rgp .- df_soh.Q_rgp_σ)) - 0.4
    hi = maximum(vcat(df_soh.Q_meas .+ df_soh.Q_meas_σ, df_soh.Q_rgp .+ df_soh.Q_rgp_σ)) + 0.4
    axC = Axis(
        fig[1, 2]; xlabel = "Reference Q / Ah", ylabel = "RGP Q / Ah",
        aspect = 1, limits = (lo, hi, lo, hi), xgridvisible = false, ygridvisible = false,
    )
    for a in (:x, :y)   # major ticks are every 2 Ah → minor ticks every 1 Ah
        setproperty!(axC, Symbol(a, :minorticks), IntervalsBetween(2))
        setproperty!(axC, Symbol(a, :minorticksvisible), true)
    end
    ablines!(axC, 0, 1; color = :gray, linestyle = :dash)
    errorbars!(axC, df_soh.Q_meas, df_soh.Q_rgp, df_soh.Q_rgp_σ; color = (c_rgp, 0.35), whiskerwidth = 0)
    errorbars!(axC, df_soh.Q_meas, df_soh.Q_rgp, df_soh.Q_meas_σ; color = (c_rgp, 0.35), whiskerwidth = 0, direction = :x)
    scatter!(axC, df_soh.Q_meas, df_soh.Q_rgp; color = c_rgp, markersize = 11, strokewidth = 0.5, strokecolor = :white)

    # (D) initial SOC — deviations from each dataset's own module mean
    L = 1.35 * maximum(abs, vcat(df_soh.s0_meas .+ df_soh.s0_meas_σ, df_soh.s0_rgp .+ df_soh.s0_rgp_σ))
    axD = Axis(
        fig[2, 2]; xlabel = "Reference ΔSOC / %", ylabel = "RGP ΔSOC / %",
        aspect = 1, limits = (-L, L, -L, L), xgridvisible = false, ygridvisible = false,
    )
    for a in (:x, :y)   # majors pinned every 5 % so the minors land on whole percent
        setproperty!(axD, Symbol(a, :ticks), -5:5:5)
        setproperty!(axD, Symbol(a, :minorticks), IntervalsBetween(5))
        setproperty!(axD, Symbol(a, :minorticksvisible), true)
    end
    ablines!(axD, 0, 1; color = :gray, linestyle = :dash)
    errorbars!(axD, df_soh.s0_meas, df_soh.s0_rgp, df_soh.s0_rgp_σ; color = (c_rgp, 0.35), whiskerwidth = 0)
    errorbars!(axD, df_soh.s0_meas, df_soh.s0_rgp, df_soh.s0_meas_σ; color = (c_rgp, 0.35), whiskerwidth = 0, direction = :x)
    scatter!(axD, df_soh.s0_meas, df_soh.s0_rgp; color = c_rgp, markersize = 11, strokewidth = 0.5, strokecolor = :white)

    for ax in (axA, axB, axC, axD)
        hidespines!(ax, :t, :r)
    end
    for (ax, l) in ((axA, "A"), (axB, "B"), (axC, "C"), (axD, "D"))
        text!(ax, 0.02, 0.98; text = l, space = :relative, align = (:left, :top), font = :bold, fontsize = 20)
    end
    colsize!(fig.layout, 1, Relative(0.6))
    return fig
end


# === measured-OCV experiment diagnostics ===

# Current-sensor cross-check on the rig data: the BMS module current vs the oscilloscope
# (tek) reference probe, their integrated charge, and the running charge error between them.
function compare_current_sources(df; m = 7)
    df_bms = integrate_current(df; current_col = "m$(m)_current", negate = true)
    df_tek = integrate_current(df; current_col = "tek_m_cur_ref", negate = true, offset = find_tek_offset(df; m))

    fig = Figure(size = (900, 700))
    ax1 = Axis(fig[1, 1], ylabel = "Current / A", title = "BMS vs Oscilloscope Current (Module $(m))")
    ax2 = Axis(fig[2, 1], ylabel = "Charge / Ah")
    ax3 = Axis(fig[3, 1], ylabel = "Charge Error / Ah", xlabel = "Time")

    lines!(ax1, df_bms.timestamp_utc, df_bms.i, label = "BMS (m$(m)_current)")
    lines!(ax1, df_tek.timestamp_utc, df_tek.i, label = "Oscilloscope (tek)")

    lines!(ax2, df_bms.timestamp_utc, df_bms.q, label = "BMS")
    lines!(ax2, df_tek.timestamp_utc, df_tek.q, label = "Oscilloscope")

    df_err = innerjoin(
        select(df_bms, :timestamp_utc, :q => :q_bms),
        select(df_tek, :timestamp_utc, :q => :q_tek),
        on = :timestamp_utc
    )
    df_err[!, :Δq] = df_err.q_bms .- df_err.q_tek
    lines!(ax3, df_err.timestamp_utc, df_err.Δq)

    linkxaxes!(ax1, ax2, ax3)
    Legend(fig[4, 1], ax2, orientation = :horizontal)

    return fig
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

function plot_ocv_extrapolation(composite; V_min = 2.9, V_max = 4.1)
    (; v_of_soc, soc_of_v) = extrapolate_ocv(composite)
    soc = collect(composite.t)

    soc_at_Vmin = soc_of_v(V_min)
    soc_at_Vmax = soc_of_v(V_max)

    fig = Figure(size = (900, 500))
    ax = Axis(
        fig[1, 1], ylabel = "Voltage / V", xlabel = "SOC",
        title = "OCV with linear extrapolation"
    )

    soc_full = range(soc_at_Vmin, soc_at_Vmax; length = 500)
    lines!(ax, collect(soc_full), v_of_soc.(soc_full), color = :red, linewidth = 2, linestyle = :dash, label = "Extrapolated")

    soc_data = range(first(soc), last(soc); length = 500)
    scatterlines!(ax, collect(soc_data), composite.(soc_data), color = :black, linewidth = 2, label = "Measured")

    hlines!(ax, [V_min, V_max], color = :gray, linestyle = :dot)
    ylims!(ax, V_min, V_max)
    axislegend(ax; position = :rc)
    return fig
end

# Diagnostic: overlay the cleaned OCV (clean_ocv) on the raw (q, v) measurement for one
# cell/branch — shows how the per-bin extraction tracks the relaxed points within the
# noisy loaded data. `q` shares clean_ocv's frame (current-integrated, zeroed at min).
function plot_ocv_cleaning(df, id; dch::Bool, current_col = "tek", i_thresh = 0.5)
    col = current_col == "tek" ? "tek_m_cur_ref" : "m$(id.m)_current"
    offset = current_col == "tek" ? find_tek_offset(df; m = id.m) : 0.0
    df_i = integrate_current(df; current_col = col, negate = true, offset)
    df_i[!, :q] = df_i.q .- minimum(df_i.q)
    cell = @sprintf("%02d", id.c)
    raw = innerjoin(select(df, :timestamp_utc, "m$(id.m)_cell$cell" => :v), df_i, on = :timestamp_utc, makeunique = true)
    f = clean_ocv(df, id; dch, i_thresh, current_col)  # interp knots: f.t=q grid, f.u=v

    wong = Makie.wong_colors()
    fig = Figure(size = (800, 500))
    ax = Axis(
        fig[1, 1]; xlabel = "Charge / Ah", ylabel = "Voltage / V",
        title = "OCV cleaning — cell m$(id.m) c$(id.c) ($(dch ? "discharge" : "charge"))",
        xgridvisible = false, ygridvisible = false,
    )
    scatter!(ax, raw.q, raw.v; color = (:gray, 0.3), markersize = 4, label = "raw")
    lines!(ax, f.t, f.u; color = wong[2], linewidth = 2, label = "cleaned")
    axislegend(ax; position = :rb, framevisible = false)
    return fig
end
