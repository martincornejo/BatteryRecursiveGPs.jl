using DataFrames
using CSV
using Dates
using Intervals
using StatsBase
using StaticArrays
using DataInterpolations
using Statistics
using Measurements
using Printf
using JSON

# master: load packages BEFORE addprocs so they precompile (else workers hit a world-age race)
using YuasaAnalysis        # cell_dataset, fit_zscore, scale_θ, module_dataset
using BatteryDigitalTwin   # fit_composite_ocv, fit_model, gp_ocv, RCGPModel

using Distributed
nprocs() == 1 && addprocs(Sys.CPU_THREADS ÷ 2)
@everywhere using BatteryDigitalTwin

using CairoMakie   # master-only: figures are built in memory, not persisted

# Slim worker payload: KF fit at a fully-specified θ, return only the GP-OCV curve.
# θ is built on the master via scale_θ before remotecall.
@everywhere function fit_ocv_curve(u, y, θ, zt)
    res = fit_model(RCGPModel, u, y, θ, zt)
    ocv = gp_ocv(res.model, res.sol)
    return (; q = collect(ocv.q), μ = collect(ocv.μ))
end

# ---- Selection logic: clean composite reference + keep-best per-cell escalation ----

# closure scoring an OCV curve against a fixed reference composite → RMSE in mV
function make_scorer(comp_ref)
    o_s = sortperm(comp_ref.soc_grid)
    o_v = sortperm(comp_ref.v_grid)
    v_of_soc = LinearInterpolation(comp_ref.v_grid[o_s], comp_ref.soc_grid[o_s]; extrapolation = ExtrapolationType.Constant)
    soc_of_v = LinearInterpolation(comp_ref.soc_grid[o_v], comp_ref.v_grid[o_v]; extrapolation = ExtrapolationType.Constant)
    v_range = extrema(comp_ref.v_grid)
    soc_lo, soc_hi = first(comp_ref.soc_grid[o_s]), last(comp_ref.soc_grid[o_s])
    function score(ocv)
        aligned = fit_cells_to_reference([(; ocv.q, ocv.μ)], soc_of_v, v_range)
        Q = Measurements.value(aligned.Q_cell[1])
        s0 = Measurements.value(aligned.s0[1])
        soc = ocv.q ./ Q .+ s0
        mask = (soc .>= soc_lo) .& (soc .<= soc_hi)
        return sqrt(mean(abs2, (ocv.μ[mask] .- v_of_soc.(soc[mask])) .* 1000))
    end
    return score
end

# Stage 1: fit every cell at ϑ_init. Any worker failure crashes — init is a known-good
# config, so a failure here is a real bug.
function fit_cells_init(pool, ids, ϑ_init, cell_data, zt)
    tasks = Dict(
        id => remotecall(
                fit_ocv_curve, pool,
                cell_data[id].u, cell_data[id].y,
                scale_θ(cell_data[id].u, cell_data[id].y, ϑ_init), zt
            )
            for id in ids
    )
    return Dict(id => fetch(t) for (id, t) in tasks)
end

# Stage 2 + pick: per-cell argmin RMSE over `(ℓ_ocv_init, ℓ_ocv_grid...)`. Starts from
# each cell's init curve; misfit cells (rmse > thresh_mV against `comp_ref`) additionally
# fit each grid ℓ in parallel and upgrade the pick when one beats the current best.
# Individual escalation-ℓ failures are tolerated (logged + skipped).
# Returns Dict{id, (; ℓ_ocv, rmse_mV, rmse_init, curve, escalated)} for every cell in `ids`.
function escalate_cells(
        curves_init, comp_ref, pool, cell_data, zt;
        ϑ, ℓ_ocv_grid, ℓ_r1_grid = Float64[], thresh_mV
    )
    score = make_scorer(comp_ref)

    # initial pick = init curve for every cell
    picks = Dict(
        id => begin
                r = score(curve)
                (; ℓ_ocv = ϑ.ocv.ℓ, ℓ_r1 = ϑ.r1.ℓ, rmse_mV = r, rmse_init = r, curve, escalated = false)
            end for (id, curve) in curves_init
    )

    # misfit cells: joint (ocv.ℓ, r1.ℓ) grid, keep argmin vs current pick
    misfits = [id for (id, p) in picks if p.rmse_mV > thresh_mV]
    ℓ_ocv_all = [ϑ.ocv.ℓ; ℓ_ocv_grid]
    ℓ_r1_all = [ϑ.r1.ℓ;  ℓ_r1_grid]
    challengers = [
        (lo, lr) for lo in ℓ_ocv_all, lr in ℓ_r1_all
            if !(lo == ϑ.ocv.ℓ && lr == ϑ.r1.ℓ)
    ]
    tasks = Dict(
        (id, ℓ_ocv, ℓ_r1) =>
            remotecall(
                fit_ocv_curve, pool, cell_data[id].u, cell_data[id].y,
                scale_θ(
                    cell_data[id].u, cell_data[id].y,
                    merge(
                        ϑ, (;
                            ocv = merge(ϑ.ocv, (; ℓ = ℓ_ocv)),
                            r1 = merge(ϑ.r1, (; ℓ = ℓ_r1)),
                        )
                    )
                ), zt
            )
            for id in misfits, (ℓ_ocv, ℓ_r1) in challengers
    )
    for id in misfits
        current = picks[id]
        for (ℓ_ocv, ℓ_r1) in challengers
            try
                curve = fetch(tasks[(id, ℓ_ocv, ℓ_r1)])
                rmse = score(curve)
                if rmse < current.rmse_mV
                    current = (;
                        ℓ_ocv, ℓ_r1, rmse_mV = rmse,
                        current.rmse_init, curve, escalated = true,
                    )
                end
            catch e
                @warn "escalation fit failed: id=$id ℓ_ocv=$ℓ_ocv ℓ_r1=$ℓ_r1" exception = e
            end
        end
        picks[id] = current
    end

    return picks
end

# cell/module id → nominal ℓ/σ stored in JSON; scale_θ reconstructs the full θ from these
function build_hyperparam_export(picks, ϑ, id_key)
    return Dict(
        id_key(id) => Dict(
                "ocv_ell" => p.ℓ_ocv, "ocv_sigma" => ϑ.ocv.σ,
                "r1_ell" => p.ℓ_r1, "r1_sigma" => ϑ.r1.σ,
            )
            for (id, p) in picks
    )
end

# ---- Figures (built in memory from top-level script bindings; not persisted) ----

compcurve(comp) = (o = sortperm(comp.soc_grid); (collect(comp.soc_grid)[o], collect(comp.v_grid)[o]))
function dvdsoc(soc, v)
    o = sortperm(soc); s = soc[o]; vv = v[o]
    return ((s[1:(end - 1)] .+ s[2:end]) ./ 2, diff(vv) ./ diff(s))
end

# align each cell's (q, μ) to a composite's SOC axis → vector of (; soc, μ) in `ids` order
function align_cells(curves_by_id, ids, comp)
    o_v = sortperm(comp.v_grid)
    soc_of_v = LinearInterpolation(comp.soc_grid[o_v], comp.v_grid[o_v]; extrapolation = ExtrapolationType.Constant)
    ocvs = [curves_by_id[id] for id in ids]
    al = fit_cells_to_reference([(; o.q, o.μ) for o in ocvs], soc_of_v, extrema(comp.v_grid))
    return [(; soc = ocvs[k].q ./ Measurements.value(al.Q_cell[k]) .+ Measurements.value(al.s0[k]), μ = ocvs[k].μ) for k in eachindex(ocvs)]
end

# 2×2: OCV (top) and dV/dSOC (bottom), before (stage 1) vs after (stage 1+2)
function plot_adaptation(; curves_init, picks, comp_ref, comp_final, ids, ℓ_ocv_init, thresh_mV, n = 1)
    cellcolor(r) = r > thresh_mV ? (:crimson, 0.8) : (:steelblue, 0.8)
    curves_final = Dict(id => picks[id].curve for id in ids)
    before = align_cells(curves_init, ids, comp_ref)
    after = align_cells(curves_final, ids, comp_final)
    rb = [picks[id].rmse_init for id in ids]
    ra = [picks[id].rmse_mV   for id in ids]
    fig = Figure(size = (1100, 920))
    for (col, (cells, rmv, comp, ttl)) in enumerate([(before, rb, comp_ref, "before (init)"), (after, ra, comp_final, "after (adapted)")])
        axo = Axis(fig[1, col]; title = "OCV — $ttl", xlabel = "SOC", ylabel = "V")
        for k in eachindex(cells)
            lines!(axo, cells[k].soc, cells[k].μ; color = cellcolor(rmv[k]))
        end
        sc, vc = compcurve(comp); lines!(axo, sc, vc; color = :black, linewidth = 2); xlims!(axo, 0, 1); ylims!(axo, n * 3.4, n * 4.15)
        axd = Axis(fig[2, col]; title = "dV/dSOC — $ttl", xlabel = "SOC", ylabel = "dV/dSOC")
        for k in eachindex(cells)
            sm, dv = dvdsoc(cells[k].soc, cells[k].μ); lines!(axd, sm, dv; color = cellcolor(rmv[k]))
        end
        sc, vc = compcurve(comp); sm, dv = dvdsoc(sc, vc); lines!(axd, sm, dv; color = :black, linewidth = 2); xlims!(axd, 0, 1); ylims!(axd, n * (-0.5), n * 3)
    end
    Label(fig[0, :], @sprintf("Stage-2 adaptation (init %.2g, %.1f mV trigger) — blue ≤%.1f mV, red >%.1f mV, black = composite", ℓ_ocv_init, thresh_mV, thresh_mV, thresh_mV), fontsize = 13)
    return fig
end

# one panel: per-cell RMSE distribution before vs after stage 2 (median dashed)
function plot_rmse_shift(; picks, ids, thresh_mV)
    rb = [picks[id].rmse_init for id in ids]
    ra = [picks[id].rmse_mV   for id in ids]
    clip_mV = 2 * thresh_mV
    clip(v) = filter(<=(clip_mV), v)
    fig = Figure(size = (660, 470))
    ax = Axis(fig[1, 1]; title = "Per-cell composite-OCV RMSE: before vs after stage 2", xlabel = "RMSE (mV)", ylabel = "density")
    density!(ax, clip(rb); color = (:steelblue, 0.25), strokecolor = :steelblue, strokewidth = 2, label = "before (init)")
    density!(ax, clip(ra); color = (:crimson, 0.25), strokecolor = :crimson, strokewidth = 2, label = "after (adapted)")
    vlines!(ax, [median(rb)]; color = :steelblue, linestyle = :dash)
    vlines!(ax, [median(ra)]; color = :crimson, linestyle = :dash)
    vlines!(ax, [thresh_mV]; color = :gray, linestyle = :dot, label = "threshold")
    xlims!(ax, 0, clip_mV); axislegend(ax)
    return fig
end

# 2 panels: distribution of the final absolute (z-q) length scales the GP actually uses
function plot_hyperparam_hist(; picks, cell_data, ϑ)
    ocv_ℓ = Float64[]; r1_ℓ = Float64[]
    for (id, p) in picks
        cd = cell_data[id]
        θ = scale_θ(cd.u, cd.y, merge(ϑ, (; ocv = merge(ϑ.ocv, (; ℓ = p.ℓ_ocv)))))
        push!(ocv_ℓ, θ.ocv.ℓ); push!(r1_ℓ, θ.r1.ℓ)
    end
    fig = Figure(size = (950, 380))
    ax1 = Axis(fig[1, 1]; title = "OCV length scale  ℓ_ocv (z-q units)", xlabel = "ℓ", ylabel = "cells")
    hist!(ax1, ocv_ℓ; bins = 30, color = (:steelblue, 0.85))
    ax2 = Axis(fig[1, 2]; title = "R1 length scale  ℓ_r1 (z-q units)", xlabel = "ℓ", ylabel = "cells")
    hist!(ax2, r1_ℓ; bins = 30, color = (:darkorange, 0.85))
    return fig
end

function load_data(datadir)
    files = Dict(
        :cell_voltage => datadir * "cell_voltages.csv",
        :module_voltage => datadir * "module_voltage.csv",
        :module_current => datadir * "module_current_average.csv",
        :battery_temperature => datadir * "battery_temperature.csv",
    )
    dfmt = dateformat"y-m-d H:M:S+00:00"
    data = Dict(id => CSV.File(f; dateformat = dfmt) |> DataFrame for (id, f) in files)
    ti = Interval(DateTime("2025-12-10T14:00:20"), DateTime("2025-12-11T02:30:20"))
    return data, ti
end

# ================================ script: config + run ================================
# Two-stage distributed selection from a generic init. Stage 1: fit every cell at ϑ_init
# and build a clean composite reference from the cells ≤thresh_mV. Stage 2: re-fit the
# misfit cells over ℓ_ocv_grid and keep, per cell, the ocv.ℓ minimising the composite-OCV
# residual. Only the per-cell hyperparam JSON is persisted — curves, composites and figures
# stay in memory at top-level for REPL inspection.

# ---- Config ----
datadir = "data/data-yuasa-cycles-2/"
out_json = joinpath(datadir, "cell_hyperparams.json")

begin
    ϑ_init = (;
        ocv = (; σ = 0.5, ℓ = 0.5),
        r1 = (; σ = 0.5, ℓ = 0.5),
    )
    thresh_mV = 4.0
    ℓ_ocv_grid = [0.6, 0.85, 1.0, 1.5, 3.0, 7.0, 15.0]  # escalation grid (challengers to the init)
    ℓ_r1_grid = [0.3, 1.0]                              # r1 length-scale challengers

    ids = [(; p, m, c) for p in 1:3, m in 1:9, c in 1:12] |> vec |> sort

    # ---- Setup ----
    data, ti = load_data(datadir)
    zt = fit_zscore()
    cell_data = Dict(id => cell_dataset(data, ti, id.p, id.m, id.c; zt) for id in ids)
    pool = WorkerPool(workers())

    # ---- Stage 1: init fit for every cell ----
    curves_init = fit_cells_init(pool, ids, ϑ_init, cell_data, zt)

    # ---- Reference composite: coarse → outlier-filter → refit ----
    comp_coarse = fit_composite_ocv(values(curves_init); uq = false, n_v_pair = 20)
    coarse_score = make_scorer(comp_coarse)
    inliers = [c for c in values(curves_init) if coarse_score(c) <= thresh_mV]
    comp_ref = fit_composite_ocv(inliers; uq = false, n_v_pair = 20)

    # ---- Stage 2: per-cell pick (init for all; escalation grid for misfits) ----
    picks = escalate_cells(
        curves_init, comp_ref, pool, cell_data, zt;
        ϑ = ϑ_init, ℓ_ocv_grid, ℓ_r1_grid, thresh_mV,
    )

    # ---- Final composite from cells passing the threshold after adaptation ----
    final_keep = [picks[id].curve for id in ids if picks[id].rmse_mV <= thresh_mV]
    comp_final = fit_composite_ocv(final_keep; uq = false, n_v_pair = 20)

    # ---- Export ----
    hyperparams = build_hyperparam_export(picks, ϑ_init, id -> "$(id.p)_$(id.m)_$(id.c)")
    write(out_json, JSON.json(hyperparams, 2))
    @info @sprintf("wrote %s (%d cells, %d escalated)", out_json, length(picks), count(p -> p.escalated, values(picks))); flush(stdout)
end

# ---- Figures (in memory only; save manually if needed) ----
fig_adaptation = plot_adaptation(; curves_init, picks, comp_ref, comp_final, ids, ℓ_ocv_init = ϑ_init.ocv.ℓ, thresh_mV)
fig_rmse_shift = plot_rmse_shift(; picks, ids, thresh_mV)
fig_hyperparams = plot_hyperparam_hist(; picks, cell_data, ϑ = ϑ_init)

# ================================ modules ================================

begin
    out_json_mod = joinpath(datadir, "module_hyperparams.json")
    ϑ_init_mod = (; ocv = (; σ = 0.5, ℓ = 0.5), r1 = (; σ = 0.5, ℓ = 0.5))
    module_ids = [(; p, m) for p in 1:3, m in 1:9] |> vec |> sort
    thresh_mV_mod = 50.0

    zt_mod = fit_zscore(12)
    module_data = Dict(id => module_dataset(data, ti, id.p, id.m; zt = zt_mod) for id in module_ids)

    # ---- Stage 1 ----
    curves_init_mod = fit_cells_init(pool, module_ids, ϑ_init_mod, module_data, zt_mod)

    # ---- Reference composite: coarse → outlier-filter → refit ----
    comp_coarse_mod = fit_composite_ocv(values(curves_init_mod); uq = false, n_v_pair = 20)
    coarse_score_mod = make_scorer(comp_coarse_mod)
    inliers_mod = [c for c in values(curves_init_mod) if coarse_score_mod(c) <= thresh_mV_mod]
    comp_ref_mod = fit_composite_ocv(inliers_mod; uq = false, n_v_pair = 20)

    # ---- Stage 2 ----
    picks_mod = escalate_cells(
        curves_init_mod, comp_ref_mod, pool, module_data, zt_mod;
        ϑ = ϑ_init_mod, ℓ_ocv_grid, ℓ_r1_grid, thresh_mV = thresh_mV_mod,
    )

    # ---- Final composite ----
    final_keep_mod = [picks_mod[id].curve for id in module_ids if picks_mod[id].rmse_mV <= thresh_mV_mod]
    comp_final_mod = fit_composite_ocv(final_keep_mod; uq = false, n_v_pair = 20)

    # ---- Export ----
    hyperparams_mod = build_hyperparam_export(picks_mod, ϑ_init_mod, id -> "$(id.p)_$(id.m)")
    write(out_json_mod, JSON.json(hyperparams_mod, 2))
    @info @sprintf("wrote %s (%d modules, %d escalated)", out_json_mod, length(picks_mod), count(p -> p.escalated, values(picks_mod))); flush(stdout)
end

# ---- Figures ----
fig_adaptation_mod = plot_adaptation(; curves_init = curves_init_mod, picks = picks_mod, comp_ref = comp_ref_mod, comp_final = comp_final_mod, ids = module_ids, ℓ_ocv_init = ϑ_init_mod.ocv.ℓ, thresh_mV = thresh_mV_mod, n = 12)
fig_rmse_shift_mod = plot_rmse_shift(; picks = picks_mod, ids = module_ids, thresh_mV = thresh_mV_mod)
fig_hyperparams_mod = plot_hyperparam_hist(; picks = picks_mod, cell_data = module_data, ϑ = ϑ_init_mod)
