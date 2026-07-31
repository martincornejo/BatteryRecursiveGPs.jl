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
using BatteryRecursiveGPs   # fit_composite_ocv, fit_model, gp_ocv, RCGPModel

using Distributed
nprocs() == 1 && addprocs(Sys.CPU_THREADS ÷ 2)
@everywhere using BatteryRecursiveGPs

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
function fit_cells_init(pool, ids, ϑ_init, cell_data, zt; n = 1)
    tasks = Dict(
        id => remotecall(
                fit_ocv_curve, pool,
                cell_data[id].u, cell_data[id].y,
                scale_θ(cell_data[id].u, cell_data[id].y, ϑ_init; n), zt
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
        ϑ, ℓ_ocv_grid, ℓ_r1_grid = Float64[], thresh_mV, n = 1
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
                    ); n
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

# Paper figure (supplementary): two-stage hyperparameter selection. Columns = cells |
# modules (module OCV ÷ n → per-cell scale); each column reads top-to-bottom as one story.
# A/B: dV/dSOC fans at the shared initial ℓ, flagged units (> thresh at init) highlighted;
# C/D: fans after per-unit adaptation; E/F: composite-OCV RMSE distribution, initial vs
# adapted (linear mV axis; off-scale outliers annotated as text instead of squeezed in).
# Selected-ℓ counts are tabulated separately (see selection_counts).
# `cells`/`modules` bundle (; curves_init, picks, comp_ref, ids, thresh_mV, n).
function plot_hyperparam_selection(; cells, modules)
    fig = Figure(size = (700, 640), figure_padding = 8)
    wong = Makie.wong_colors()
    c_flag = wong[6]  # vermillion: flagged at init (> threshold)
    c_bulk = (:gray, 0.35)
    c_init = :gray30; c_adap = wong[1]

    axs = Matrix{Axis}(undef, 3, 2)
    for (col, lvl) in enumerate((cells, modules))
        (; curves_init, picks, comp_ref, ids, thresh_mV, n) = lvl
        curves_final = Dict(id => picks[id].curve for id in ids)
        flags = [picks[id].rmse_init > thresh_mV for id in ids]

        # rows 1-2: dV/dSOC fans, initial vs adapted, composite in black
        for (row, curves_by_id) in ((1, curves_init), (2, curves_final))
            ax = Axis(fig[row, col]); axs[row, col] = ax
            curves = align_cells(curves_by_id, ids, comp_ref)
            for flagged in (false, true), k in eachindex(ids)  # bulk first, flagged on top
                flags[k] == flagged || continue
                sm, dv = dvdsoc(curves[k].soc, curves[k].μ ./ n)
                lines!(ax, sm, dv; color = flagged ? (c_flag, 0.8) : c_bulk, linewidth = flagged ? 1.0 : 0.8)
            end
            sc, vc = compcurve(comp_ref)
            sm, dv = dvdsoc(sc, vc ./ n)
            edge = (sm .> 0.01) .& (sm .< 0.995)  # steep-edge blowup of the composite derivative
            lines!(ax, sm[edge], dv[edge]; color = :black, linewidth = 2)
            ylims!(ax, -0.5, 3); xlims!(ax, 0, 1)
        end
        axs[2, col].xlabel = "SOC"

        # row 3: RMSE shift, initial vs adapted; off-scale outliers noted as text
        ax = Axis(fig[3, col]); axs[3, col] = ax
        xmax = 8.0
        rb = [picks[id].rmse_init for id in ids] ./ n
        ra = [picks[id].rmse_mV for id in ids] ./ n
        density!(ax, rb; color = (c_init, 0.2), strokecolor = c_init, strokewidth = 1.8, label = rich("initial ℓ", subscript("OCV")))
        density!(ax, ra; color = (c_adap, 0.2), strokecolor = c_adap, strokewidth = 1.8, label = rich("adapted ℓ", subscript("OCV")))
        # off-scale units: one paired note each, named, numbers in the curve colors
        unitname(id) = hasproperty(id, :c) ? "Cell P$(id.p)M$(id.m)C$(id.c)" : "Module P$(id.p)M$(id.m)"
        out = [(ids[k], rb[k], ra[k]) for k in eachindex(rb) if rb[k] > xmax || ra[k] > xmax]
        for (i, (id, b, a)) in enumerate(out)
            text!(
                ax, 0.97, 0.95 - 0.15 * (i - 1);
                text = rich("$(unitname(id)): ", rich(string(round(Int, b)), color = c_init), " → ",
                            rich(string(round(Int, a)), color = c_adap), " mV"),
                space = :relative, align = (:right, :top), fontsize = 10,
            )
        end
        vlines!(ax, [thresh_mV / n]; color = :black, linestyle = :dot, linewidth = 1.2)
        ax.xticks = 0:2:8
        xlims!(ax, 0, xmax); ylims!(ax, 0, nothing)
        ax.xlabel = "Composite-OCV RMSE / mV"
    end

    # fan-line legend (colors classify by the stage-1 fit; caption details)
    fanleg = [
        LineElement(color = (:gray, 0.7), linewidth = 1.2) => "≤ threshold (stage 1)",
        LineElement(color = (c_flag, 0.9), linewidth = 1.2) => "> threshold (stage 1)",
        LineElement(color = :black, linewidth = 2) => "composite",
    ]
    axislegend(axs[2, 1], first.(fanleg), last.(fanleg); position = :rt, framevisible = false, patchsize = (16, 10), rowgap = 0)

    Label(fig[0, 1], "Cells"; font = :bold, fontsize = 16, tellwidth = false)
    Label(fig[0, 2], "Modules (per cell)"; font = :bold, fontsize = 16, tellwidth = false)
    for col in 1:2
        axs[1, col].title = rich("Initial ℓ", subscript("OCV"), " (stage 1)")
        axs[2, col].title = rich("Adapted ℓ", subscript("OCV"), " (stage 2)")
        axs[1, col].titlefont = :regular; axs[2, col].titlefont = :regular
    end
    axs[1, 1].ylabel = "dV/dSOC / V"; axs[2, 1].ylabel = "dV/dSOC / V"; axs[3, 1].ylabel = "Density"
    foreach(ax -> hidexdecorations!(ax; ticks = false, grid = false), axs[1, :])
    for row in 1:3
        linkyaxes!(axs[row, 1], axs[row, 2])
        hideydecorations!(axs[row, 2]; ticks = false, grid = false)
    end
    axislegend(axs[3, 2]; position = :rt, framevisible = false, patchsize = (14, 10), rowgap = 0)
    for ax in vec(axs)
        ax.xgridvisible = false; ax.ygridvisible = false
    end
    for (tag, pos) in zip(("A", "B", "C", "D", "E", "F"), ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)))
        Label(fig[pos..., TopLeft()], tag; fontsize = 18, font = :bold, padding = (0, 25, 2, 0), halign = :left)
    end
    rowgap!(fig.layout, 6)
    return fig
end

# marginal counts of the selected nominal length scales — the paper's selection table
function selection_counts(picks, picks_mod)
    cnt(f, ps) = countmap([f(p) for p in values(ps)])
    cols = [
        "cells_ocv" => cnt(p -> p.ℓ_ocv, picks), "modules_ocv" => cnt(p -> p.ℓ_ocv, picks_mod),
        "cells_r1" => cnt(p -> p.ℓ_r1, picks), "modules_r1" => cnt(p -> p.ℓ_r1, picks_mod),
    ]
    ls = sort(unique(reduce(vcat, collect.(keys.(last.(cols))))))
    return DataFrame("ℓ" => ls, (name => [get(c, l, 0) for l in ls] for (name, c) in cols)...)
end

# Data-scaled GP hyperparameters per unit in physical units: length scales in Ah (ℓ from
# scale_θ is in z-scored charge → × zt.q scale), σ as the PRIOR STD per cell in mV (OCV) /
# mΩ (R1) — θ.σ multiplies the kernel, i.e. it is a variance, so the std is √θσ (rcgp.jl).
# The selected nominal values are carried along for the figure's color coding.
function calc_scaled_hyperparams(picks, unit_data, ids, ϑ, zt; n = 1)
    return map(ids) do id
        p = picks[id]
        d = unit_data[id]
        θ = scale_θ(
            d.u, d.y,
            merge(ϑ, (; ocv = merge(ϑ.ocv, (; ℓ = p.ℓ_ocv)), r1 = merge(ϑ.r1, (; ℓ = p.ℓ_r1)))); n
        )
        (;
            id,
            ℓ_ocv = θ.ocv.ℓ * zt.q.scale[1], ℓ_r1 = θ.r1.ℓ * zt.q.scale[1],
            σ_ocv = sqrt(θ.ocv.σ) * zt.σ.scale[1] / n * 1000, σ_r1 = sqrt(θ.r1.σ) * zt.r.scale[1] / n * 1000,
            nom_ocv = p.ℓ_ocv, nom_r1 = p.ℓ_r1,
        )
    end
end

# Paper figure (supplementary): all four GP hyperparameters of every unit — histograms in
# physical units (2 rows cells/modules × 4 equal columns ℓ_ocv/ℓ_r1/σ_ocv/σ_r1, ~40 bins
# each so bar widths match; ℓ_ocv log-x). Bars are stack-colored by the selected NOMINAL
# value (gray = 0.5 init, lipari steps for escalations) — the legend speaks the text's
# vocabulary, and the σ columns come out all-gray since σ is never adapted. Units beyond
# any cell capacity (ℓ_ocv > 100 Ah) are named; their 1-count bars would be invisible.
function plot_hyperparam_scales(scaled_cells, scaled_mods)
    nom_esc = [0.3, 0.6, 0.85, 1.0, 1.5, 15.0]  # escalation-grid values (union of both ℓ grids)
    colors = Dict(v => get(cgrad(:lipari), t) for (v, t) in zip(nom_esc, range(0.8, 0.12, length = length(nom_esc))))
    colors[0.5] = RGBAf(0.65, 0.65, 0.65, 1)    # init: neutral gray
    nom_all = [0.5; nom_esc]                    # init drawn first (bottom of the stacks)

    function stacked_hist!(ax, groups, edges; logx = false)
        centers = logx ? sqrt.(edges[1:(end - 1)] .* edges[2:end]) : (edges[1:(end - 1)] .+ edges[2:end]) ./ 2
        base = zeros(length(centers))
        for (vals, color) in groups
            h = StatsBase.fit(Histogram, vals, edges).weights
            barplot!(ax, centers, Float64.(h); offset = base, width = diff(edges), color, gap = 0, strokewidth = 0.4, strokecolor = :white)
            base .+= h
        end
        return
    end

    # (value, nominal-for-color, xscale, xticks, xlims, bin edges, log-spaced?, label)
    specs = [
        (x -> x.ℓ_ocv, x -> x.nom_ocv, log10, ([10, 20, 50, 100, 200], ["10", "20", "50", "100", "200"]),
            (9, 220), 10 .^ (log10(9):0.035:log10(220)), true, rich("ℓ", subscript("OCV"), " / Ah")),
        (x -> x.ℓ_r1, x -> x.nom_r1, identity, (0:20:60, string.(0:20:60)),
            (5, 68), 5:1.5:68, false, rich("ℓ", subscript("R1"), " / Ah")),
        (x -> x.σ_ocv, x -> 0.5, identity, ([180, 230, 280], ["180", "230", "280"]),
            (175, 295), 178:2.9:294, false, rich("σ", subscript("OCV"), " / mV")),
        (x -> x.σ_r1, x -> 0.5, identity, (7:1:10, string.(7:1:10)),
            (6.1, 10.3), 6.2:0.1:10.2, false, rich("σ", subscript("R1"), " / mΩ")),
    ]

    fig = Figure(size = (700, 360), figure_padding = 8)
    axs = Matrix{Axis}(undef, 2, 4)
    for (col, (val, nom, xscale, xticks, lims, bins, logx, xlab)) in enumerate(specs)
        # Modules*: σ shown per cell (module value ÷ n) for a scale comparable to the cells
        for (row, (a, name)) in enumerate(((scaled_cells, "Cells"), (scaled_mods, "Modules*")))
            ax = Axis(fig[row, col]; xscale, xticks)
            axs[row, col] = ax
            groups = [(Float64[val(x) for x in a if nom(x) == v], colors[v]) for v in nom_all]
            stacked_hist!(ax, groups, collect(bins); logx)
            xlims!(ax, lims...); ylims!(ax, 0, nothing)
            col == 1 && (ax.ylabel = "$name / count")
            row == 2 ? (ax.xlabel = xlab) : hidexdecorations!(ax; ticks = false, grid = false)
            ax.xgridvisible = false; ax.ygridvisible = false
        end
        linkxaxes!(axs[:, col]...)
    end
    for row in 1:2
        linkyaxes!(axs[row, :]...)
        foreach(ax -> hideydecorations!(ax; ticks = false, grid = false), axs[row, 2:4])
    end
    for x in scaled_cells
        x.ℓ_ocv > 100 || continue
        lines!(axs[1, 1], [x.ℓ_ocv, x.ℓ_ocv], [45, 6]; color = :black, linewidth = 0.8)
        text!(axs[1, 1], x.ℓ_ocv, 50; text = "P$(x.id.p)M$(x.id.m)C$(x.id.c)", align = (:right, :bottom), fontsize = 9)
    end
    fmt(v) = v == floor(v) ? string(Int(v)) : string(v)
    order = sort(nom_all)
    Legend(
        fig[3, 1:4],
        [PolyElement(color = colors[v]) for v in order],
        [(v == 0.5 ? "0.5 (init)" : fmt(v)) for v in order],
        "nominal ℓ / σ:";
        framevisible = false, patchsize = (12, 12), orientation = :horizontal,
        titleposition = :left, titlefont = :regular, colgap = 12,
    )
    rowgap!(fig.layout, 8); colgap!(fig.layout, 10)
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

end
# ---- Export ----
hyperparams = build_hyperparam_export(picks, ϑ_init, id -> "$(id.p)_$(id.m)_$(id.c)")
write(out_json, JSON.json(hyperparams, 2))

# ================================ modules ================================

begin
    out_json_mod = joinpath(datadir, "module_hyperparams.json")
    ϑ_init_mod = (; ocv = (; σ = 0.5, ℓ = 0.5), r1 = (; σ = 0.5, ℓ = 0.5))
    module_ids = [(; p, m) for p in 1:3, m in 1:9] |> vec |> sort
    thresh_mV_mod = 50.0

    zt_mod = fit_zscore(12)
    module_data = Dict(id => module_dataset(data, ti, id.p, id.m; zt = zt_mod) for id in module_ids)

    # ---- Stage 1 ----
    curves_init_mod = fit_cells_init(pool, module_ids, ϑ_init_mod, module_data, zt_mod; n = 12)

    # ---- Reference composite: coarse → outlier-filter → refit ----
    comp_coarse_mod = fit_composite_ocv(values(curves_init_mod); uq = false, n_v_pair = 20)
    coarse_score_mod = make_scorer(comp_coarse_mod)
    inliers_mod = [c for c in values(curves_init_mod) if coarse_score_mod(c) <= thresh_mV_mod]
    comp_ref_mod = fit_composite_ocv(inliers_mod; uq = false, n_v_pair = 20)

    # ---- Stage 2 ----
    picks_mod = escalate_cells(
        curves_init_mod, comp_ref_mod, pool, module_data, zt_mod;
        ϑ = ϑ_init_mod, ℓ_ocv_grid, ℓ_r1_grid, thresh_mV = thresh_mV_mod, n = 12,
    )

    # ---- Final composite ----
    final_keep_mod = [picks_mod[id].curve for id in module_ids if picks_mod[id].rmse_mV <= thresh_mV_mod]
    comp_final_mod = fit_composite_ocv(final_keep_mod; uq = false, n_v_pair = 20)

end
# ---- Export ----
hyperparams_mod = build_hyperparam_export(picks_mod, ϑ_init_mod, id -> "$(id.p)_$(id.m)")
write(out_json_mod, JSON.json(hyperparams_mod, 2))

# ---- Figure + selection table (in memory only; save manually if needed) ----
fig_hyperparam_selection = plot_hyperparam_selection(;
    cells = (; curves_init, picks, comp_ref, ids, thresh_mV, n = 1),
    modules = (;
        curves_init = curves_init_mod, picks = picks_mod, comp_ref = comp_ref_mod,
        ids = module_ids, thresh_mV = thresh_mV_mod, n = 12,
    ),
)
df_selection = selection_counts(picks, picks_mod)
fig_hyperparam_scales = plot_hyperparam_scales(
    calc_scaled_hyperparams(picks, cell_data, ids, ϑ_init, zt),
    calc_scaled_hyperparams(picks_mod, module_data, module_ids, ϑ_init_mod, zt_mod; n = 12),
)
