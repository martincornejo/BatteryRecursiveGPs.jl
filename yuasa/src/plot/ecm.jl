# terminal-voltage fit over measurement, with the innovation below at ≈ 1/3 the height
function plot_sim!(gl, model::AbstractBatteryModel, sol; Ts = 1.0)
    kf = model.kf
    zt = kf.p.zt
    (; idx, u, yt, yμ, yΣ) = sol

    μ = StatsBase.reconstruct(zt.v, first.(yμ))
    σ = StatsBase.reconstruct(zt.σ, sqrt.(first.(yΣ)))
    t = (0:(length(u) - 1)) * Ts / 3600 |> collect

    c_meas = :gray30        # measured reference (neutral, dark enough to balance the model)
    c_model = Makie.wong_colors()[2]   # cell-model prediction / error (orange), matching "cells" in the accuracy figure

    ax_v = Axis(gl[1, 1]; ylabel = "Voltage / V")
    ax_e = Axis(gl[2, 1]; ylabel = "Error / mV", xlabel = "Time / h")

    v = StatsBase.reconstruct(zt.v, first.(yt))
    band!(ax_v, t[idx], μ - 2σ, μ + 2σ; color = (c_model, 0.3), label = "Model")
    lines!(ax_v, t[idx], μ; color = c_model, label = "Model")
    lines!(ax_v, t[idx], v; color = c_meas, label = "Measured")

    e = v - μ
    rmse = sqrt(sum(abs2, e) / length(e)) * 1.0e3
    elim = maximum(abs, vcat(e - 2σ, e + 2σ)) * 1.0e3
    hlines!(ax_e, [0]; color = (:black, 0.4), linewidth = 1.2)
    band!(ax_e, t[idx], (e - 2σ) * 1.0e3, (e + 2σ) * 1.0e3; color = (c_model, 0.3))
    lines!(ax_e, t[idx], e * 1.0e3; color = c_model)
    text!(
        ax_e, 0.99, 0.96; text = "RMSE = $(round(rmse; digits = 1)) mV",
        space = :relative, align = (:right, :top), fontsize = 11
    )

    if 0 < sol.tt < length(sol.u)
        vlines!(ax_v, t[sol.tt]; color = :red)
        vlines!(ax_e, t[sol.tt]; color = :red)
    end

    xlims!(ax_v, t[begin], t[end])
    xlims!(ax_e, t[begin], t[end])
    ylims!(ax_e, -1.1elim, 1.1elim)
    for a in (ax_v, ax_e)
        a.xgridvisible = false
        a.ygridvisible = false
        a.yminorticks = IntervalsBetween(2)
        a.yminorticksvisible = true
        a.xminorticks = IntervalsBetween(2)
        a.xminorticksvisible = true
    end
    linkxaxes!(ax_v, ax_e)
    hidexdecorations!(ax_v; grid = false, ticks = false)
    axislegend(ax_v; merge = true, framevisible = false, position = :rb, padding = (4, 4, 2, 2))
    rowsize!(gl, 1, Auto(2))   # error panel ≈ 1/3 of the example height
    rowsize!(gl, 2, Auto(1))
    rowgap!(gl, 6)
    return (ax_v, ax_e)
end

"""
    plot_sim(model, sol; Ts = 1.0) -> Figure

Terminal voltage of one unit, measured against modelled, with the innovation in a panel below.
Pair with [`eval_model`](@ref)'s solution for the open-loop fit rather than the one-step-ahead
prediction. `plot_sim!(gl, model, sol)` draws the same pair into an existing layout.
"""
function plot_sim(model::AbstractBatteryModel, sol; Ts = 1.0)
    fig = Figure(size = (700, 300))
    plot_sim!(GridLayout(fig[1, 1]), model, sol; Ts)
    return fig
end

"""
    plot_q_estimation(q_ref, sol, model) -> Figure

Filtered charge estimate against the reference `q_ref` and the Coulomb-counting baseline, with
their errors in a panel below.
"""
function plot_q_estimation(q_ref, sol, model::AbstractBatteryModel)
    kf = model.kf
    (; zt) = kf.p

    t = sol.idx  # observation times only
    q = StatsBase.reconstruct(zt.q, [u.q for u in sol.ut])
    qμ = StatsBase.reconstruct(zt.q, sol.qμ)
    qσ = StatsBase.reconstruct(zt.q, sqrt.(sol.qσ))
    q_ref_t = q_ref[t]

    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]

    lines!(ax[1], t / 3600, q_ref_t; color = :black, label = "Reference")
    lines!(ax[1], t / 3600, q; color = :red, linestyle = :dash, label = "Coulomb counting")
    lines!(ax[1], t / 3600, qμ; color = Cycled(2), label = "Estimated Q")
    band!(ax[1], t / 3600, qμ - 2qσ, qμ + 2qσ; color = Cycled(2), alpha = 0.5, label = "Estimated Q")

    lines!(ax[2], t / 3600, qμ - q_ref_t; color = Cycled(2), label = "Estimated Q")
    band!(ax[2], t / 3600, (qμ - q_ref_t) - 2qσ, (qμ - q_ref_t) + 2qσ; color = Cycled(2), alpha = 0.5, label = "Estimated Q")
    lines!(ax[2], t / 3600, q - q_ref_t; color = :red, linestyle = :dash, label = "Coulomb counting")

    xlims!(ax[1], t[begin] / 3600, t[end] / 3600)
    ax[1].ylabel = "Charge / Ah"
    ax[2].ylabel = "Error / Ah"
    ax[2].xlabel = "Time / h"
    Legend(fig[3, 1], ax[1]; merge = true, orientation = :horizontal)
    return fig
end

# ECM curves for one cell/module onto a 2-row axis (OCV top, R0+R1(q) bottom).
# Helper for plot_ecms_comparison below.
function plot_ecm!(ax, model::YuasaModel, sol; color = nothing)
    kw = isnothing(color) ? (;) : (; color)
    kf = model.kf
    zt = kf.p.zt

    q̂ = collect(range(extrema(sol.qμ)...; step = 0.01))
    q = StatsBase.reconstruct(zt.q, q̂)

    # OCV
    ocv = predict_gp(kf, q̂, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
    ocvσ = StatsBase.reconstruct(zt.σ, sqrt.(diag(ocv.Σ)))
    lines!(ax[1], q, ocvμ; kw...)
    band!(ax[1], q, ocvμ + 2ocvσ, ocvμ - 2ocvσ; alpha = 0.8, kw...)

    # R0 scalar from the final state estimate
    xc = ComponentVector(sol.x_end, kf.p.xid)
    Σ = ComponentMatrix(sol.R_end, kf.p.Σid)
    r0μ = StatsBase.reconstruct(zt.r, [abs(xc.r0.r)]) * 1.0e3 |> first
    r0σ = StatsBase.reconstruct(zt.r, [sqrt(Σ[:r0, :r0][:r, :r])]) * 1.0e3 |> first

    # R1 GP curve overlaid on the R0 panel
    r1 = predict_gp(kf, q̂, :r1)
    r1μ = StatsBase.reconstruct(zt.r, r1.μ) * 1.0e3
    r1σ = StatsBase.reconstruct(zt.r, sqrt.(diag(r1.Σ))) * 1.0e3
    rμ = r0μ .+ r1μ
    rσ = r0σ .+ r1σ
    lines!(ax[2], q, rμ; kw...)
    band!(ax[2], q, rμ + 2rσ, rμ - 2rσ; alpha = 0.8, kw...)
    return nothing
end

"""
    plot_ecms_comparison(cell_models, cell_sols, module_models, module_sols;
                         n_cell = 1, n_mod = 12, tags = true, colors = MODULE_COLORS) -> Figure

Identified OCV and RΣ(q) curves at both levels side by side, cells on the left and modules on
the right. Every unit in the dictionaries is drawn, coloured by module ID.

`n_cell` and `n_mod` are the number of series cells each level represents; they set the axis
limits and ticks, so each column reads in its own units rather than a shared one.
"""
function plot_ecms_comparison(
        cell_models, cell_sols, module_models, module_sols;
        n_cell = 1, n_mod = 12, tags = true,
        colors = MODULE_COLORS,
    )
    fig = Figure(size = (700, 450))
    gl1 = GridLayout(fig[1, 1])
    gl2 = GridLayout(fig[1, 2])

    scenarios = [
        (; title = "Cell level", gl = gl1, models = cell_models, sols = cell_sols, n = n_cell, tags = ("A", "C")),
        (; title = "Module level", gl = gl2, models = module_models, sols = module_sols, n = n_mod, tags = ("B", "D")),
    ]

    for args in scenarios
        (; title, gl, models, sols, n) = args
        ax = [Axis(gl[i, 1]) for i in 1:2]
        if tags
            for i in 1:2  # tags inside the axes: no layout space needed
                text!(ax[i], 0.02, 0.98; text = args.tags[i], space = :relative, align = (:left, :top), font = :bold, fontsize = 20)
            end
        end
        ax[1].title = title
        ax[1].ylabel = "OCV / V"
        ax[2].ylabel = rich("R", subscript("Σ"), " / mΩ")
        ax[2].xlabel = "Cumulative charge / Ah"
        ax[1].xgridvisible = false
        ax[1].ygridvisible = false
        ax[2].xgridvisible = false
        ax[2].ygridvisible = false
        hidexdecorations!(ax[1], ticks = false)

        for (id, model) in models
            plot_ecm!(ax, model, sols[id]; color = colors[id.m])
        end

        ylims!(ax[1], n * 3.35, n * 4.15)
        ylims!(ax[2], n * 0.0, n * 16)
        # 0.25 V/cell steps so module ticks (×12) land on integers (42, 45, 48)
        ax[1].yticks = (n * 3.5):(n * 0.25):(n * 4.0)
        ax[2].yticks = (0):(n * 5):(n * 15)
        linkxaxes!(ax...)
    end

    mod_elems = [LineElement(color = colors[m], linewidth = 3) for m in 1:9]
    Legend(
        fig[2, 1:2], mod_elems, ["M$m" for m in 1:9], "Module ID";
        orientation = :horizontal, titleposition = :left, framevisible = false
    )

    return fig
end

"""
    plot_ecm_parameters(df, df_mod = nothing; v_ref = 3.9, Δr = 0.06, Δτ = 5.0,
                        T = 5:1:40, T0 = 25) -> Figure

Fleet distribution of the fitted ECM parameters from a [`calc_ecm_parameters`](@ref) table:
R1, R0, RΣ, τ and the Arrhenius correction over `T`. Pass `df_mod` to overlay the module fits
on the cells.

Shown as densities rather than counts, since the two levels have 324 and 27 units. Resistances
are compared at `v_ref`, and the module table must come from a builder run with `n = 12` for
the levels to share a scale.
"""
function plot_ecm_parameters(df, df_mod = nothing; v_ref = 3.9, Δr = 0.06, Δτ = 5.0, T = 5:1:40, T0 = 25)
    wong = Makie.wong_colors()
    c_cell = wong[2]
    c_mod = wong[1]
    c_cell_med = wong[6]

    fig = Figure(size = (700, 400))

    pct(v) = (x = v * 100; isapprox(x, round(x); atol = 1.0e-9) ? string(round(Int, x)) : string(round(x; digits = 1)))
    function overlay!(ax, vc, vm; bins, ymax, ystep)
        ax.ytickformat = vs -> pct.(vs)
        ylims!(ax, 0, ymax)
        ax.yticks = 0:ystep:ymax
        ax.yminorticks = IntervalsBetween(2)
        ax.yminorticksvisible = true
        hist!(ax, vc; bins, normalization = :probability, color = (c_cell, 0.85), strokewidth = 0.5, strokecolor = :white)
        isnothing(vm) && return
        hist!(ax, vm; bins, normalization = :probability, color = (c_mod, 0.55), strokewidth = 0.5, strokecolor = :white)
        return
    end

    shared(v, Δ) = (floor(minimum(v) / Δ) * Δ):Δ:(ceil(maximum(v) / Δ) * Δ)
    r = isnothing(df_mod) ? vcat(df.R0, df.R1) : vcat(df.R0, df.R1, df_mod.R0, df_mod.R1)
    edges = shared(r, Δr)
    τ_edges = shared(isnothing(df_mod) ? df.τ : vcat(df.τ, df_mod.τ), Δτ)

    axA = Axis(fig[1, 1]; xlabel = rich("R", subscript("1"), " ($(v_ref) V) / mΩ"), ylabel = "Share / %")
    overlay!(axA, df.R1, isnothing(df_mod) ? nothing : df_mod.R1; bins = edges, ymax = 0.25, ystep = 0.1)

    # name the R1 outlier cells in place — their single-cell bars are barely visible at this scale
    fence = quantile(df.R1, 0.75) + 1.5 * (quantile(df.R1, 0.75) - quantile(df.R1, 0.25))
    for r in eachrow(df[df.R1 .> fence, :])
        text!(
            axA, r.R1, 1 / nrow(df) + 0.012; text = "P$(r.id.p)M$(r.id.m)C$(r.id.c)",
            rotation = π / 2, align = (:left, :center), fontsize = 9, color = c_cell
        )
    end

    axC = Axis(fig[2, 1]; xlabel = rich("R", subscript("0"), " / mΩ"), ylabel = "Share / %")
    overlay!(axC, df.R0, isnothing(df_mod) ? nothing : df_mod.R0; bins = edges, ymax = 0.5, ystep = 0.2)
    linkxaxes!(axA, axC)
    # the same cells flagged in (A); their R0 values coincide, so one label carries both
    out = df[df.R1 .> fence, :]
    if !isempty(out)
        text!(
            axC, mean(out.R0), nrow(out) / nrow(df) + 0.024;
            text = join(("P$(r.id.p)M$(r.id.m)C$(r.id.c)" for r in eachrow(out)), "\n"),
            rotation = π / 2, align = (:left, :center), fontsize = 9, color = c_cell, offset = (6, 0)
        )
    end

    axB = Axis(fig[1, 2]; xlabel = "τ / s", ylabel = "Share / %")
    overlay!(axB, df.τ, isnothing(df_mod) ? nothing : df_mod.τ; bins = τ_edges, ymax = 0.25, ystep = 0.1)

    # (D) the Arrhenius factor k(T) itself, unity at the reference temperature T0 = 25 °C
    axD = Axis(fig[2, 2]; xlabel = "Temperature / °C", ylabel = "Arrhenius factor k(T)")
    # k(T) = exp(k·(1/T − 1/T0)) is a closed form of the fitted `k`, not a separate quantity, so it
    # is rendered here rather than prepared upstream
    Tg = collect(T)
    Δ = 1 ./ (Tg .+ 273.15) .- 1 / (T0 + 273.15)
    kfac(ks) = [exp.(k .* Δ) for k in ks]
    fac = kfac(df.k)
    fac_mod = isnothing(df_mod) ? nothing : kfac(df_mod.k)

    for f in fac
        lines!(axD, Tg, f; color = (c_cell, 0.15), linewidth = 0.8)
    end
    if !isnothing(fac_mod)
        for f in fac_mod
            lines!(axD, Tg, f; color = (c_mod, 0.35), linewidth = 0.8)
        end
    end

    lines!(axD, Tg, [median(getindex.(fac, i)) for i in eachindex(Tg)]; color = c_cell_med, linewidth = 2)
    if !isnothing(fac_mod)
        lines!(
            axD, Tg, [median(getindex.(fac_mod, i)) for i in eachindex(Tg)];
            color = :black, linewidth = 2
        )
    end
    hlines!(axD, [1.0]; color = :black, linestyle = :dot, linewidth = 1)

    ylims!(axD, nothing, 2.5)
    xlims!(axD, extrema(Tg)...)
    axD.yticks = 1:1:2
    axD.yminorticks = IntervalsBetween(2)
    axD.yminorticksvisible = true

    entries = isnothing(fac_mod) ?
        ([[LineElement(color = (c_cell, 0.6)), LineElement(color = c_cell_med, linewidth = 2)]], ["Cell-level median"]) :
        (
            [
                [LineElement(color = (c_cell, 0.6)), LineElement(color = c_cell_med, linewidth = 2)],
                [LineElement(color = (c_mod, 0.7)), LineElement(color = :black, linewidth = 2)],
            ],
            ["Cell-level median", "Module-level median"],
        )
    axislegend(axD, entries...; position = :rt, framevisible = false, patchsize = (18, 10), rowgap = 0)

    for ax in (axA, axB, axC, axD)
        ax.xgridvisible = false
        ax.ygridvisible = false
        hidespines!(ax, :t, :r)
        ax.xminorticks = IntervalsBetween(2)
        ax.xminorticksvisible = true
    end
    if !isnothing(df_mod)
        axislegend(
            axA, [PolyElement(color = (c_cell, 0.85)), PolyElement(color = (c_mod, 0.55))],
            ["Cell-level", "Module-level"]; position = :rt, framevisible = false, patchsize = (16, 10), rowgap = 0,
        )
    end
    for (tag, ax) in (("A", axA), ("B", axB), ("C", axC), ("D", axD))
        text!(ax, 0.02, 0.98; text = tag, space = :relative, align = (:left, :top), font = :bold, fontsize = 20)
    end
    return fig
end
