# === validation plots ===

# The validation figure, pooled over the comparable modules (one `validate_module` result per
# module, `labels` = their names): (A) measured-vs-model OCV overlay, (B) the residual behind it
# (clipped at the low-SOC knee, where the steep dV/dq amplifies a small charge error), (C)
# absolute capacity per cell, (D) initial-SOC deviation from each module's mean (absolute initial
# SOC is not comparable between the two experiments). Wong palette + black by module.
function plot_validation(vals, labels)
    cols = vcat(Makie.wong_colors(), [RGBAf(0, 0, 0, 1), RGBAf(0.6, 0.6, 0.6, 1)])
    c_meas = :gray55

    fig = Figure(size = (630, 450))

    axA = Axis(
        fig[1, 1]; xlabel = "SOC / %", ylabel = "OCV / V",
        limits = (0, 100, nothing, nothing), xgridvisible = false, ygridvisible = false,
    )
    for v in vals, cv in v.curves
        lines!(axA, cv.soc_meas, cv.v_meas; color = (c_meas, 0.8))
    end
    for (j, v) in enumerate(vals), cv in v.curves
        lines!(axA, cv.soc_rgp, cv.v_rgp; color = (cols[j], 0.8), linestyle = :dash)
    end
    axislegend(
        axA,
        [LineElement(color = c_meas), LineElement(color = :black, linestyle = :dash)],
        ["Reference", "Model"];
        position = :rb, framevisible = false,
    )
    axA.yticks = 3.4:0.2:4.0

    axB = Axis(
        fig[2, 1]; xlabel = "SOC / %", ylabel = "ΔOCV / mV",
        xgridvisible = false, ygridvisible = false, limits = (0, 100, -25, 15),
    )
    for (j, v) in enumerate(vals), cv in v.curves
        lines!(axB, cv.soc, cv.r; color = (cols[j], 0.5))
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

    axC = Axis(
        fig[1, 2]; xlabel = "Reference Q / Ah", ylabel = "Model Q / Ah",
        aspect = 1, limits = (69, 81, 69, 81), xticks = 70:2:80, yticks = 70:2:80,
        xgridvisible = false, ygridvisible = false,
    )
    s0_all = vcat([vcat(v.df_soh.s0_meas, v.df_soh.s0_rgp) for v in vals]...)
    L = 1.15 * maximum(abs, s0_all)
    axD = Axis(
        fig[2, 2]; xlabel = "Reference ΔSOC / %", ylabel = "Model ΔSOC / %",
        aspect = 1, limits = (-L, L, -L, L), xgridvisible = false, ygridvisible = false,
    )
    for ax in (axC, axD)
        ablines!(ax, 0, 1; color = :black, linestyle = :dash)
    end
    for (j, v) in enumerate(vals)
        d = v.df_soh
        errorbars!(axC, d.Q_meas, d.Q_rgp, d.Q_rgp_σ; color = (cols[j], 0.3), whiskerwidth = 0)
        errorbars!(axC, d.Q_meas, d.Q_rgp, d.Q_meas_σ; color = (cols[j], 0.3), whiskerwidth = 0, direction = :x)
        scatter!(axC, d.Q_meas, d.Q_rgp; color = cols[j], markersize = 8, strokewidth = 0.1, strokecolor = :white, label = labels[j])
        errorbars!(axD, d.s0_meas, d.s0_rgp, d.s0_rgp_σ; color = (cols[j], 0.3), whiskerwidth = 0)
        scatter!(axD, d.s0_meas, d.s0_rgp; color = cols[j], markersize = 8, strokewidth = 0.1, strokecolor = :white)
    end
    for a in (:x, :y)
        setproperty!(axC, Symbol(a, :minorticks), IntervalsBetween(2))
        setproperty!(axC, Symbol(a, :minorticksvisible), true)
        setproperty!(axD, Symbol(a, :ticks), -5:5:5)
        setproperty!(axD, Symbol(a, :minorticks), IntervalsBetween(5))
        setproperty!(axD, Symbol(a, :minorticksvisible), true)
    end

    # grid cells exclude the axis decorations, so a cell-centred legend sits half the left
    # protrusion right of the visual centre; shrink its box on the right by that amount
    leg = Legend(
        fig[3, 1:2], axC; framevisible = false, orientation = :horizontal, nbanks = 1,
        patchsize = (12, 12), colgap = 14, tellheight = true,
    )
    colsize!(fig.layout, 1, Relative(0.58))
    leg.margin = (0, axA.layoutobservables.protrusions[].left, 0, 0)

    for ax in (axA, axB, axC, axD)
        hidespines!(ax, :t, :r)
    end
    for (ax, l) in ((axA, "A"), (axB, "B"), (axC, "C"), (axD, "D"))
        text!(ax, 0.02, 0.98; text = l, space = :relative, align = (:left, :top), font = :bold, fontsize = 20)
    end
    return fig
end

# Supplementary: distribution of the per-cell OCV RMSE, stacked by module (1 mV bins). The tail
# above ~7 mV is the knee-bearing cells (their windows reach the steep low knee); plateau cells
# sit at 3–5 mV.
function plot_validation_rmse(vals, labels; bins = 0:1:16)
    cols = vcat(Makie.wong_colors(), [RGBAf(0, 0, 0, 1), RGBAf(0.6, 0.6, 0.6, 1)])
    nb = length(bins) - 1
    counts = zeros(Int, nb, length(vals))
    for (j, v) in enumerate(vals), r in v.df_shape.ocv_rmse
        counts[clamp(floor(Int, r) + 1, 1, nb), j] += 1
    end

    fig = Figure(size = (500, 350))
    ax = Axis(
        fig[1, 1]; xlabel = "OCV RMSE / mV", ylabel = "cells",
        limits = (first(bins), last(bins), 0, 50), xticks = first(bins):4:last(bins),
        xgridvisible = false, ygridvisible = false,
    )
    xs = repeat(collect(bins[1:(end - 1)]) .+ 0.5, length(vals))
    grp = repeat(1:length(vals); inner = nb)
    barplot!(ax, xs, vec(counts); stack = grp, color = cols[grp], gap = 0.05, strokewidth = 0.5, strokecolor = :white)
    Legend(
        fig[1, 1], [PolyElement(color = cols[j]) for j in eachindex(vals)], labels;
        tellwidth = false, tellheight = false, halign = :right, valign = :top, framevisible = false,
        patchsize = (10, 10), rowgap = 1, nbanks = 2,
    )
    hidespines!(ax, :t, :r)
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
