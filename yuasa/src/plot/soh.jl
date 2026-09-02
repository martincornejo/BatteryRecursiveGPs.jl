"""
    plot_soh_heatmap(comp_fit) -> Figure

Fitted capacity of all 324 cells as a cell × module heatmap, on the lipari scale in Ah.
`plot_soh_heatmap!(layout, comp_fit)` draws it into an existing layout.
"""
function plot_soh_heatmap(comp_fit)
    fig = Figure(size = (300, 400))
    gl = GridLayout(fig[1, 1])
    plot_soh_heatmap!(gl, comp_fit)
    return fig
end

function plot_soh_heatmap!(layout, comp_fit)
    ax = Axis(layout[1, 1], xlabel = "Cell ID", ylabel = "Module ID")
    ax.xticks = [3, 6, 9, 12]
    ax.xminorticks = 1:12
    ax.xminorticksvisible = true

    ax.ytickformat = values -> module_label.(values)
    ax.yticks = [1, 6, 10, 15, 19, 24]
    ax.yminorticks = 1:27
    ax.yminorticksvisible = true

    colormap = :lipari
    colorrange = (60, 82)
    lowclip = :black
    soh = reshape(Measurements.value.(comp_fit.Q_cell), 12, 27)
    heatmap!(ax, soh; colorrange, colormap, lowclip)
    Colorbar(layout[1, 2]; limits = colorrange, colormap, lowclip, label = "Cell Capacity / Ah")
    return nothing
end

"""
    plot_composite_ocv(comp_fit, cells; xaxis = :soc) -> Figure

Every cell's reconstructed OCV placed on the composite's gauge, coloured by capacity, with the
consensus composite in black over them and dV/dSOC below. `xaxis = :ah` puts the x-axis in
capacity instead of SOC. `plot_composite_ocv!(layout, …)` draws it into an existing layout.
"""
function plot_composite_ocv(comp_fit, cells; xaxis = :soc)
    fig = Figure(size = (400, 400))
    gl = GridLayout(fig[1, 1])
    plot_composite_ocv!(gl, comp_fit, cells; xaxis)
    return fig
end

function plot_composite_ocv!(layout, comp_fit, cells; xaxis = :soc, vertical = true)
    (; soc_grid, v_grid, Q_cell, s0, Q_full) = comp_fit
    xscale = xaxis == :ah ? Q_full : 100.0   # SOC in %, as elsewhere
    xlabel = xaxis == :ah ? "Capacity / Ah" : "SOC / %"
    dlabel = xaxis == :ah ? "dV/dQ / (mV/Ah)" : "dV/dSOC / (mV/%)"

    if vertical
        ax1 = Axis(layout[1, 1]; ylabel = "Voltage / V")
        ax2 = Axis(layout[2, 1]; ylabel = dlabel, xlabel)
        hidexdecorations!(ax1; grid = false, ticks = false)
        linkxaxes!(ax1, ax2)
        rowgap!(layout, 1, 5)
    else
        ax1 = Axis(layout[1, 1]; ylabel = "Voltage / V", xlabel)
        ax2 = Axis(layout[1, 2]; ylabel = dlabel, xlabel)
    end

    for ax in (ax1, ax2)
        xaxis == :soc && (ax.xticks = 0:25:100)
        ax.xminorticks = IntervalsBetween(5)   # every 5 % SOC
        ax.xminorticksvisible = true
        ax.yminorticks = IntervalsBetween(2)
        ax.yminorticksvisible = true
    end
    # sparse majors, the intermediate values demoted to minor ticks
    ax1.yticks = 3.5:0.2:4.1
    ax2.yticks = 0:20:60

    Q_cell_μ = Measurements.value.(Q_cell)
    s0_μ = Measurements.value.(s0)
    colormap = :lipari
    colorrange = (60, 82)
    for i in eachindex(cells)
        q = cells[i].q
        v = cells[i].μ
        x = (q ./ Q_cell_μ[i] .+ s0_μ[i]) .* xscale
        lines!(ax1, x, v; color = Q_cell_μ[i], colorrange, colormap, alpha = 0.6, label = "Cell OCV")
        lines!(ax2, x[2:end], diff(v) ./ diff(x) .* 1000; color = Q_cell_μ[i], colorrange, colormap, alpha = 0.4)
    end

    order = sortperm(soc_grid)
    itp = PCHIPInterpolation(v_grid[order], soc_grid[order])
    soc_dense = range(soc_grid[order[1]], soc_grid[order[end]]; length = 500)
    dv_dx_raw = [DataInterpolations.derivative(itp, s) / xscale * 1000 for s in soc_dense]
    hw = 15
    n = length(dv_dx_raw)
    dv_dx = [mean(dv_dx_raw[max(1, i - hw):min(n, i + hw)]) for i in 1:n]

    x_comp = soc_grid .* xscale
    lines!(ax1, x_comp, v_grid; color = :black, linewidth = 2, label = "Composite OCV")
    lines!(ax2, collect(soc_dense) .* xscale, dv_dx; color = :black, linewidth = 2)
    ylims!(ax1, 3.4, 4.1)
    ylims!(ax2, 0, 60)
    xaxis == :soc && xlims!(ax2, 0, 100)   # linked, so ax1 follows
    ax1.xgridvisible = false
    ax1.ygridvisible = false
    ax2.xgridvisible = false
    ax2.ygridvisible = false

    axislegend(ax1; position = :rb, merge = true, framevisible = false)
    return
end

"""
    plot_cell_soh_hist(comp_fit; Q_nom = 100, size = (420, 380)) -> Figure

Histogram of cell SOH across the fleet, as capacity relative to `Q_nom` in %.
"""
function plot_cell_soh_hist(comp_fit; Q_nom = 100, size = (420, 380))
    soh = Measurements.value.(comp_fit.Q_cell) ./ Q_nom .* 100

    fig = Figure(; size)
    ax = Axis(fig[1, 1], xlabel = "Cell SOH / %", ylabel = "Cell count")

    bins = 35:1:85
    hist!(ax, soh; bins, color = :gray, strokewidth = 1)
    xlims!(ax, first(bins), last(bins))
    ylims!(ax, 0, 100)
    ax.yticks = 0:20:100

    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.rightspinevisible = false
    ax.topspinevisible = false
    ax.ytrimspine = (false, true)
    return fig
end

"""
    plot_cell_soh(comp_fit, cells; xaxis = :soc) -> Figure

The composite-OCV fit as a pair: (A) [`plot_composite_ocv`](@ref) beside (B) the capacity
[`plot_soh_heatmap`](@ref), sharing one capacity colour scale.
"""
function plot_cell_soh(comp_fit, cells; xaxis = :soc)
    fig = Figure(size = (700, 400))
    gl1 = GridLayout(fig[1, 1])
    gl2 = GridLayout(fig[1, 2])
    plot_composite_ocv!(gl1, comp_fit, cells; xaxis, vertical = true)
    plot_soh_heatmap!(gl2, comp_fit)
    colsize!(fig.layout, 2, Relative(0.4))
    Label(fig[1, 1, TopLeft()], "A"; fontsize = 20, font = :bold, padding = (0, 0, 5, 0))
    Label(fig[1, 2, TopLeft()], "B"; fontsize = 20, font = :bold, padding = (0, 0, 5, 0))
    return fig
end

"""
    plot_module_soh(df_soh; whiskers = true) -> Figure

Per-module SOH from the cell fits against the module model's own estimate, one pair per
module, from a [`calc_module_soh_summary`](@ref) table. `whiskers` draws the fit uncertainties.
`plot_module_soh!(layout, df_soh)` draws it into an existing layout.
"""
function plot_module_soh(df_soh; whiskers = true)
    fig = Figure(size = (700, 360))
    gl = GridLayout(fig[1, 1])
    plot_module_soh!(gl, df_soh; whiskers)
    return fig
end

function plot_module_soh!(
        layout, df_soh;
        whiskers = true, legend = true,
        # (module-based, cell-based)
        colors = Makie.wong_colors()[[2, 1]],
    )
    ax = Axis(layout[1, 1])

    soh_μ = Measurements.value.(df_soh.soh) * 100
    soh_σ = Measurements.uncertainty.(df_soh.soh) * 100

    m_soh_μ = Measurements.value.(df_soh.soh_module) * 100
    m_soh_σ = Measurements.uncertainty.(df_soh.soh_module) * 100

    ids = 1:27

    # dumbbell: gray connector emphasizes the module-vs-cell gap
    linesegments!(ax, repeat(ids; inner = 2), collect(Iterators.flatten(zip(m_soh_μ, soh_μ))); color = (:gray, 0.6), linewidth = 2)
    if whiskers
        errorbars!(ax, ids, m_soh_μ, 2 .* m_soh_σ; color = (colors[1], 0.6), whiskerwidth = 6)
        errorbars!(ax, ids, soh_μ, 2 .* soh_σ; color = (colors[2], 0.6), whiskerwidth = 6)
    end
    scatter!(ax, ids, m_soh_μ; color = colors[1], markersize = 10, label = "Module-based")
    scatter!(ax, ids, soh_μ; color = colors[2], markersize = 10, label = "Cell-based")

    vlines!(ax, [9.5, 18.5]; color = (:black, 0.3), linewidth = 1)

    ylims!(ax, 35, 90)
    xlims!(ax, 0.3, 27.7)
    ax.yticks = 40:10:100
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.rightspinevisible = false
    ax.topspinevisible = false
    ax.ylabel = "Module SOH / %"
    ax.xlabel = "Module ID"
    module_id_xticks!(ax)

    if legend
        Legend(layout[2, 1], ax; orientation = :horizontal, framevisible = false)
    end
    return ax
end

"""
    plot_module_inhomogeneity(df_soh; whiskers = true) -> Figure

Capacity each module cannot use, split into the irreversible part from cell-to-cell capacity
spread and the part a balancing cycle would recover, as stacked bars per module. Consumes a
[`calc_module_soh_summary`](@ref) table. `plot_module_inhomogeneity!(layout, df_soh)` draws it
into an existing layout.
"""
function plot_module_inhomogeneity(df_soh; whiskers = true)
    fig = Figure(size = (700, 360))
    gl = GridLayout(fig[1, 1])
    plot_module_inhomogeneity!(gl, df_soh; whiskers)
    return fig
end

function plot_module_inhomogeneity!(
        layout, df_soh;
        whiskers = true, legend = true,
        # (irreversible, reversible)
        bar_colors = Makie.wong_colors()[[3, 4]],
    )
    ax = Axis(layout[1, 1])

    total = (df_soh.loss_soh .+ df_soh.loss_soc) * 100
    irrev_μ = Measurements.value.(df_soh.loss_soh) * 100
    rev_μ = Measurements.value.(df_soh.loss_soc) * 100
    total_μ = Measurements.value.(total)
    total_σ = Measurements.uncertainty.(total)

    ids = 1:27

    barplot!(ax, ids, irrev_μ; width = 0.6, color = bar_colors[1], label = "Irreversible")
    barplot!(ax, ids, rev_μ; offset = irrev_μ, width = 0.6, color = bar_colors[2], label = "Reversible")
    if whiskers
        errorbars!(ax, ids, total_μ, 2 .* total_σ; color = (:black, 0.5), linewidth = 1.2, whiskerwidth = 5)
    end

    vlines!(ax, [9.5, 18.5]; color = (:black, 0.3), linewidth = 1)

    ylims!(ax, 0, 35)
    xlims!(ax, 0.3, 27.7)
    ax.yticks = 0:10:30
    ax.yminorticks = 0:5:35
    ax.yminorticksvisible = true
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.rightspinevisible = false
    ax.topspinevisible = false
    ax.ylabel = "SOH loss / %"
    ax.xlabel = "Module ID"
    module_id_xticks!(ax)

    if legend
        Legend(layout[2, 1], ax; orientation = :horizontal, framevisible = false)
    end
    return ax
end

"""
    plot_module_summary(df_soh; whiskers = true, soh_colors = …, bar_colors = …) -> Figure

[`plot_module_soh`](@ref) stacked over [`plot_module_inhomogeneity`](@ref) on a shared module
axis: the SOH each module reaches, and the capacity its cell spread costs.
"""
function plot_module_summary(
        df_soh; whiskers = true,
        soh_colors = Makie.wong_colors()[[1, 2]],
        bar_colors = Makie.wong_colors()[[4, 3]],
    )
    fig = Figure(size = (700, 350))
    gl = GridLayout(fig[1, 1])
    gl1 = GridLayout(gl[1, 1])
    gl2 = GridLayout(gl[2, 1])
    ax1 = plot_module_soh!(gl1, df_soh; whiskers, legend = false, colors = soh_colors)
    ax2 = plot_module_inhomogeneity!(gl2, df_soh; whiskers, legend = false, bar_colors)

    hidexdecorations!(ax1, ticks = false, minorticks = false)
    linkxaxes!(ax1, ax2)
    rowsize!(gl, 1, Auto(1.4))
    rowgap!(gl, 1, 8)

    axislegend(ax1; position = :cb, framevisible = false)
    axislegend(ax2; position = :ct, framevisible = false, patchsize = (12, 12), padding = (0, 0, 0, 0))

    for (ax, tag) in zip((ax1, ax2), ("A", "B"))
        text!(ax, 0.01, 1.02; text = tag, space = :relative, align = (:left, :top), font = :bold, fontsize = 20)
    end
    return fig
end
