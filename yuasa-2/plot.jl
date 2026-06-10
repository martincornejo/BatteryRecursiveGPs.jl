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

    ax.ytickformat = values -> begin
        map(values) do value
            p, m = divrem(value, 9)
            "P$(Int(p) + 1)M$(Int(m))"
        end
    end
    ax.yticks = [1, 6, 10, 15, 19, 24] #27
    ax.yminorticks = 1:27
    ax.yminorticksvisible = true

    colormap = :lipari
    colorrange = (65, 87)
    lowclip = :black
    soh = reshape(Measurements.value.(comp_fit.Q_cell), 12, 27)
    heatmap!(ax, soh; colorrange, colormap, lowclip)
    Colorbar(layout[1, 2]; limits = colorrange, colormap, lowclip, label = "Cell Capacity / Ah")
    return nothing
end

function plot_composite_ocv(comp_fit, cells; xaxis = :soc)
    fig = Figure(size = (400, 400))
    gl = GridLayout(fig[1, 1])
    plot_composite_ocv!(gl, comp_fit, cells; xaxis)
    return fig
end


function plot_composite_ocv!(layout, comp_fit, cells; xaxis = :soc, vertical = true)
    (; soc_grid, v_grid, Q_cell, s0, Q_full) = comp_fit
    xscale = xaxis == :ah ? Q_full : 1.0
    xlabel = xaxis == :ah ? "Capacity / Ah" : "SOC"

    if vertical
        ax1 = Axis(layout[1, 1]; ylabel = "Voltage / V")
        ax2 = Axis(layout[2, 1]; ylabel = "dV/d$(xlabel)", xlabel)
        hidexdecorations!(ax1; grid = false, ticks = false)
        linkxaxes!(ax1, ax2)
        rowgap!(layout, 1, 5)
    else
        ax1 = Axis(layout[1, 1]; ylabel = "Voltage / V", xlabel)
        ax2 = Axis(layout[1, 2]; ylabel = "dV/d$(xlabel)", xlabel)
    end

    Q_cell_μ = Measurements.value.(Q_cell)
    s0_μ = Measurements.value.(s0)
    colormap = :lipari
    colorrange = (65, 87)
    lowclip = :black
    for i in eachindex(cells)
        q = cells[i].q
        v = cells[i].μ
        x = (q ./ Q_cell_μ[i] .+ s0_μ[i]) .* xscale
        lines!(ax1, x, v; color = Q_cell_μ[i], colorrange, colormap, alpha = 0.6, label = "Cell OCV")
        lines!(ax2, x[2:end], diff(v) ./ diff(x); color = Q_cell_μ[i], colorrange, colormap, alpha = 0.4)
    end

    order = sortperm(soc_grid)
    itp = PCHIPInterpolation(v_grid[order], soc_grid[order])
    soc_dense = range(soc_grid[order[1]], soc_grid[order[end]]; length = 500)
    dv_dx_raw = [DataInterpolations.derivative(itp, s) / xscale for s in soc_dense]
    hw = 15
    n = length(dv_dx_raw)
    dv_dx = [mean(dv_dx_raw[max(1, i - hw):min(n, i + hw)]) for i in 1:n]

    x_comp = soc_grid .* xscale
    lines!(ax1, x_comp, v_grid; color = :black, linewidth = 2, label = "Composite OCV")
    lines!(ax2, collect(soc_dense) .* xscale, dv_dx; color = :black, linewidth = 2)
    ylims!(ax2, 0, nothing)
    ax1.xgridvisible = false
    ax1.ygridvisible = false
    ax2.xgridvisible = false
    ax2.ygridvisible = false

    axislegend(ax1; position = :rb, merge = true, framevisible = false)
    return
end

function plot_soh_hist(comp_fit)
    fig = Figure()
    plot_soh_hist!(fig, comp_fit)
    return fig
end

function plot_soh_hist!(fig, comp_fit)
    ax = Axis(fig[1, 1], xlabel = "Cell Capacity / Ah", ylabel = "Cell Count")
    ylims!(ax, 0, nothing)

    bins = 40.5:85.5
    Q_cell = Measurements.value.(comp_fit.Q_cell)
    hist!(ax, Q_cell; bins, color = :gray, strokewidth = 1)
    return
end

function plot_cell_soh(comp_fit, cells)
    fig = Figure(size = (700, 370))
    gl1 = GridLayout(fig[1, 1])
    gl2 = GridLayout(fig[1, 2])
    plot_composite_ocv!(gl1, comp_fit, cells; vertical = true)
    plot_soh_heatmap!(gl2, comp_fit)
    colsize!(fig.layout, 2, Relative(0.4))
    Label(fig[1, 1, TopLeft()], "A"; fontsize = 20, font = :bold, padding = (0, 0, 5, 0))
    Label(fig[1, 2, TopLeft()], "B"; fontsize = 20, font = :bold, padding = (0, 0, 5, 0))
    return fig
end

fig1 = plot_composite_ocv(comp_ocv_scaled, cell_ocvs)
fig2 = plot_soh_heatmap(comp_ocv_scaled)

fig3 = plot_cell_soh(comp_ocv_scaled, cell_ocvs)

plot_soh_hist(comp_ocv_scaled)
