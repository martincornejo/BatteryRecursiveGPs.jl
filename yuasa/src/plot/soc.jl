function plot_soc_trajectories!(
        layout, df;
        title = "",
        # (module-based, cell-based)
        colors = Makie.wong_colors()[[1, 2]],
    )
    ax = Axis(layout[1, 1]; title)
    th = df.t ./ 3600

    for c in 1:12
        μ = Measurements.value.(df[:, "soc_cell_$c"]) .* 100
        σ = Measurements.uncertainty.(df[:, "soc_cell_$c"]) .* 100
        band!(ax, th, μ .- 2σ, μ .+ 2σ; color = (:gray, 0.15))
        lines!(ax, th, μ; color = (:gray, 0.6), linewidth = 1)
    end

    for (col, color) in ((:soc_module, colors[1]), (:soc_pack, colors[2]))
        μ = Measurements.value.(df[!, col]) .* 100
        σ = Measurements.uncertainty.(df[!, col]) .* 100
        band!(ax, th, μ .- 2σ, μ .+ 2σ; color = (color, 0.3))
        lines!(ax, th, μ; color, linewidth = 2)
    end

    xlims!(ax, first(th), last(th))
    ax.xlabel = "Time / h"
    ax.ylabel = "SOC / %"
    ax.xgridvisible = false
    ax.ygridvisible = false
    return ax
end

"""
    plot_soc_comparison(df_norm, df_out; titles = ("", ""), colors = …) -> Figure

SOC trajectories of two modules side by side, each from a [`calc_module_soc`](@ref) table: the
12 cells in gray behind the aggregated string SOC and the module model's own estimate, all
with 2σ bands.
"""
function plot_soc_comparison(df_norm, df_out; titles = ("", ""), colors = Makie.wong_colors()[[1, 2]])
    fig = Figure(size = (700, 320))
    gl = GridLayout(fig[1, 1])
    ax1 = plot_soc_trajectories!(GridLayout(gl[1, 1]), df_norm; colors, title = titles[1])
    ax2 = plot_soc_trajectories!(GridLayout(gl[1, 2]), df_out; colors, title = titles[2])
    linkyaxes!(ax1, ax2)

    elements = [
        [PolyElement(color = (colors[1], 0.3)), LineElement(color = colors[1], linewidth = 2)],
        [PolyElement(color = (colors[2], 0.3)), LineElement(color = colors[2], linewidth = 2)],
        [PolyElement(color = (:gray, 0.15)), LineElement(color = (:gray, 0.6), linewidth = 1)],
    ]
    Legend(
        fig[2, 1], elements, ["Module-based", "Cell-based", "Cells"];
        orientation = :horizontal, framevisible = false,
    )
    return fig
end

# per-module distribution of the SOC error trajectories
function plot_soc_discrepancy!(layout, soc_err)
    ax = Axis(layout[1, 1])
    n_t, n_mod = size(soc_err)
    xs = repeat(1:n_mod; inner = n_t)

    boxplot!(ax, xs, vec(Measurements.value.(soc_err)) .* 100; width = 0.6, color = (:gray, 0.6), whiskerwidth = 0.4, markersize = 3)
    hlines!(ax, [0]; color = (:black, 0.4), linewidth = 1)
    vlines!(ax, [9.5, 18.5]; color = (:black, 0.3), linewidth = 1)

    xlims!(ax, 0.3, n_mod + 0.7)
    ax.ylabel = "Module − cell SOC / %"
    ax.xlabel = "Module ID"
    module_id_xticks!(ax)
    ax.xgridvisible = false
    ax.ygridvisible = false
    return ax
end

"""
    plot_soc_discrepancy(soc_err) -> Figure

Distribution of each module's SOC error over time, as a boxplot per module, from a
[`calc_soc_error`](@ref) matrix. `plot_soc_discrepancy!(layout, soc_err)` draws it into an
existing layout.
"""
function plot_soc_discrepancy(soc_err)
    fig = Figure(size = (700, 300))
    plot_soc_discrepancy!(GridLayout(fig[1, 1]), soc_err)
    return fig
end

"""
    plot_soc_overview(df_norm, df_out, soc_err; titles = ("", ""), colors = …) -> Figure

The fleet SOC discrepancy over (A) [`plot_soc_discrepancy`](@ref) for all modules, and below
it (B, C) the two example modules of [`plot_soc_comparison`](@ref).
"""
function plot_soc_overview(
        df_norm, df_out, soc_err;
        titles = ("", ""), colors = Makie.wong_colors()[[1, 2]],
    )
    fig = Figure(size = (700, 530))

    ax_disc = plot_soc_discrepancy!(GridLayout(fig[1, 1:2]), soc_err)

    ax1 = plot_soc_trajectories!(GridLayout(fig[2, 1]), df_norm; colors, title = titles[1])
    ax2 = plot_soc_trajectories!(GridLayout(fig[2, 2]), df_out; colors, title = titles[2])
    linkyaxes!(ax1, ax2)
    hideydecorations!(ax2; ticks = false, grid = false)

    elements = [
        [PolyElement(color = (colors[1], 0.3)), LineElement(color = colors[1], linewidth = 2)],
        [PolyElement(color = (colors[2], 0.3)), LineElement(color = colors[2], linewidth = 2)],
        [PolyElement(color = (:gray, 0.15)), LineElement(color = (:gray, 0.6), linewidth = 1)],
    ]
    Legend(
        fig[3, 1:2], elements, ["Module-based", "Cell-based", "Cells"];
        orientation = :horizontal, framevisible = false,
    )

    rowsize!(fig.layout, 1, Auto(1.0))
    rowsize!(fig.layout, 2, Auto(1.2))

    for (ax, tag) in zip((ax_disc, ax1, ax2), ("A", "B", "C"))
        text!(ax, 0.01, 1.0; text = tag, space = :relative, align = (:left, :top), font = :bold, fontsize = 20)
    end

    return fig
end

# strip of per-cell RMSEs by module, with the combined-cell and module-model ticks over it
function plot_v_accuracy!(layout, df_v_cell, df_v_module; legend = true, colors = Makie.wong_colors()[[1, 2, 6]])
    ax = Axis(layout[1, 1])
    q99 = quantile(df_v_cell.rmse, 0.99)

    # focus the axis on the bulk + module comparison; the lone extreme cell (a desync
    # artifact) is named off-scale rather than allowed to stretch the whole axis.
    ytop = ceil(max(maximum(df_v_module.rmse_module) / 12, q99) * 1.05)
    # floor the axis just below the lowest point (not at 0) so the 4–14 mV band fills the panel
    ybot = floor(min(minimum(df_v_cell.rmse), minimum(df_v_module.rmse_module ./ 12))) - 0.5

    ax.xlabel = "Module ID"
    ax.ylabel = "Voltage RMSE / mV"
    mod_idx = Dict((; p, m) => (p - 1) * 9 + m for p in 1:3, m in 1:9)
    xs = [mod_idx[(; p = r.p, m = r.m)] for r in eachrow(df_v_cell)]

    # every on-scale cell as a point (the off-scale cell is named below) + a per-module
    # combined tick (cell models as a virtual module, ÷12); the point cloud shows whether a
    # module's spread is one stray cell or broad, the tick the like-for-like module comparison
    ks = 1:nrow(df_v_module)
    keep = df_v_cell.rmse .<= ytop
    cell_combined = df_v_module.rmse_cells ./ 12

    scatter!(ax, xs[keep], df_v_cell.rmse[keep]; color = (colors[2], 0.55), markersize = 7)
    scatter!(ax, ks, cell_combined; marker = :hline, markersize = 15, color = colors[2])
    scatter!(ax, ks, df_v_module.rmse_module ./ 12; color = colors[1], markersize = 9)

    # name the single worst cell in place — it is the off-scale point readers ask about
    worst = argmax(df_v_cell.rmse)
    if df_v_cell.rmse[worst] > ytop
        r = df_v_cell[worst, :]
        y_mark = ytop - 0.035 * (ytop - ybot)  # just below the top edge so the ▲ shows fully
        scatter!(ax, [xs[worst]], [y_mark]; color = colors[3], markersize = 11, marker = :utriangle)
        text!(
            ax, xs[worst] + 0.5, y_mark; text = "P$(r.p)M$(r.m)C$(r.c): $(round(Int, r.rmse)) mV",
            align = (:left, :center), fontsize = 12, color = colors[3]
        )
    end

    vlines!(ax, [9.5, 18.5]; color = (:black, 0.3), linewidth = 1)
    xlims!(ax, 0.3, 27.7)
    module_id_xticks!(ax)
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.yminorticks = IntervalsBetween(5)  # 1 mV steps between the 5 mV majors
    ax.yminorticksvisible = true
    ylims!(ax, ybot, ytop)

    # three entries below the axis (outside the data) — individual cells (cloud), the 12 cells
    # combined as a virtual module (tick), and the module model. orange = cell-level approach,
    # blue = module-level; per-cell scaling (÷12) is a derivation detail, kept to the caption.
    # placed below rather than inside because the tall points span the full width of the axis.
    if legend
        cell_pt_el = MarkerElement(; marker = :circle, color = (colors[2], 0.55), markersize = 7)
        cell_comb_el = MarkerElement(; marker = :hline, color = colors[2], markersize = 15)
        mod_el = MarkerElement(; marker = :circle, color = colors[1], markersize = 9)
        Legend(
            layout[2, 1],
            [cell_pt_el, cell_comb_el, mod_el],
            ["Cell-level (individual)", "Cell-level (virtual module)", "Module-level"];
            orientation = :horizontal, framevisible = false,
        )
    end
    return ax
end

"""
    plot_v_accuracy(df_v_cell, df_v_module; colors = …) -> Figure

Fleet-wide voltage accuracy, grouped by module: every cell's RMSE as a point, the 12 cells
combined as a virtual module as a tick, and the dedicated module model as a second point.
Consumes [`calc_v_summary`](@ref) and [`calc_module_v_summary`](@ref) tables.

Module quantities are drawn per cell (÷12) so all three sit on one scale. The combined tick,
not the mean of the per-cell RMSEs, is the like-for-like counterpart to the module model,
since both score a sum of cell voltages. Cells above the axis limit are named in place.
"""
function plot_v_accuracy(df_v_cell, df_v_module; colors = Makie.wong_colors()[[1, 2, 6]])
    fig = Figure(size = (700, 360))
    plot_v_accuracy!(GridLayout(fig[1, 1]), df_v_cell, df_v_module; colors)
    return fig
end

"""
    plot_v_accuracy_overview(model, sol, df_v_cell, df_v_module; Ts = 1.0, title = "") -> Figure

One example open-loop cell fit from [`plot_sim`](@ref) over the fleet-wide accuracy of
[`plot_v_accuracy`](@ref), as panels A and B.
"""
function plot_v_accuracy_overview(model::AbstractBatteryModel, sol, df_v_cell, df_v_module; Ts = 1.0, title = "")
    fig = Figure(size = (700, 550))

    gl_sim = GridLayout(fig[1, 1])
    ax_v, _ = plot_sim!(gl_sim, model, sol; Ts)
    ax_v.title = title

    plot_v_accuracy!(GridLayout(fig[2, 1]), df_v_cell, df_v_module)

    Label(gl_sim[1, 1, TopLeft()], "A"; font = :bold, fontsize = 18, padding = (0, 0, 5, 0))
    Label(fig[2, 1, TopLeft()], "B"; font = :bold, fontsize = 18, padding = (0, 0, 5, 0))
    rowgap!(fig.layout, -10)
    # panel A stacks two subplots (voltage + error) and panel B's row also carries the legend,
    # so give row B more weight to keep the two plotting areas similar in size
    rowsize!(fig.layout, 1, Auto(1))
    rowsize!(fig.layout, 2, Auto(1.5))
    return fig
end

"""
    plot_charge_error(df; color = …, size = (550, 250)) -> Figure

SOC error against the oscilloscope reference over time, one line per cell of the reference
module. Consumes a [`calc_charge_error`](@ref) table.
"""
function plot_charge_error(df; color = Makie.wong_colors()[1], size = (550, 250))
    fig = Figure(; size)
    ax = Axis(fig[1, 1], xlabel = "Time / h", ylabel = "SOC error / %")
    cols = filter(!=("t"), names(df))

    hlines!(ax, [0]; color = :gray, linewidth = 0.5)
    for c in cols
        lines!(ax, df.t, df[!, c]; color = (color, 0.7), linewidth = 1.5)
    end

    xlims!(ax, extrema(df.t)...)
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.yminorticksvisible = true
    return fig
end

"""
    plot_soc_diagnostic(diag; color = :dodgerblue) -> Figure

One column per injected-fault scenario from [`calc_soc_diagnostic`](@ref): the charge
trajectory on top and its error below, for the oscilloscope reference, Coulomb counting, and
the filter estimate with a 2σ band.
"""
function plot_soc_diagnostic(diag; color = :dodgerblue)
    fig = Figure(size = (700, 400))
    ax = [Axis(fig[i, j]) for i in 1:2, j in 1:2]
    for (col, d) in enumerate(diag)
        e_cc, e = d.q_cc .- d.q_ref, d.q .- d.q_ref

        xlims!(ax[1, col], d.t[begin], d.t[end])
        lines!(ax[1, col], d.t, d.q_ref; color = :black, label = "Reference (oscilloscope CC)")
        lines!(ax[1, col], d.t, d.q_cc; color = :red, linestyle = :dash, label = "Coulomb counting (module + faults)")
        band!(ax[1, col], d.t, d.q .- 2d.qσ, d.q .+ 2d.qσ; color = (color, 0.5), label = "EKF estimate")
        lines!(ax[1, col], d.t, d.q; color, label = "EKF estimate")
        hidexdecorations!(ax[1, col]; ticks = false)

        xlims!(ax[2, col], d.t[begin], d.t[end])
        ax[2, col].yautolimitmargin = (0.1, 0.1)
        hlines!(ax[2, col], 0; color = :black, linestyle = :dot)
        lines!(ax[2, col], d.t, e_cc; color = :red, linestyle = :dash)
        lines!(ax[2, col], d.t, e; color)
        band!(ax[2, col], d.t, e .- 2d.qσ, e .+ 2d.qσ; color = (color, 0.5))
        linkxaxes!(ax[1, col], ax[2, col])

        ax[1, col].title = d.offset == 0 && d.bias == 0 ? "No faults" : "$(d.offset) Ah offset + $(d.bias) A current bias"
        ax[2, col].xlabel = "Time / h"
    end
    for col in 1:2
        ax[1, col].ylabel = "Charge / Ah"
        ax[2, col].ylabel = "Error / Ah"
    end
    foreach(a -> (a.xgridvisible = false; a.ygridvisible = false), ax)
    linkyaxes!(ax[1, 1], ax[1, 2])
    linkyaxes!(ax[2, 1], ax[2, 2])
    rowsize!(fig.layout, 2, Relative(0.3))
    for (a, tag) in zip((ax[1, 1], ax[1, 2], ax[2, 1], ax[2, 2]), ("A", "B", "C", "D"))
        text!(a, 0.02, 0.98; text = tag, space = :relative, align = (:left, :top), font = :bold, fontsize = 20)
    end
    Legend(fig[3, 1:2], fig.content[1]; orientation = :horizontal, merge = true, framevisible = false)
    return fig
end

"""
    plot_soc_discrepancy_heatmap(tg, soc_err) -> Figure

The [`calc_soc_error`](@ref) trajectories as a module × time heatmap, keeping the temporal
structure that [`plot_soc_discrepancy`](@ref) collapses.
"""
function plot_soc_discrepancy_heatmap(tg, soc_err)
    fig = Figure(size = (700, 340))
    ax = Axis(fig[1, 1], xlabel = "Time / h", ylabel = "Module ID")
    n_mod = size(soc_err, 2)

    soc_err_v = Measurements.value.(soc_err) * 100
    cr = maximum(abs, soc_err_v)
    hm = heatmap!(ax, tg ./ 3600, 1:n_mod, soc_err_v; colormap = :vik, colorrange = (-cr, cr))
    hlines!(ax, [9.5, 18.5]; color = :black, linewidth = 1)
    Colorbar(fig[1, 2], hm; label = "Module − cell SOC / %")

    ax.yticks = ([1, 6, 10, 15, 19, 24], ["P1M1", "P1M6", "P2M1", "P2M6", "P3M1", "P3M6"])
    return fig
end
