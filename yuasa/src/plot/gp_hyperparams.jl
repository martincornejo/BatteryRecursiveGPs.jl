# Paper figure (supplementary): two-stage hyperparameter selection. Columns = cells |
# modules (module OCV ÷ n → per-cell scale); each column reads top-to-bottom as one story.
# A/B: dV/dSOC fans at the shared initial ℓ, flagged units (> thresh at init) highlighted;
# C/D: fans after per-unit adaptation; E/F: composite-OCV RMSE distribution, initial vs
# adapted (linear mV axis; off-scale outliers annotated as text instead of squeezed in).
# Selected-ℓ counts are tabulated separately (see selection_counts).
# `cells`/`modules` are calc_hyperparam_selection bundles (; fans, comp, rmse, thresh).
function plot_hyperparam_selection(cells, modules)
    fig = Figure(size = (700, 640), figure_padding = 8)
    wong = Makie.wong_colors()
    c_flag = wong[6]  # vermillion: flagged at init (> threshold)
    c_bulk = (:gray, 0.35)
    c_init = :gray30; c_adap = wong[1]

    axs = Matrix{Axis}(undef, 3, 2)
    for (col, lvl) in enumerate((cells, modules))
        # rows 1-2: dV/dSOC fans, initial vs adapted, composite in black
        for (row, stage) in ((1, :init), (2, :adapted))
            ax = Axis(fig[row, col]); axs[row, col] = ax
            fans = subset(lvl.fans, :stage => ByRow(==(stage)))
            for flagged in (false, true), r in eachrow(fans)  # bulk first, flagged on top
                r.flagged == flagged || continue
                lines!(ax, r.soc, r.dvdsoc; color = flagged ? (c_flag, 0.8) : c_bulk, linewidth = flagged ? 1.0 : 0.8)
            end
            lines!(ax, lvl.comp.soc, lvl.comp.dvdsoc; color = :black, linewidth = 2)
            ylims!(ax, -5, 30); xlims!(ax, 0, 100)
            ax.xticks = 0:25:100
            ax.xminorticks = IntervalsBetween(5)   # every 5 % SOC
            ax.xminorticksvisible = true
        end
        axs[2, col].xlabel = "SOC / %"

        # row 3: RMSE shift, initial vs adapted; off-scale outliers noted as text
        ax = Axis(fig[3, col]); axs[3, col] = ax
        xmax = 8.0
        density!(ax, lvl.rmse.init; color = (c_init, 0.2), strokecolor = c_init, strokewidth = 1.8, label = rich("initial ℓ", subscript("OCV")))
        density!(ax, lvl.rmse.adapted; color = (c_adap, 0.2), strokecolor = c_adap, strokewidth = 1.8, label = rich("adapted ℓ", subscript("OCV")))
        # off-scale units: one paired note each, named, numbers in the curve colors
        out = [r for r in eachrow(lvl.rmse) if r.init > xmax || r.adapted > xmax]
        for (i, r) in enumerate(out)
            text!(
                ax, 0.97, 0.95 - 0.15 * (i - 1);
                text = rich(
                    "$(r.name): ", rich(string(round(Int, r.init)), color = c_init), " → ",
                    rich(string(round(Int, r.adapted)), color = c_adap), " mV"
                ),
                space = :relative, align = (:right, :top), fontsize = 10,
            )
        end
        vlines!(ax, [lvl.thresh]; color = :black, linestyle = :dot, linewidth = 1.2)
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
    axs[1, 1].ylabel = "dV/dSOC / (mV/%)"; axs[2, 1].ylabel = "dV/dSOC / (mV/%)"; axs[3, 1].ylabel = "Density"
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

# Paper figure (supplementary): all four GP hyperparameters of every unit — histograms in
# physical units (2 rows cells/modules × 4 equal columns ℓ_ocv/ℓ_r1/σ_ocv/σ_r1, ~40 bins
# each so bar widths match; ℓ_ocv log-x). Bars are stack-colored by the selected
# dimensionless ℓ (gray = 0.5 init, lipari steps for escalations) — the legend speaks the text's
# vocabulary, and the σ columns come out all-gray since σ is never adapted. Units beyond
# any cell capacity (ℓ_ocv > 100 Ah) are named; their 1-count bars would be invisible.
# Inputs are calc_scaled_hyperparams DataFrames.
function plot_hyperparam_scales(scaled_cells, scaled_mods)
    # escalated ℓ values actually selected, ascending (0.5 is the init and gets its own color)
    ℓ_esc = sort(setdiff(union(scaled_cells.ℓ_ocv_rel, scaled_cells.ℓ_r1_rel, scaled_mods.ℓ_ocv_rel, scaled_mods.ℓ_r1_rel), [0.5]))
    colors = Dict(v => get(cgrad(:lipari), t) for (v, t) in zip(ℓ_esc, range(0.8, 0.12, length = length(ℓ_esc))))
    colors[0.5] = RGBAf(0.65, 0.65, 0.65, 1)    # init: neutral gray
    ℓ_all = [0.5; ℓ_esc]                        # init drawn first (bottom of the stacks)

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

    # (value, ℓ_rel-for-color, xscale, xticks, xlims, bin edges, log-spaced?, label)
    specs = [
        (
            x -> x.ℓ_ocv, x -> x.ℓ_ocv_rel, log10, ([10, 20, 50, 100, 200], ["10", "20", "50", "100", "200"]),
            (9, 220), 10 .^ (log10(9):0.035:log10(220)), true, rich("ℓ", subscript("OCV"), " / Ah"),
        ),
        (
            x -> x.ℓ_r1, x -> x.ℓ_r1_rel, identity, (0:20:60, string.(0:20:60)),
            (5, 68), 5:1.5:68, false, rich("ℓ", subscript("R1"), " / Ah"),
        ),
        (
            x -> x.σ_ocv, x -> 0.5, identity, ([180, 230, 280], ["180", "230", "280"]),
            (175, 295), 178:2.9:294, false, rich("σ", subscript("OCV"), " / mV"),
        ),
        (
            x -> x.σ_r1, x -> 0.5, identity, (7:1:10, string.(7:1:10)),
            (6.1, 10.3), 6.2:0.1:10.2, false, rich("σ", subscript("R1"), " / mΩ"),
        ),
    ]

    fig = Figure(size = (700, 360), figure_padding = 8)
    axs = Matrix{Axis}(undef, 2, 4)
    for (col, (val, rel, xscale, xticks, lims, bins, logx, xlab)) in enumerate(specs)
        # Modules*: σ shown per cell (module value ÷ n) for a scale comparable to the cells
        for (row, (a, name)) in enumerate(((scaled_cells, "Cells"), (scaled_mods, "Modules*")))
            ax = Axis(fig[row, col]; xscale, xticks)
            axs[row, col] = ax
            groups = [(Float64[val(x) for x in eachrow(a) if rel(x) == v], colors[v]) for v in ℓ_all]
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
    for r in eachrow(scaled_cells)
        r.ℓ_ocv > 100 || continue
        lines!(axs[1, 1], [r.ℓ_ocv, r.ℓ_ocv], [45, 6]; color = :black, linewidth = 0.8)
        text!(axs[1, 1], r.ℓ_ocv, 50; text = r.name, align = (:right, :bottom), fontsize = 9)
    end
    fmt(v) = v == floor(v) ? string(Int(v)) : string(v)
    order = sort(ℓ_all)
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
