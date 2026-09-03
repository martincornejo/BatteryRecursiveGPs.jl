"""
    plot_cell_voltage_system(data; panel_tags = true, highlights = …) -> Figure

Cell voltages of the whole system, one panel per module in a 9 × 3 grid of module by phase.
`highlights` maps a `(phase, module)` pair to the colour its panel is drawn in.
"""
function plot_cell_voltage_system(
        data; panel_tags = true,
        # highlight the outlier modules discussed in the text
        highlights = Dict((3, 5) => :firebrick, (1, 6) => :firebrick),
        size = (700, 600),
    )
    df = copy(data[:cell_voltage])
    fig = Figure(; size)
    ax = [Axis(fig[i, j]) for i in 1:9, j in 1:3]

    t0 = first(df._time)
    df[!, :t] = Dates.value.(df._time .- t0) * 1.0e-3 # seconds
    phases = 1:3
    modules = 1:9

    for p in phases, m in modules
        for i in 1:12
            lines!(ax[m, p], df.t / 3600, df[:, "cell_voltage_$(p)_$(m)_1_$(i)"])
        end
    end

    for i in modules, j in phases[2:end]
        hideydecorations!(ax[i, j], ticks = false)
    end
    for i in modules[1:(end - 1)], j in phases
        hidexdecorations!(ax[i, j], ticks = false)
    end

    for ((p, m), color) in highlights
        for spine in (:topspinecolor, :bottomspinecolor, :leftspinecolor, :rightspinecolor)
            setproperty!(ax[m, p], spine, color)
        end
        ax[m, p].spinewidth = 2
    end

    for p in phases, m in modules
        hlines!(ax[m, p], [3.4, 4.07]; color = (:black, 0.3), linestyle = :dash, linewidth = 0.8)
    end

    for i in modules, j in phases
        ylims!(ax[i, j], 3.35, 4.15)
        xlims!(ax[i, j], first(df.t) / 3600, last(df.t) / 3600)
        ax[i, j].xticks = 0:4:12
        ax[i, j].yticks = [3.5, 4.0]
        ax[i, j].xminorticks = IntervalsBetween(5)
        ax[i, j].xminorticksvisible = true
        ax[i, j].yminorticks = IntervalsBetween(2)
        ax[i, j].yminorticksvisible = true
        ax[i, j].xgridvisible = false
        ax[i, j].ygridvisible = false
    end

    if panel_tags
        for p in phases, m in modules
            text!(
                ax[m, p], 0.97, 0.06; text = "P$(p)M$(m)",
                space = :relative, align = (:right, :bottom), font = :bold, fontsize = 12,
                color = get(highlights, (p, m), :gray40),  # tag matches the frame color
            )
        end
    else
        for i in 1:9
            Label(fig[i, 4], "Module $i", font = :bold, fontsize = 11, rotation = pi / 2, tellheight = false)
        end
        for j in 1:3
            Label(fig[0, j], "Phase $j", font = :bold, fontsize = 11, tellwidth = false)
        end
    end
    for j in 1:3
        ax[end, j].xlabel = "Time / h"
    end

    ax[5, 1].ylabel = "Cell voltages / V"

    rowgap!(fig.layout, 2.5)
    colgap!(fig.layout, 2.5)
    return fig
end

"""
    plot_data_resolution(data; completeness = nothing, yscale = log10) -> Figure

Sampling-interval histogram per signal table. Pass a [`calc_data_completeness`](@ref) table as
`completeness` to annotate each panel with the fraction of expected samples present.
"""
function plot_data_resolution(data; completeness = nothing, yscale = log10, size = (600, 550))
    avail = isnothing(completeness) ? nothing : Dict(r.signal => r.completeness for r in eachrow(completeness))
    colors = Makie.wong_colors()
    # signal → color mapping matches plot_dataset_overview
    signals = [
        (:module_current, "Module current", colors[2]),
        (:module_voltage, "Module voltage", colors[3]),
        (:battery_temperature, "Module temperature", colors[4]),
        (:cell_voltage, "Cell voltage", colors[1]),
    ]
    logscale = yscale === log10

    fig = Figure(; size)
    ax = [
        Axis(
                fig[i, 1]; yscale, ylabel = "Count", titlealign = :left, titlesize = 12,
                xticks = 0:10:80, xminorticks = IntervalsBetween(10), xminorticksvisible = true
            )
            for i in 1:4
    ]
    bins = 0.5:1:80.5  # integer-second timestamps, center bars on integers

    nmax = 0
    for (i, (key, name, color)) in enumerate(signals)
        Δt = Dates.value.(diff(data[key][!, "_time"])) * 1.0e-3
        hist!(
            ax[i], Δt; strokewidth = 1, strokecolor = :black, color, bins,
            fillto = logscale ? 0.5 : 0.0
        )
        nmax = max(nmax, maximum(StatsBase.fit(Histogram, Δt, bins).weights))
        ax[i].title = if isnothing(avail)
            "$name  (median $(round(Int, median(Δt))) s)"
        else
            "$name  (median $(round(Int, median(Δt))) s, $(round(Int, 100avail[key]))% available)"
        end
    end

    linkaxes!(ax...)
    xlims!(ax[1], 0, 81)
    logscale && ylims!(ax[1], 0.5, 2nmax)
    foreach(a -> hidexdecorations!(a; grid = false, ticks = false, minorticks = false), ax[1:3])
    ax[4].xlabel = "Sampling interval / s"

    return fig
end

"""
    plot_module_data(data; N = 5, zoom = (3.05, 3.22)) -> Figure

Module voltage, current and temperature for all 27 modules, coloured by module ID. Group A
covers the full window decimated by `N`; group B redraws the `zoom` hours shaded in A at full
resolution, where the individual switching events are resolvable.
"""
function plot_module_data(data; N = 5, zoom = (3.05, 3.22), size = (600, 660))
    fig = Figure(; size)
    gl_full = fig[1, 1] = GridLayout()
    gl_zoom = fig[2, 1] = GridLayout()
    axf = [Axis(gl_full[i, 1]) for i in 1:3]
    axz = [Axis(gl_zoom[i, 1]) for i in 1:2]

    colors = MODULE_COLORS

    t0 = data[:module_voltage]._time[begin]
    t_end = Dates.value(data[:module_voltage]._time[end] - t0) * 1.0e-3 / 3600

    ids = [(; p, m) for p in 1:3, m in 1:9] |> vec |> sort |> reverse
    for id in ids
        (; p, m) = id
        df_V = select(data[:module_voltage], "_time" => "time", "module_voltage_$(p)_$(m)" => "value")
        df_i = select(data[:module_current], "_time" => "time", "module_average_current_$(p)_$(m)" => ByRow(x -> -x) => "value")
        df_T = select(data[:battery_temperature], "_time" => "time", "battery_sensor_temperature_$(p)_$(m)_1" => "value")
        for (k, df) in enumerate((df_V, df_i, df_T))
            df[!, :t] = Dates.value.(df.time .- t0) * 1.0e-3 / 3600
            lines!(axf[k], df.t[1:N:end], df.value[1:N:end]; color = colors[m])
        end
        # zoom: full resolution, no decimation, so the switching is not aliased away
        for (k, df) in ((1, df_V), (2, df_i))
            w = (df.t .>= zoom[1]) .& (df.t .<= zoom[2]) .& .!ismissing.(df.value)
            lines!(axz[k], (df.t[w] .- zoom[1]) .* 60, df.value[w]; color = colors[m])
        end
    end

    for i in (1, 2)
        vspan!(axf[i], zoom[1], zoom[2]; color = (:black, 0.3))
    end

    for i in eachindex(axf)
        xlims!(axf[i], 0, t_end)
        axf[i].xgridvisible = false
        axf[i].ygridvisible = false
        axf[i].xminorticksvisible = true
        axf[i].xminorticks = IntervalsBetween(5)

        if i < 3
            hidexdecorations!(axf[i], ticks = false, minorticks = false)
        end
    end
    for i in eachindex(axz)
        xlims!(axz[i], 0, (zoom[2] - zoom[1]) * 60)
        axz[i].xgridvisible = false
        axz[i].ygridvisible = false
        axz[i].xminorticksvisible = true
        axz[i].xminorticks = IntervalsBetween(5)
    end
    hidexdecorations!(axz[1], ticks = false, minorticks = false)

    axf[1].ylabel = "Voltage / V"
    axf[2].ylabel = "Current / A"
    axf[3].ylabel = "Temperature / °C"
    axf[3].xlabel = "Time / h"
    axz[1].ylabel = "Voltage / V"
    axz[2].ylabel = "Current / A"
    axz[2].xlabel = "Time / min"

    rowsize!(fig.layout, 2, Relative(0.32))
    Label(gl_full[1, 1, TopLeft()], "A"; font = :bold, fontsize = 16, padding = (0, 0, 2, 0))
    Label(gl_zoom[1, 1, TopLeft()], "B"; font = :bold, fontsize = 16, padding = (0, 0, 2, 0))

    mod_elems = [LineElement(color = colors[m], linewidth = 3) for m in 1:9]
    Legend(
        fig[1:2, 2], mod_elems, ["M$m" for m in 1:9], "Module ID";
        orientation = :vertical, titleposition = :top, framevisible = false
    )

    return fig
end

"""
    plot_dataset_overview(data; id_norm = (3, 7), id_out = (3, 5)) -> Figure

Two modules side by side over the full window, four rows each: cell voltages, module voltage,
current and temperature. `id_norm` and `id_out` are the `(phase, module)` pairs to draw.
"""
function plot_dataset_overview(data; id_norm = (3, 7), id_out = (3, 5), size = (700, 450))
    fig = Figure(; size)
    colors = Makie.wong_colors()

    df_v = copy(data[:cell_voltage])
    t0 = first(df_v._time)
    df_v[!, :t] = Dates.value.(df_v._time .- t0) * 1.0e-3 / 3600 # time in hours

    df_dr = coalesce.(data[:derating_current], NaN)
    t_dr = Dates.value.(df_dr._time .- t0) * 1.0e-3 / 3600

    axs = [Axis(fig[i, j]) for i in 1:4, j in 1:2]
    for (j, (p, m)) in enumerate((id_norm, id_out))
        for i in 1:12
            lines!(axs[1, j], df_v.t, df_v[:, "cell_voltage_$(p)_$(m)_1_$(i)"])
        end

        df_V = select(data[:module_voltage], "_time" => "time", "module_voltage_$(p)_$(m)" => "value")
        df_i = select(data[:module_current], "_time" => "time", "module_average_current_$(p)_$(m)" => ByRow(x -> -x) => "value")
        df_T = select(data[:battery_temperature], "_time" => "time", "battery_sensor_temperature_$(p)_$(m)_1" => "value")
        N = 5
        for (k, df) in enumerate((df_V, df_i, df_T))
            df[!, :t] = Dates.value.(df.time .- t0) * 1.0e-3 / 3600
            lines!(axs[1 + k, j], df.t[1:N:end], df.value[1:N:end]; color = colors[[3, 2, 4][k]])
        end
        # BMS limits: cell/module voltage window and derating current envelope
        hlines!(axs[1, j], [3.4, 4.07]; color = (:black, 0.6), linestyle = :dash)
        hlines!(axs[2, j], 12 .* [3.4, 4.07]; color = (:black, 0.6), linestyle = :dash)
        lines!(axs[3, j], t_dr, df_dr[:, "dr_ch_p$(p)_m$(m)"]; color = (:black, 0.6), linestyle = :dash)
        lines!(axs[3, j], t_dr, -df_dr[:, "dr_dch_p$(p)_m$(m)"]; color = (:black, 0.6), linestyle = :dash)

        axs[1, j].title = "Module P$(p)M$(m)"
        axs[4, j].xlabel = "Time / h"
        for k in 1:4
            xlims!(axs[k, j], first(df_v.t), last(df_v.t))
            axs[k, j].xticks = 0:4:12
        end
    end

    # same voltage range and 0.25 V/cell tick lattice as the ECM comparison figure
    ylims!.(axs[1, :], 3.35, 4.15)
    ylims!.(axs[2, :], 12 * 3.35, 12 * 4.15)
    ylims!.(axs[3, :], -110, 110)  # full derating range: no-limit level is ±100 A
    ylims!.(axs[4, :], 18, 33)
    for j in 1:2
        axs[1, j].yticks = 3.5:0.25:4.0
        axs[2, j].yticks = 42:3:48
        axs[3, j].yticks = -100:50:100
        axs[4, j].yticks = 20:5:30
    end
    axs[1, 1].ylabel = "Cell\nvoltages / V"
    axs[2, 1].ylabel = "Module\nvoltage / V"
    axs[3, 1].ylabel = "Module\ncurrent / A"
    axs[4, 1].ylabel = "Module\ntemp. / °C"

    for i in 1:4
        hideydecorations!(axs[i, 2], ticks = false, grid = false)
        i < 4 && hidexdecorations!.(axs[i, :], ticks = false, grid = false)
        for j in 1:2
            axs[i, j].xgridvisible = false
            axs[i, j].ygridvisible = false
            axs[i, j].xminorticks = IntervalsBetween(5)
            axs[i, j].xminorticksvisible = true
            axs[i, j].yminorticks = IntervalsBetween(2)
            axs[i, j].yminorticksvisible = true
        end
    end
    rowgap!(fig.layout, 8)
    colgap!(fig.layout, 10)
    return fig
end
