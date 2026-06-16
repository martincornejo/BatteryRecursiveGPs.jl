# let df = data[:cell_voltage]
#     for p in 1:3, m in 1:9
#         fig = Figure()
#         ax = Axis(fig[1, 1])

#         for i in 1:12
#             lines!(ax, df._time, df[:, "cell_voltage_$(p)_$(m)_1_$(i)"])
#         end
#         ax.title = "Phase: $(p), Module: $(m)" #, Cell: $(i)"
#         fig |> display
#     end
# end

function plot_cell_voltage(data, p, m, c)
    fig = Figure()
    ax = Axis(fig[1, 1])

    df = data[:cell_voltage]
    for i in c
        lines!(ax, df._time, df[:, "cell_voltage_$(p)_$(m)_1_$(i)"])
    end
    ax.title = "Phase: $(p), Module: $(m), Cell: $(c)" #, Cell: $(i)"

    xlims!(ax, first(df._time), last(df._time))
    ylims!(ax, 3.35, 4.1)
    return fig
end

# for p in 1:3, m in 1:9
#     fig = plot_cell_voltage(data, p, m, 1:12)
#     fig |> display
# end

function plot_cell_voltage_system(
        data; panel_tags = true,
        # highlight the outlier modules discussed in the text
        highlights = Dict((3, 5) => :firebrick, (1, 6) => :firebrick),
    )
    df = copy(data[:cell_voltage])
    fig = Figure(size = (700, 600))
    ax = [Axis(fig[i, j]) for i in 1:9, j in 1:3]

    t0 = first(df._time)
    df[!, :t] = Dates.value.(df._time .- t0) * 1.0e-3 # seconds
    phases = 1:3
    modules = 1:9

    for p in phases, m in modules

        for i in 1:12
            lines!(ax[m, p], df.t / 3600, df[:, "cell_voltage_$(p)_$(m)_1_$(i)"])
        end
        # ax[m, p].title = "Phase: $(p), Module: $(m)" #, Cell: $(i)"
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

# let df = data[:module_current]
#     for p in 1:3, m in 1:9
#         fig, ax = lines(df._time, df[:, "module_average_current_$(p)_$(m)"])
#         ax.title = "Phase: $(p), Module: $(m)"
#         fig |> display
#     end
# end

# let df = data[:derating_current]
#     df = coalesce.(df, NaN)
#     # for p in 1:3, m in 1:9
#     p = 3
#     m = 5
#     fig = Figure()
#     ax = Axis(fig[1, 1])
#     # df_ch = dropmissing(df, "dr_ch_p$(p)_m$(m)")
#     # df_dc = dropmissing(df, "dr_ch_p$(p)_m$(m)")
#     lines!(ax, df._time, df[:, "dr_ch_p$(p)_m$(m)"])
#     lines!(ax, df._time, -df[:, "dr_dch_p$(p)_m$(m)"])
#     ax.title = "Phase: $(p), Module: $(m)"
#     fig |> display
#     # end
# end

function plot_derating_phase(data)
    # df = dropmissing(data[:derating_current])
    df = coalesce.(data[:derating_current], NaN)
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:3]

    for p in 1:3
        for m in 1:9
            lines!(ax[p], df._time, df[:, "dr_ch_p$(p)_m$(m)"]; color = Cycled(m))
            lines!(ax[p], df._time, -df[:, "dr_dch_p$(p)_m$(m)"]; color = Cycled(m))
        end
    end

    hidexdecorations!(ax[1], ticks = false, grid = false)
    hidexdecorations!(ax[2], ticks = false, grid = false)
    linkxaxes!(ax...)
    return fig
end


function plot_derating_state(data, p, m)
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]

    ## cell voltage
    ids = ["cell_voltage_$(p)_$(m)_1_$(i)" for i in 1:12]
    df_v = select(
        data[:cell_voltage],
        "_time",
        AsTable(ids) => ByRow(minimum) => :min_voltage,
        AsTable(ids) => ByRow(maximum) => :max_voltage,
    )
    lines!(ax[1], df_v._time, df_v.min_voltage)
    lines!(ax[1], df_v._time, df_v.max_voltage)
    hlines!(ax[1], [3.4, 4.05]; color = :gray, linestyle = :dash)

    # cell SOCs (estimated)
    # ids = ["cell_state_of_charge_$(p)_$(m)_1_$(i)" for i in 1:12]
    # df_soc = select(data[:cell_soc],
    #     "_time",
    #     AsTable(ids) => ByRow(minimum) => :min_soc,
    #     AsTable(ids) => ByRow(maximum) => :max_soc,
    # )
    # lines!(ax[1], df_soc._time, df_soc.min_soc)
    # lines!(ax[1], df_soc._time, df_soc.max_soc)
    # ylims!(ax[1], 0.0, 0.95)

    ## Derating current (max charge/discharge current)
    df_dr = coalesce.(data[:derating_current], NaN) # convert missing values to NaN for plot
    lines!(ax[2], df_dr._time, -df_dr[:, "dr_dch_p$(p)_m$(m)"])
    lines!(ax[2], df_dr._time, df_dr[:, "dr_ch_p$(p)_m$(m)"])

    ax[1].title = "Phase: $(p), Module: $(m)"
    return fig
end

# for p in 1:3, m in 1:9
#     plot_derating_state(data, p, m) |> display
# end

function plot_current_sensor_error(data; fillmissing = false)

    if fillmissing
        df_i = ffill(data[:module_current], Second(1))
        select!(df_i, "_time" => "time", "module_average_current_1_9" => "i")
    else
        df_i = select(data[:module_current], "_time" => "time", "module_average_current_1_9" => "i")
    end

    df_î = CSV.File(datadir * "oscilloscope_p1_m9.csv"; dateformat = dateformat"y-m-dTH:M:S.sss+00:00") |> DataFrame

    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:3]

    lines!(ax[1], df_i.time, df_i.i)
    lines!(ax[1], df_î.timestamp_utc, df_î.MEAS1)
    # lines!(ax[1], df_î.timestamp_utc, df_î.MEAS3)


    ## Current error
    t0 = first(df_î.timestamp_utc)
    fi = ConstantInterpolation(df_î.MEAS1, Dates.value.(df_î.timestamp_utc - t0) * 1.0e-3)

    ts = Dates.value.(df_i.time - t0) * 1.0e-3
    idx = findall(>(0.0), ts)
    i2 = fi(ts[idx])

    Δi = df_i[idx, :i] - i2
    lines!(ax[2], df_i[idx, :time], Δi)


    ## Coloumb counting error
    dt1 = [Dates.value.(diff(df_i.time)) * 1.0e-3; 0]
    cc1 = -cumsum(df_i.i .* dt1) / 3600

    dt2 = [Dates.value.(diff(df_î.timestamp_utc)) * 1.0e-3; 0]
    cc2 = -cumsum(df_î.MEAS1 .* dt2) / 3600

    lines!(ax[3], df_i.time, cc1)
    lines!(ax[3], df_î.timestamp_utc, cc2)


    ## Axes
    t_start = first(df_i.time)
    t_end = last(df_î.timestamp_utc)

    xlims!(ax[1], t_start, t_end)
    xlims!(ax[2], t_start, t_end)
    xlims!(ax[3], t_start, t_end)
    linkxaxes!(ax...)

    ax[1].ylabel = "Avg. current / A"
    ax[2].ylabel = "Sensor error / A"
    ax[3].ylabel = "Coloumb counting / Ah"

    return fig
end


# let df = data[:battery_temperature]
#     fig = Figure()
#     ax = Axis(fig[1, 1])
#     for p in 1:3, m in 1:9
#         lines!(ax, df._time, df[:, "battery_sensor_temperature_$(p)_$(m)_1"])
#     end
#     ax.ylabel = "Battery temperature / °C"
#     fig
# end


function plot_heat_generation(data)
    df_i = data[:module_current]
    df_T = data[:battery_temperature]
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]
    for p in 1:3, m in 1:9
        lines!(ax[1], df_T._time, df_T[:, "battery_sensor_temperature_$(p)_$(m)_1"])

        r = 1.0e-3 # 1 mΩ
        Qloss = cumsum(r * df_i[:, "module_average_current_$(p)_$(m)"] .^ 2)
        lines!(ax[2], df_i._time, Qloss)
    end
    ax[1].ylabel = "Battery temperature / °C"
    ax[2].ylabel = "Pseudo thermal loss / J"
    linkxaxes!(ax...)
    return fig
end


# let df_i = data[:module_current]
#     fig = Figure(size=(1400, 800))
#     ax = [Axis(fig[i, j]) for i in 1:9, j in 1:3]

#     phases = 1:3
#     modules = 1:9

#     for p in phases, m in modules

#         for i in 1:12
#             dt = [Dates.value.(diff(df_i._time)); 0] * 1e-3 / 3600
#             Q = -cumsum(df_i[:, "module_average_current_$(p)_$(m)"] .* dt)
#             lines!(ax[m, p], df_i._time, Q)
#         end
#     end

#     for i in modules, j in phases[2:end]
#         hideydecorations!(ax[i, j], grid=false, ticks=false)
#     end
#     for i in modules[1:end-1], j in phases
#         hidexdecorations!(ax[i, j], grid=false, ticks=false)
#     end

#     for i in modules, j in phases
#         ylims!(ax[i, j], -65, 5)
#     end
#     fig |> display
# end


# alignemt voltage current
function plot_voltage_current_alignment(data, p, m, c)
    df_i = select(data[:module_current], "_time" => "time", "module_average_current_$(p)_$(m)" => "value")
    # df_v = select(data[:module_voltage], "_time" => "time", "module_voltage_$(p)_$(m)" => "value")
    df_v = select(data[:cell_voltage], "_time" => "time", "cell_voltage_$(p)_$(m)_1_$(c)" => "value")

    Δv = abs.(diff(df_v.value))
    idx = findall(>(0.01), Δv) .+ 1  # +1 because diff() reduces length by 1

    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]

    t_v = Dates.value.(df_v.time - first(ti)) * 1.0e-3
    t_i = Dates.value.(df_i.time - first(ti)) * 1.0e-3

    scatterlines!(ax[1], t_v, df_v.value)
    scatterlines!(ax[2], t_i, df_i.value)

    vlines!(ax[1], t_v[idx], color = (:gray, 0.5), linestyle = :dash)
    vlines!(ax[2], t_v[idx], color = (:gray, 0.5), linestyle = :dash)

    linkxaxes!(ax...)

    return fig
end

# All tables share a single dense `_time` column, so the sampling intervals are a
# property of each signal table, not of individual modules.
function plot_data_resolution(data; yscale = log10)
    colors = Makie.wong_colors()
    # signal → color mapping matches plot_dataset_overview
    signals = [
        (:module_current, "Module current", colors[2]),
        (:module_voltage, "Module voltage", colors[3]),
        (:battery_temperature, "Module temperature", colors[4]),
        (:cell_voltage, "Cell voltage", colors[1]),
    ]
    logscale = yscale === log10

    fig = Figure(size = (550, 500))
    ax = [
        Axis(fig[i, 1]; yscale, ylabel = "Count", titlealign = :left, titlesize = 12,
            xticks = 0:10:80, xminorticks = IntervalsBetween(10), xminorticksvisible = true)
            for i in 1:4
    ]
    bins = 0.5:1:80.5  # integer-second timestamps, center bars on integers

    nmax = 0
    for (i, (key, name, color)) in enumerate(signals)
        Δt = Dates.value.(diff(data[key][!, "_time"])) * 1.0e-3
        hist!(ax[i], Δt; strokewidth = 1, strokecolor = :black, color, bins,
            fillto = logscale ? 0.5 : 0.0)
        nmax = max(nmax, maximum(StatsBase.fit(Histogram, Δt, bins).weights))
        ax[i].title = "$name  (median $(round(Int, median(Δt))) s)"
    end

    linkaxes!(ax...)
    xlims!(ax[1], 0, 81)
    logscale && ylims!(ax[1], 0.5, 2nmax)
    foreach(a -> hidexdecorations!(a; grid = false, ticks = false, minorticks = false), ax[1:3])
    ax[4].xlabel = "Sampling interval / s"

    return fig
end


function plot_module_dataset(data, p, m)
    fig = Figure(size = (550, 500))
    colors = Makie.wong_colors()
    ax = [Axis(fig[i, 1]) for i in 1:4]

    # cell voltage
    df_v = copy(data[:cell_voltage])
    t0 = first(df_v._time)
    df_v[!, :t] = Dates.value.(df_v._time .- t0) * 1.0e-3 # time in seconds

    # module voltage
    df_V = select(data[:module_voltage], "_time" => "time", "module_voltage_$(p)_$(m)" => "value")
    df_V[!, :t] = Dates.value.(df_V.time .- t0) * 1.0e-3 # time in seconds

    # current
    df_i = select(data[:module_current], "_time" => "time", "module_average_current_$(p)_$(m)" => ByRow(x -> -x) => "value")
    df_i[!, :t] = Dates.value.(df_i.time .- t0) * 1.0e-3 # time in seconds

    # temperature
    df_T = select(data[:battery_temperature], "_time" => "time", "battery_sensor_temperature_$(p)_$(m)_1" => "value")
    df_T[!, :t] = Dates.value.(df_T.time .- t0) * 1.0e-3 # time in seconds

    # charge throughput (coulomb counting)
    dt = [Dates.value.(diff(df_i.time)) * 1.0e-3; 0]
    q = cumsum(df_i.value .* dt) / 3600

    N = 5

    for i in 1:12
        lines!(ax[1], df_v[:, :t] / 3600, df_v[:, "cell_voltage_$(p)_$(m)_1_$(i)"])
    end
    lines!(ax[2], df_V[1:N:end, :t] / 3600, df_V[1:N:end, :value], color = colors[3])
    lines!(ax[3], df_i[1:N:end, :t] / 3600, df_i[1:N:end, :value], color = colors[2])
    lines!(ax[4], df_T[1:N:end, :t] / 3600, df_T[1:N:end, :value], color = colors[4])
    # lines!(ax[4], df_i.t / 3600, q, color=colors[3])

    for i in 1:4
        xlims!(ax[i], df_v[begin, :t] / 3600, df_v[end, :t] / 3600)
    end
    for i in 1:3
        hidexdecorations!(ax[i], ticks = false, grid = false)
    end

    ylims!(ax[1], 3.4, 4.2)
    ax[1].yticks = 3.4:0.4:4.2
    ylims!(ax[2], 42, 50)
    ax[2].yticks = 42:4:50
    ylims!(ax[3], -50, 50)
    ax[3].yticks = -50:50:50
    ylims!(ax[4], 18, 33)
    ax[4].yticks = 20:5:30
    # ax[1].yticks = 3.4:0.2:4.2

    ax[1].title = "Exemplary module dataset: Phase $p, Module $m"
    ax[1].ylabel = "Cell\nvoltages (V)"
    ax[2].ylabel = "Module\nvoltage (V)"
    ax[3].ylabel = "Module\ncurrent (A)"
    ax[4].ylabel = "Module\ntemperature (°C)"
    # ax[4].ylabel = "Coloumb \n Counting / Ah"
    ax[4].xlabel = "Time (h)"

    for i in 1:4
        ax[i].xgridvisible = false
        ax[i].ygridvisible = false
        ax[i].xminorticks = IntervalsBetween(5)
        ax[i].xminorticksvisible = true
        # ax[i].xminorgridvisible = true
        ax[i].yminorticks = IntervalsBetween(2)
        ax[i].yminorticksvisible = true
        # ax[i].yminorgridvisible = true
    end

    # Label(fig[0, :], "Exemplary module dataset",
    #     font=:bold, tellwidth=false,
    # )
    rowgap!(fig.layout, 10)

    return fig
end


function plot_dataset_overview(data; id_norm = (3, 7), id_out = (3, 5))
    fig = Figure(size = (700, 450))
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

        axs[1, j].title = "Phase $p, Module $m" * (j == 2 ? " (outlier)" : "")
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


function calc_throughput(data, p, m)
    df_i = select(data[:module_current], "_time" => "time", "module_average_current_$(p)_$(m)" => "value")
    dt = [Dates.value.(diff(df_i.time)) * 1.0e-3; 0]
    return sum(abs, df_i.value .* dt) / 3600
end
