


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
    fig
end

for p in 1:3, m in 1:9
    fig = plot_cell_voltage(data, p, m, 1:12)
    fig |> display
end

function plot_cell_voltage_system(data)
    df = data[:cell_voltage]
    fig = Figure(size=(1400, 800))
    ax = [Axis(fig[i, j]) for i in 1:9, j in 1:3]

    phases = 1:3
    modules = 1:9

    for p in phases, m in modules

        for i in 1:12
            lines!(ax[m, p], df._time, df[:, "cell_voltage_$(p)_$(m)_1_$(i)"])
        end
        # ax[m, p].title = "Phase: $(p), Module: $(m)" #, Cell: $(i)"
    end

    for i in modules, j in phases[2:end]
        hideydecorations!(ax[i, j], grid=false, ticks=false)
    end
    for i in modules[1:end-1], j in phases
        hidexdecorations!(ax[i, j], grid=false, ticks=false)
    end

    for i in modules, j in phases
        ylims!(ax[i, j], 3.35, 4.1)
    end
    fig |> display
end

# let df = data[:module_current]
#     for p in 1:3, m in 1:9
#         fig, ax = lines(df._time, df[:, "module_average_current_$(p)_$(m)"])
#         ax.title = "Phase: $(p), Module: $(m)"
#         fig |> display
#     end
# end

let df = data[:derating_current]
    df = coalesce.(df, NaN)
    # for p in 1:3, m in 1:9
    p = 3
    m = 5
    fig = Figure()
    ax = Axis(fig[1, 1])
    # df_ch = dropmissing(df, "dr_ch_p$(p)_m$(m)")
    # df_dc = dropmissing(df, "dr_ch_p$(p)_m$(m)")
    lines!(ax, df._time, df[:, "dr_ch_p$(p)_m$(m)"])
    lines!(ax, df._time, -df[:, "dr_dch_p$(p)_m$(m)"])
    ax.title = "Phase: $(p), Module: $(m)"
    fig |> display
    # end
end

function plot_derating_phase(data)
    # df = dropmissing(data[:derating_current])
    df = coalesce.(data[:derating_current], NaN)
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:3]

    for p in 1:3
        for m in 1:9
            lines!(ax[p], df._time, df[:, "dr_ch_p$(p)_m$(m)"]; color=Cycled(m))
            lines!(ax[p], df._time, -df[:, "dr_dch_p$(p)_m$(m)"]; color=Cycled(m))
        end
    end

    hidexdecorations!(ax[1], ticks=false, grid=false)
    hidexdecorations!(ax[2], ticks=false, grid=false)
    linkxaxes!(ax...)
    fig
end



function plot_derating_state(data, p, m)
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]

    ## cell voltage
    ids = ["cell_voltage_$(p)_$(m)_1_$(i)" for i in 1:12]
    df_v = select(data[:cell_voltage],
        "_time",
        AsTable(ids) => ByRow(minimum) => :min_voltage,
        AsTable(ids) => ByRow(maximum) => :max_voltage,
    )
    lines!(ax[1], df_v._time, df_v.min_voltage)
    lines!(ax[1], df_v._time, df_v.max_voltage)
    hlines!(ax[1], [3.4, 4.05]; color=:gray, linestyle=:dash)

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
    fig
end

# for p in 1:3, m in 1:9
#     plot_derating_state(data, p, m) |> display
# end

function plot_current_sensor_error(data; fillmissing=false)

    if fillmissing
        df_i = ffill(data[:module_current], Second(1))
        select!(df_i, "_time" => "time", "module_average_current_1_9" => "i")
    else
        df_i = select(data[:module_current], "_time" => "time", "module_average_current_1_9" => "i")
    end

    df_î = CSV.File(datadir * "oscilloscope_p1_m9.csv"; dateformat=dateformat"y-m-dTH:M:S.sss+00:00") |> DataFrame

    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:3]

    lines!(ax[1], df_i.time, df_i.i)
    lines!(ax[1], df_î.timestamp_utc, df_î.MEAS1)
    # lines!(ax[1], df_î.timestamp_utc, df_î.MEAS3)


    ## Current error
    t0 = first(df_î.timestamp_utc)
    fi = ConstantInterpolation(df_î.MEAS1, Dates.value.(df_î.timestamp_utc - t0) * 1e-3)

    ts = Dates.value.(df_i.time - t0) * 1e-3
    idx = findall(>(0.0), ts)
    i2 = fi(ts[idx])

    Δi = df_i[idx, :i] - i2
    lines!(ax[2], df_i[idx, :time], Δi)


    ## Coloumb counting error
    dt1 = [Dates.value.(diff(df_i.time)) * 1e-3; 0]
    cc1 = -cumsum(df_i.i .* dt1) / 3600

    dt2 = [Dates.value.(diff(df_î.timestamp_utc)) * 1e-3; 0]
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

    fig
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

        r = 1e-3 # 1 mΩ
        Qloss = cumsum(r * df_i[:, "module_average_current_$(p)_$(m)"] .^ 2)
        lines!(ax[2], df_i._time, Qloss)
    end
    ax[1].ylabel = "Battery temperature / °C"
    ax[2].ylabel = "Pseudo thermal loss / J"
    linkxaxes!(ax...)
    fig
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

    t_v = Dates.value.(df_v.time - first(ti)) * 1e-3
    t_i = Dates.value.(df_i.time - first(ti)) * 1e-3

    scatterlines!(ax[1], t_v, df_v.value)
    scatterlines!(ax[2], t_i, df_i.value)

    vlines!(ax[1], t_v[idx], color=(:gray, 0.5), linestyle=:dash)
    vlines!(ax[2], t_v[idx], color=(:gray, 0.5), linestyle=:dash)

    linkxaxes!(ax...)

    fig
end

function plot_data_resolution(data, p, m; yscale=identity)
    # p = 1, m = 9, yscale = log10 # identity # log10
    df_i = select(data[:module_current], "_time" => "time", "module_average_current_$(p)_$(m)" => "value")
    df_v = select(data[:module_voltage], "_time" => "time", "module_voltage_$(p)_$(m)" => "value")
    df_c = select(data[:cell_voltage], "_time" => "time", "cell_voltage_$(p)_$(m)_1_1" => "value")

    fig = Figure()
    ax = [Axis(fig[i, 1]; yscale) for i in 1:3]
    bins = 1:80

    hist!(ax[1], Dates.value.(diff(df_i.time)) * 1e-3; strokewidth=1, strokecolor=:black, bins)
    hist!(ax[2], Dates.value.(diff(df_v.time)) * 1e-3; strokewidth=1, strokecolor=:black, bins)
    hist!(ax[3], Dates.value.(diff(df_c.time)) * 1e-3; strokewidth=1, strokecolor=:black, bins)


    xlims!(ax[1], 0, 80)
    xlims!(ax[2], 0, 80)
    xlims!(ax[3], 0, 80)


    ax[1].title = "Measurement time resolution"
    ax[1].ylabel = "Module current"
    ax[2].ylabel = "Module voltage"
    ax[3].ylabel = "Cell voltage"
    ax[3].xlabel = "Time step / s"

    fig
end


function plot_module_dataset(data, p, m)
    fig = Figure()
    colors = Makie.wong_colors()
    ax = [Axis(fig[i, 1]) for i in 1:4]

    df_v = data[:cell_voltage]
    df_i = select(data[:module_current], "_time" => "time", "module_average_current_$(p)_$(m)" => ByRow(x -> -x) => "value")
    df_V = select(data[:module_voltage], "_time" => "time", "module_voltage_$(p)_$(m)" => "value")
    df_T = select(data[:battery_temperature], "_time" => "time", "battery_sensor_temperature_$(p)_$(m)_1" => "value")

    dt = [Dates.value.(diff(df_i.time)) * 1e-3; 0]
    q = cumsum(df_i.value .* dt) / 3600

    for i in 1:12
        lines!(ax[1], df_v._time, df_v[:, "cell_voltage_$(p)_$(m)_1_$(i)"])
    end
    lines!(ax[2], df_V.time, df_V.value, color=colors[1])
    lines!(ax[3], df_i.time, df_i.value, color=colors[2])
    lines!(ax[3], df_i.time, df_i.value, color=colors[2])
    # lines!(ax[4], df_T.time, df_T.value, color=colors[3])
    lines!(ax[4], df_i.time, q, color=colors[3])

    # for i in 1:4
    #     xlims!(ax[i], df[begin, :t] / 3600, df[end, :t] / 3600)
    # end
    for i in 1:3
        hidexdecorations!(ax[i], ticks=false, grid=false)
    end

    ax[1].ylabel = "Cell / V"
    ax[2].ylabel = "Module / V"
    ax[3].ylabel = "Current / A"
    # ax[4].ylabel = "Temperature / °C"
    ax[4].ylabel = "Coloumb Counting / Ah"
    ax[4].xlabel = "Time / h"

    fig
end

for p in 1:3, m in 1:9
    plot_module_dataset(data, p, m) |> display
end

