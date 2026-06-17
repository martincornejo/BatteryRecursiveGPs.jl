let df = data_1, profile = profile_1, t0 = first(ti_1) # power
    fig = Figure()
    ax = Axis(fig[1, 1], ylabel = "L3 Active Power / kW")

    lines!(ax, df.timestamp_utc, coalesce.(df[:, :mb_ActualPowerAcL3] * 1.0e-3, NaN); color = Cycled(2), label = "Actual")

    # t0 = DateTime("2026-05-04T07:05:15")
    # t0 = DateTime("2026-03-27T08:41:13")
    # timestamp = t0 .+ Second.(profile.time_seconds)
    timestamp = t0 .+ Second.(profile.time_seconds) .+ Millisecond.(round.(cumsum(fill(0.135, nrow(profile))) * 1000))
    stairs!(ax, timestamp, profile.power_setpoint_watts * 1.0e-3; color = Cycled(3), label = "Target")

    axislegend(ax, position = :rt)

    fig
end

let data = data_1, profile = profile_1 # reactive power
    fig = Figure(size = (800, 500))
    ax = [Axis(fig[i, 1]) for i in 1:3]

    df = select(
        data,
        :timestamp_utc,
        :mb_ActualPowerAcL3 => :active_power,
        :mb_ReactivePowerAcL3 => :reactive_power,
        :mb_GridVoltageL3 => :grid_voltage,
    )
    subset!(df, :grid_voltage => ByRow(>(200)), skipmissing = true)

    lines!(ax[1], df.timestamp_utc, df.active_power * 1.0e-3; color = Cycled(2))
    lines!(ax[2], df.timestamp_utc, df.reactive_power * 1.0e-3; color = Cycled(3))
    lines!(ax[3], df.timestamp_utc, df.grid_voltage; color = Cycled(4))

    for i in eachindex(ax)
        xlims!(ax[i], df[begin, :timestamp_utc], df[end, :timestamp_utc])
        if i != length(ax)
            hidexdecorations!(ax[i], ticks = false, grid = false)
        end
    end

    ax[1].ylabel = "Active Power / kW"
    ax[2].ylabel = "Active Power / kVar"
    ax[3].ylabel = "Grid Voltage / V"

    fig
end


let # derating currents
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]

    df_sys_ch = select(data, :timestamp_utc, :mb_CurrentLimitChargeSystem => :p_max)
    df_sys_dch = select(data, :timestamp_utc, :mb_CurrentLimitDischargeSystem => :p_max)
    dropmissing!(df_sys_ch)
    dropmissing!(df_sys_dch)

    lines!(ax[1], df_sys_ch.timestamp_utc, -df_sys_ch.p_max)
    lines!(ax[1], df_sys_dch.timestamp_utc, df_sys_dch.p_max)

    for m in 1:9
        df_ch = select(data, "timestamp_utc", "m$(m)_derating_charge_limit" => "i_max")
        df_dch = select(data, "timestamp_utc", "m$(m)_derating_discharge_limit" => "i_max")
        dropmissing!(df_ch)
        dropmissing!(df_dch)
        lines!(ax[2], df_ch.timestamp_utc, -df_ch[:, "i_max"])
        lines!(ax[2], df_dch.timestamp_utc, df_dch[:, "i_max"])
    end

    ax[1].ylabel = "System max. current / A"
    ax[2].ylabel = "Module max. current / A"

    linkxaxes!(ax...)
    fig
end

let # temperatatures plot
    fig = Figure()
    ax = Axis(fig[1, 1], ylabel = "Temperature / °C")
    df_temp1 = select(data, "timestamp_utc", "28-00000024d79c" => "T")
    dropmissing!(df_temp1)
    df_temp1 = subset(df_temp1, "T" => ByRow(>(15)))
    # scatterlines!(ax, df_temp1.timestamp_utc, df_temp1.T; color = :black)


    df_temp2 = select(data, "timestamp_utc", "28-00000024dcda" => "T")
    dropmissing!(df_temp2)
    df_temp2 = subset(df_temp2, "T" => ByRow(>(15)))
    df_temp2 = subset(df_temp2, "T" => ByRow(<(22.5)))
    # scatterlines!(ax, df_temp2.timestamp_utc, df_temp2.T; color = :black)

    for m in 1:9
        for temp in ["temp01", "temp02"]
            df_temp = select(data, "timestamp_utc", "m$(m)_$(temp)" => "T")
            dropmissing!(df_temp)
            lines!(ax, df_temp.timestamp_utc, df_temp.T)
        end
    end

    fig
end

function find_tek_offset(df; m = 7, bms_thresh = 0.1)
    df_both = select(df, :timestamp_utc, "m$(m)_current" => :bms, :tek_m_cur_ref => :tek)
    dropmissing!(df_both)

    # Only use rest periods where BMS reads ~0
    df_rest = subset(df_both, :bms => ByRow(x -> abs(x) < bms_thresh))

    # Offset = mean tek value at rest (so that tek + offset ≈ 0)
    return -mean(df_rest.tek)
end


let data = data_1 # oscilloscope current data
    fig = Figure(size = (600, 600))
    ax = Axis(fig[1, 1])
    df_osci = select(data, "timestamp_utc", "tek_m_cur_ref" => "i")
    dropmissing!(df_osci)
    i_offset = find_tek_offset(data_1; m = 7)
    lines!(ax, df_osci.timestamp_utc, df_osci.i .+ i_offset; label="osci")

    df_i = select(data, "timestamp_utc", "m7_current" => "i")
    dropmissing!(df_i)
    lines!(ax, df_i.timestamp_utc, df_i.i; label="cmu")

    ax2 = Axis(fig[2, 1])
    Δt_osci = [Dates.value.(diff(df_osci.timestamp_utc)); 0] * 1.0e-3
    q_osci = cumsum((df_osci.i .+ i_offset) .* Δt_osci) ./ 3600
    lines!(ax2, df_osci.timestamp_utc, q_osci)

    Δt_i = [Dates.value.(diff(df_i.timestamp_utc)); 0] * 1.0e-3
    q_i = cumsum(df_i.i .* Δt_i) ./ 3600
    lines!(ax2, df_i.timestamp_utc, q_i)

    ax.ylabel = "Current / A"
    ax2.ylabel = "Charge / Ah"

    Legend(fig[3, 1], ax, orientation = :horizontal)
    linkxaxes!(ax, ax2)
    fig
end

let data = data_2 # cell voltages
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], ylabel = "Voltage / V")
    for m in 1:9
        for c in 1:12
            cell = @sprintf("cell%02d", c)
            df_v = select(data, "timestamp_utc", "m$(m)_$(cell)" => "v")
            dropmissing!(df_v)
            lines!(ax, df_v.timestamp_utc, df_v.v)
        end
    end
    fig
end