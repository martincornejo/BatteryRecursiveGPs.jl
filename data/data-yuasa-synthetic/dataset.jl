using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks
using OrdinaryDiffEq

using Dates
using CSV, DataFrames, DataInterpolations
using JSON

"""
First order equivalent circuit model (ECM) 
"""
@component function ECM(; name)
    @named oneport = OnePort()

    params = @parameters begin
        Q, [description = "Cell capacity in Ah"]
        R1, [description = "RC resistance in Ohm"]
        tau1, [description = "RC time constant in s"]
        fR0(::Real, ::Real)::Real, [description = "Internal resistance as a function of SOC and temperature in Ohm"]
        focv(::Real)::Real, [description = "Open circuit voltage as a function of SOC in V"]
    end

    vars = @variables begin
        v(t), [description = "Terminal voltage in V"]
        i(t), [description = "Current in A"]
        v1(t) = 0, [description = "RC voltage in V"]
        vr(t), [description = "Voltage drop across internal resistance in V"]
        r0(t), [description = "Internal resistance in Ohm"]
        ocv(t), [description = "Open circuit voltage in V"]
        soc(t), [description = "State of charge in p.u."]
        T(t), [description = "Temperature in °C"]
    end

    eqs = [
        ocv ~ focv(soc)
        r0 ~ fR0(soc, T)
        D(soc) ~ i / (Q * 3600.0)
        D(v1) ~ -v1 / tau1 + i * (R1 / tau1)
        vr ~ i * r0
        v ~ ocv + vr + v1
    ]

    extend(System(eqs, t, vars, params; name), oneport)
end

"""
Battery module consisting of s cells in series
"""
@component function BatteryModule(s; name)

    # TODO: system derating

    params = @parameters begin
        fi(::Real)::Real
        fT(::Real)::Real
    end

    vars = @variables begin
        T(t) # temperature
    end

    systems = @named begin
        cell[1:s] = ECM()
        current = Current()
        voltage = VoltageSensor()
        ground = Ground()
    end

    eqs = [
        connect(current.p, voltage.p, cell[1].p)
        [connect(cell[i].n, cell[i+1].p) for i in 1:s-1]
        connect(current.n, voltage.n, cell[end].n, ground.g)
        current.i ~ fi(t)
        T ~ fT(t)
        [cell[i].T ~ T for i in 1:s]
    ]

    System(eqs, t; systems, name)
end

function get_var(sys, var::Symbol)
    getproperty(sys, var)
end

function get_var(sys, var::String)
    getproperty(sys, Symbol(var))
end

function get_var(sys, var, vars...)
    next = get_var(sys, var)
    isempty(vars) ? next : get_var(next, vars...)
end

function simulate_module(params, fi, fT, tspan; Ts=1.0)
    # create model
    n = length(params)
    @mtkcompile pack = BatteryModule(n)

    # initial condition
    u0 = Pair[
        pack.fi=>fi,
        pack.fT=>fT,
    ]
    for (cell, p) in params
        for (var, val) in p
            push!(u0, get_var(pack, cell, var) => val)
        end
    end

    # simulate
    prob = ODEProblem(pack, u0, tspan)
    sol = solve(prob, Vern7(); saveat=Ts, reltol=1e-10)

    return DataFrame(
        "t" => sol.t,
        "v" => sol[pack.voltage.v],
        "i" => -sol[pack.current.i],
        "q" => -cumsum(sol[pack.current.i]) * Ts / 3600,
        "T" => sol[pack.T],
        ["v_cell_$i" => sol[get_var(pack, "cell_$i", "v")] for i in 1:n]...,
        ["soc_cell_$i" => sol[get_var(pack, "cell_$i", "soc")] for i in 1:n]...,
    )
end

function plot_dataset(df)
    fig = Figure(size=(600, 600))
    colors = Makie.wong_colors()
    ax = [Axis(fig[i, 1]) for i in 1:5]
    for i in 1:12
        lines!(ax[1], df.t / 3600, df[:, "v_cell_$i"])
    end
    lines!(ax[2], df.t / 3600, df.v, color=colors[1])
    lines!(ax[3], df.t / 3600, df.î, color=colors[6])
    lines!(ax[3], df.t / 3600, df.i, color=colors[2])
    lines!(ax[4], df.t / 3600, df.q, color=colors[3])
    lines!(ax[5], df.t / 3600, df.T, color=colors[4])

    for i in 1:5
        xlims!(ax[i], df[begin, :t] / 3600, df[end, :t] / 3600)
    end
    for i in 1:4
        hidexdecorations!(ax[i], ticks=false, grid=false)
    end

    ylims!(ax[5], 18, 28)

    ax[1].ylabel = "Cell / V"
    ax[2].ylabel = "Module / V"
    ax[3].ylabel = "Current / A"
    ax[4].ylabel = "Coloumb / Ah"
    ax[5].ylabel = "Temperature / °C"
    ax[5].xlabel = "Time / h"

    linkxaxes!(ax...)
    fig
end

function plot_current_profiles(df)
    fig = Figure()
    ax = [Axis(fig[i, j]) for i in 1:2, j in 1:2]
    lines!(ax[1, 1], df.t / 3600, df.i, label="OSCI")
    lines!(ax[1, 1], df.t / 3600, df.î, label="CMU")
    lines!(ax[2, 1], df.t / 3600, df.i - df.î, label="CMU")

    Ts = 1.0
    q = cumsum(df.i) * Ts / 3600
    q̂ = cumsum(df.î) * Ts / 3600
    lines!(ax[1, 2], df.t / 3600, q, label="OSCI")
    lines!(ax[1, 2], df.t / 3600, q̂, label="CMU")
    lines!(ax[2, 2], df.t / 3600, (q - q̂))

    ax[1, 1].ylabel = "Current / A"
    ax[2, 1].ylabel = "Current error / A"
    ax[1, 2].ylabel = "Coloumb / Ah"
    ax[2, 2].ylabel = "Coloumb Error / Ah"
    ax[2, 1].xlabel = "Time / h"
    ax[2, 2].xlabel = "Time / h"

    linkxaxes!(ax...)
    fig
end

function read_profiles(datadir)
    p = 1 # phase
    m = 9 # module

    t0 = DateTime("2025-12-10T14:00:20")
    dateformat = dateformat"y-m-d H:M:S+00:00"

    # battery module averave current
    df_i = CSV.File(datadir * "module_current_average.csv"; dateformat) |> DataFrame
    select!(df_i, :_time => :time, "module_average_current_$(p)_$(m)" => :i)
    transform!(df_i, :time => ByRow(t -> Dates.value(t - t0) / 1000) => :t)
    fi = ConstantInterpolation(df_i.i, df_i.t)

    # battery module averae current from oscilloscope
    df_o = CSV.File(datadir * "oscilloscope_p1_m9.csv"; dateformat=dateformat"y-m-dTH:M:S.sss+00:00") |> DataFrame
    select!(df_o, :timestamp_utc => :time, "MEAS1" => :i)
    transform!(df_o, :time => ByRow(t -> Dates.value(t - t0) / 1000) => :t)
    fo = ConstantInterpolation(df_o.i, df_o.t)

    # battery module temperature
    df_T = CSV.File(datadir * "battery_temperature.csv"; dateformat) |> DataFrame
    select!(df_T, :_time => :time, "battery_sensor_temperature_$(p)_$(m)_1" => :T)
    transform!(df_T, :time => ByRow(t -> Dates.value(t - t0) / 1000) => :t)
    fT = ConstantInterpolation(df_T.T, df_T.t)

    # battery module voltage
    df_v = CSV.File(datadir * "module_voltage.csv"; dateformat) |> DataFrame
    select!(df_v, :_time => :time, "module_voltage_$(p)_$(m)" => :v)
    transform!(df_v, :time => ByRow(t -> Dates.value(t - t0) / 1000) => :t)
    fv = ConstantInterpolation(df_v.v, df_v.t)

    (; fi, fi_oscilloscope=fo, fT, fv)
end

function plot_voltage_comparison(df, fv) # similar voltage ranges
    (; t, v) = df
    v_real = fv(t)

    real = extrema(v_real)
    synthetic = extrema(v)
    @info "Voltage ranges" real synthetic

    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, t / 3600, v_real, label="Real")
    lines!(ax, t / 3600, v, label="Synthetic")
    ax.ylabel = "Voltage / V"
    ax.xlabel = "Time / h"
    axislegend(ax, position=:rb)
    fig
end

function plot_socs(df) # cell socs in boundaries
    fig = Figure()
    ax = Axis(fig[1, 1])
    for i in 1:12
        lines!(ax, df.t / 3600, df[:, "soc_cell_$i"])
    end
    ax.ylabel = "SOC / p.u."
    ax.xlabel = "Time / h"
    fig
end


begin # function main()
    # read OCV look-up-table 
    df_ocv = CSV.File("data/ocv/ocv-yuasa.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc, extrapolation=ExtrapolationType.Constant)

    # internal resistance function
    fR0(s, T; kT=20, T0=25) = (0.0015 + 0.0004 * sinpi(-0.2 + s * 1.5)) * exp(kT * (1 / T - 1 / T0))

    # read current profile
    (; fi, fi_oscilloscope, fT, fv) = read_profiles("data/data-yuasa-cycles-2/")

    params = Dict(
        # :cell_1 => Dict(:soc => 0.5, :Q => 38, :R0 => 1.5e-3, :R1 => 0.8e-3, :tau1 => 60, :focv => focv),
        :cell_1 => Dict(:soc => 0.82, :Q => 89, :R1 => 1.2e-3, :tau1 => 300, :focv => focv, :fR0 => fR0),
        :cell_2 => Dict(:soc => 0.80, :Q => 90, :R1 => 1.2e-3, :tau1 => 300, :focv => focv, :fR0 => fR0),
        :cell_3 => Dict(:soc => 0.78, :Q => 91, :R1 => 1.2e-3, :tau1 => 300, :focv => focv, :fR0 => fR0),
        :cell_4 => Dict(:soc => 0.87, :Q => 90, :R1 => 1.2e-3, :tau1 => 300, :focv => focv, :fR0 => fR0),
        :cell_5 => Dict(:soc => 0.87, :Q => 92, :R1 => 1.2e-3, :tau1 => 300, :focv => focv, :fR0 => fR0),
        :cell_6 => Dict(:soc => 0.86, :Q => 91, :R1 => 1.2e-3, :tau1 => 300, :focv => focv, :fR0 => fR0),
        :cell_7 => Dict(:soc => 0.86, :Q => 91, :R1 => 1.2e-3, :tau1 => 300, :focv => focv, :fR0 => fR0),
        :cell_8 => Dict(:soc => 0.86, :Q => 92, :R1 => 1.2e-3, :tau1 => 300, :focv => focv, :fR0 => fR0),
        :cell_9 => Dict(:soc => 0.87, :Q => 90, :R1 => 1.2e-3, :tau1 => 300, :focv => focv, :fR0 => fR0),
        :cell_10 => Dict(:soc => 0.87, :Q => 91, :R1 => 1.2e-3, :tau1 => 300, :focv => focv, :fR0 => fR0),
        :cell_11 => Dict(:soc => 0.86, :Q => 92, :R1 => 1.2e-3, :tau1 => 300, :focv => focv, :fR0 => fR0),
        :cell_12 => Dict(:soc => 0.86, :Q => 92, :R1 => 1.2e-3, :tau1 => 300, :focv => focv, :fR0 => fR0)
    )


    # simulate
    tspan = (0, 12.5 * 3600)
    df = simulate_module(params, fi_oscilloscope, fT, tspan)
    df[!, :î] = -fi(df.t)

    # plot
    plot_dataset(df) |> display
    plot_current_profiles(df) |> display
    plot_socs(df) |> display
    plot_voltage_comparison(df, fv) |> display

    # save data
    CSV.write("data/data-yuasa-synthetic/data-yuasa-synthetic.csv", df)
    JSON.json("data/data-yuasa-synthetic/battery-params.json",
        Dict(battery => Dict(key => val for (key, val) in batt_params if key != :focv && key != :fR0)
             for (battery, batt_params) in params); pretty=true
    )

    nothing
end
