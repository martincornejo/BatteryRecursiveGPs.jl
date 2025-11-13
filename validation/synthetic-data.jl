
function fit_zscore(df)
    v = StatsBase.fit(ZScoreTransform, df.v)
    σ = StatsBase.fit(ZScoreTransform, df.v, center=false)
    i = StatsBase.fit(ZScoreTransform, df.i, center=false)
    s = StatsBase.fit(ZScoreTransform, df.s)
    r = ZScoreTransform(1, 1, [0.0], [σ.scale[1] / i.scale[1]])
    return (; v, σ, i, s, r)
end

function normalize_data(df, zt)
    v = StatsBase.transform(zt.v, df.v)
    i = StatsBase.transform(zt.i, df.i)
    # s = StatsBase.transform(zt.s, df.s) # do not normalize soc
    return DataFrame(; df.t, v, i, df.s)
end

@component function ECM(; name, fi, focv, fR0)
    params = @parameters begin
        Q = 4.8 # Ah
        # R0 = 15e-3 # Ohm
    end
    vars = @variables begin
        i(t), [input = true]
        v(t), [output = true]
        vr0(t)
        ocv(t)
        R0(t)
        soc(t)
    end
    eqs = [
        # param and profile lookups
        ocv ~ focv(soc),
        R0 ~ fR0(soc),
        i ~ fi(t),

        # system
        D(soc) ~ i / (Q * 3600.0),
        vr0 ~ i * R0,
        # D(vrc1) ~ -vrc1 / τ1 + i * (R1 / τ1)
        v ~ ocv + vr0,
    ]

    System(eqs, t, vars, params; name)
end

function generate_timeseries(; tspan, focv, fi, fR0, Ts=1.0)
    @mtkcompile ecm = ECM(; focv, fi, fR0)
    ode = ODEProblem{false}(ecm, [ecm.soc => 0.5], tspan)
    sol = solve(ode, Tsit5(); saveat=Ts)

    DataFrame(
        t=sol.t,
        v=sol[ecm.v],
        s=sol[ecm.soc],
        i=sol[ecm.i],
    )
end

function plot_timeseries(df)
    fig = Figure()
    color = Makie.wong_colors()
    ax = [Axis(fig[i, 1]) for i in 1:3]
    lines!(ax[1], df.t / 3600, df.i, color=color[1])
    lines!(ax[2], df.t / 3600, df.s, color=color[2])
    lines!(ax[3], df.t / 3600, df.v, color=color[3])

    ax[1].ylabel = "Current / A"
    ax[2].ylabel = "SOC / p.u."
    ax[3].ylabel = "Voltage / V"
    ax[3].xlabel = "Time / h"

    hidexdecorations!(ax[1]; ticks=false, grid=false)
    hidexdecorations!(ax[2]; ticks=false, grid=false)
    xlims!(ax[1], df[begin, :t] / 3600, df[end, :t] / 3600)
    xlims!(ax[2], df[begin, :t] / 3600, df[end, :t] / 3600)
    xlims!(ax[3], df[begin, :t] / 3600, df[end, :t] / 3600)
    linkxaxes!(ax...)
    fig
end
