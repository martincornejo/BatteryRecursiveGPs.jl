@component function ECM_T(; name, fi, focv, fR0, fT)
    params = @parameters begin
        Q = 80 # Ah
        # R0 = 15e-3 # Ohm
    end
    vars = @variables begin
        i(t), [input = true]
        v(t), [output = true]
        T(t)
        vr0(t)
        ocv(t)
        R0(t)
        soc(t)
    end
    eqs = [
        # param and profile lookups
        ocv ~ focv(soc)
        R0 ~ fR0(soc, T)
        i ~ fi(t)
        T ~ fT(t)

        # system
        D(soc) ~ i / (Q * 3600.0)
        vr0 ~ i * R0
        # D(vrc1) ~ -vrc1 / τ1 + i * (R1 / τ1)
        v ~ ocv + vr0
    ]

    System(eqs, t, vars, params; name)
end

function generate_timeseries_T(; tspan, focv, fi, fR0, fT, soc0=0.5, Ts=1.0)
    @mtkcompile ecm = ECM_T(; focv, fi, fR0, fT)
    ode = ODEProblem{false}(ecm, [ecm.soc => soc0], tspan)
    sol = solve(ode, Vern7(); saveat=Ts, reltol=1e-10)

    DataFrame(
        t=sol.t,
        v=sol[ecm.v],
        s=sol[ecm.soc],
        i=sol[ecm.i],
        T=sol[ecm.T],
    )
end

function plot_timeseries_T(df)
    fig = Figure()
    color = Makie.wong_colors()
    ax = [Axis(fig[i, 1]) for i in 1:4]
    lines!(ax[1], df.t / 3600, df.i, color=color[1])
    lines!(ax[2], df.t / 3600, df.s, color=color[2])
    lines!(ax[3], df.t / 3600, df.v, color=color[3])
    lines!(ax[4], df.t / 3600, df.T, color=color[4])

    ax[1].ylabel = "Current / A"
    ax[2].ylabel = "SOC / p.u."
    ax[3].ylabel = "Voltage / V"
    ax[4].ylabel = "Temperature / °C"
    ax[4].xlabel = "Time / h"

    hidexdecorations!(ax[1]; ticks=false, grid=false)
    hidexdecorations!(ax[2]; ticks=false, grid=false)
    hidexdecorations!(ax[3]; ticks=false, grid=false)
    xlims!(ax[1], df[begin, :t] / 3600, df[end, :t] / 3600)
    xlims!(ax[2], df[begin, :t] / 3600, df[end, :t] / 3600)
    xlims!(ax[3], df[begin, :t] / 3600, df[end, :t] / 3600)
    xlims!(ax[4], df[begin, :t] / 3600, df[end, :t] / 3600)
    linkxaxes!(ax...)
    fig
end

