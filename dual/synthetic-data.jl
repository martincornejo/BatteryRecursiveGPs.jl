using OrdinaryDiffEq
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

using DataFrames
using CSV
using DataInterpolations

using StatsBase
using StaticArrays

using LinearAlgebra

using CairoMakie

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
    # s = StatsBase.transform(zt.s, df.s)
    return DataFrame(; df.t, v, i, df.s)
end

@mtkmodel ECM begin
    @parameters begin
        Q = 4.8
        # R1 = 15e-3
        # τ1 = 60.0
        # R2 = 15e-3
        # τ2 = 600.0
    end
    @structural_parameters begin
        focv
        fR0
        fi
    end
    @variables begin
        i(t), [input = true]
        v(t), [output = true]
        vr(t)
        # v1(t) = 0.0
        # v2(t) = 0.0
        ocv(t)
        R0(t)
        soc(t)
    end
    @equations begin
        # param and profile lookups
        ocv ~ focv(soc)
        R0 ~ fR0(soc)
        i ~ fi(t)

        # system
        D(soc) ~ i / (Q * 3600.0)
        # D(v1) ~ -v1 / τ1 + i * (R1 / τ1)
        # D(v2) ~ -v2 / τ2 + i * (R2 / τ2)
        vr ~ i * R0
        v ~ ocv + vr # + v1
    end
end

begin # read OCV look-up-table and current profile
    df_ocv = CSV.File("data/ocv.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc, extrapolation=ExtrapolationType.Constant)

    df = CSV.File("data/profile.csv") |> DataFrame
    fi = ConstantInterpolation(df.i, df.t)

    fR0(s) = 0.01 + 0.005 * s + 0.005 * sinpi(-0.2 + s * 1.5)
end

begin # create model
    @mtkbuild ecm = ECM(; focv, fi, fR0)
    tspan = (0, 24 * 3600) # one day
    ode = ODEProblem{false}(ecm, [ecm.soc => 0.5], tspan)
end

let # create synthetic data
    Ts = 1.0 # time sampling
    sol = solve(ode, Tsit5(); saveat=Ts)

    v = sol[ecm.v]
    s = sol[ecm.soc]
    i = sol[ecm.i]

    # plot
    color = Makie.wong_colors()
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:3]
    lines!(ax[1], sol.t / 3600, i, color=color[1])
    lines!(ax[2], sol.t / 3600, s, color=color[2])
    lines!(ax[3], sol.t / 3600, v, color=color[3])

    # ax[1].ylabel = "i"
    # ax[2].ylabel = "s"
    # ax[3].ylabel = "v"

    fig
end


begin
    Ts = 1.0 # time sampling
    sol = solve(ode, Tsit5(); saveat=Ts)

    v = sol[ecm.v]
    s = sol[ecm.soc]
    i = sol[ecm.i]

    df = DataFrame(
        t=sol.t,
        v=sol[ecm.v],
        s=sol[ecm.soc],
        i=sol[ecm.i],
    )
    zt = fit_zscore(df)
    dfn = normalize_data(df, zt)

    ys = [SA[y] for y in dfn.v]
    us = [(; s=x.s, i=x.i) for x in eachrow(dfn)]
    # CSV.write("output/simulated_data.csv", df_out)
end


