using RecursiveGPs
using AbstractGPs
using StaticArrays

using Distributions
using LowLevelParticleFilters
import LowLevelParticleFilters as LLPF

using DataFrames
using CSV
using DataFrames
using Random
using DataInterpolations

using StatsBase

using BenchmarkTools

using ForwardDiff
using LinearAlgebra

import ComponentArrays: ComponentVector, ComponentMatrix, getaxes

using CairoMakie

using PreallocationTools

using OrdinaryDiffEq
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

using CairoMakie

include("synthetic-data-T.jl")
include("model-T.jl")

begin
    # read OCV look-up-table and current profile, define internal resistance function
    df_ocv = CSV.File("data/ocv.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc, extrapolation=ExtrapolationType.Constant)

    df = CSV.File("data/yuasa-p1-m1.csv") |> DataFrame
    select!(df, :time => :t, :module_current => :i, :module_temperature => :T)

    fi = ConstantInterpolation(df.i, df.t)
    fT = ConstantInterpolation(df.T, df.t)
    # fT = x -> 25

    kT = 20
    T0 = 25
    fR0(s, T) = (0.001 + 0.0002 * sinpi(-0.2 + s * 1.5)) * exp(kT * (1 / T - 1 / T0))

    # simulate battery operation
    ttest = 8 * 3600
    ttrain = 8 * 3600
    tspan = (0, ttest) # one day
    Ts = 10.0 # time sampling

    df = generate_timeseries_T(; Ts, tspan, fi, focv, fR0, fT, soc0=0.6)
    df[!, :q] = cumsum(df.i) * Ts / 3600
    # df = generate_timeseries_rc(; Ts, tspan, fi, focv, fR0, soc0=0.65)
    plot_timeseries_T(df) |> display

    # create input / output data 
    zt = fit_zscore(df)
    dfn = normalize_data(df, zt)

    df_train = subset(df, :t => ByRow(<=(ttrain)))
    df_train_n = subset(dfn, :t => ByRow(<=(ttrain)))
    df_test = subset(df, :t => ByRow(>(ttrain)))
    df_test_n = subset(dfn, :t => ByRow(>(ttrain)))

    us = [(; i, q) for (i, q) in zip(df_train_n.i, df_train_n.q)]
    ut = [(; i, q) for (i, q) in zip(df_test_n.i, df_test_n.q)]
    ys = [SA[y] for y in df_train_n.v]
end;

let fig = Figure()
    ax = Axis(fig[1, 1])
    soc = 0:0.01:1
    for T in [20, 25, 30]
        lines!(ax, soc, fR0.(soc, T), label="T = $T °C")
    end
    ax.ylabel = "Resistance / Ω"
    ax.ylabel = "SOC / p.u."
    fig
end

kf = build_kf(dfn);
# @time run_kf!(kf, us, ys)

for (u, y) in zip(us, ys)
    LLPF.update!(kf, u, y)
end



let fig = Figure(size=(600, 600))
    colors = Makie.wong_colors()
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / mΩ"


    # predict new points -> mean and std
    qmin, qmax = extrema(df.q)
    q = qmin:0.01:qmax
    q̂ = StatsBase.transform(zt.q, q)
    ocv = predict_gp(kf, q̂, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
    ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)

    s = q / 80 .+ 0.6

    # plot results 
    lines!(ax[1], q, ocvμ)
    band!(ax[1], q, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha=0.8, label="GP")
    lines!(ax[1], q, focv.(s), color=:black, linestyle=:dot, label="Real")
    axislegend(ax[1]; merge=true, position=:lt)

    # predict new points -> mean and std
    r0 = predict_gp(kf, q̂, :r0)
    rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1e3
    rσ = StatsBase.reconstruct(zt.r, r0.σ) * 1e3

    # plot results 
    lines!(ax[2], q, rμ)
    band!(ax[2], q, rμ + 2rσ, rμ - 2rσ, alpha=0.8)
    lines!(ax[2], q, fR0.(s, 25) * 1e3, color=:black, linestyle=:dot)

    xlims!(ax[1], qmin, qmax)
    xlims!(ax[2], qmin, qmax)

    fig
end

let fig = Figure()
    colors = Makie.wong_colors()
    ax = [Axis(fig[i, 1]) for i in 1:2]
    μ, σ = predict(kf, df, zt)
    lines!(ax[1], df.t / 3600, μ)
    band!(ax[1], df.t / 3600, μ - 2σ, μ + 2σ, alpha=0.5)
    lines!(ax[1], df.t / 3600, df.v)

    Δv = μ - df.v
    lines!(ax[2], df.t / 3600, Δv, color=colors[3])
    band!(ax[2], df.t / 3600, Δv - 2σ, Δv + 2σ, color=(colors[3], 0.5))
    fig
end

