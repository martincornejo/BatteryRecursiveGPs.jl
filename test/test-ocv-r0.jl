using RecursiveGPs
using AbstractGPs
using StaticArrays

using Distributions
using LowLevelParticleFilters
import LowLevelParticleFilters as LLPF

using CSV
using DataFrames
using Random

using StatsBase

using BenchmarkTools

using ForwardDiff
using LinearAlgebra

import ComponentArrays: ComponentVector, ComponentMatrix, getaxes

using CairoMakie

function run_kf!(kf, us, ys)
    for (u, y) in zip(us, ys)
        LLPF.update!(kf, u, y)
    end
end

function prof_kf(kf, us, ys)
    for i in 1:1000
        run_kf(kf, us, ys)
    end
end

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
    s = StatsBase.transform(zt.s, df.s)
    return DataFrame(; df.t, v, i, s)
end

## === dataset
begin
    df = CSV.read("data/profile.csv", DataFrame)
    zt = fit_zscore(df)
    dfn = normalize_data(df, zt)

    horizon = 24 * 3600
    df_train = dfn[1:horizon, :]
    df_train = dfn[1:horizon, :]

    ys = [SA[y] for y in df_train.v]
    us = [(; s=x.s, i=x.i) for x in eachrow(df_train)]
end

# === model
dynamics(x, u, p, t) = x

function measurement_combined(x, u, p, t)
    (; xid) = p
    xc = ComponentVector(x, xid)
    ocv = measurement_gp(p.ocv, xc.ocv, u.s)
    r0 = measurement_gp(p.r0, xc.r0, u.s)
    ocv + u.i * r0 |> SVector{1}
end

function R2combined(x, u, p, t)
    ocv = uncertainty_gp(p.ocv, u.s)
    r0 = uncertainty_gp(p.r0, u.s)
    ocv + u.i^2 * r0 |> SMatrix{1,1}
end

function Cjac(x, u, p, t)
    (; C) = p.cache
    ForwardDiff.jacobian!(C, x -> measurement_combined(x, u, p, t), x)
    # return Cjac
end

function Ajac(x, u, p, t)
    (; A) = p.cache
    return A
end

function predict(kf, df)
    dfn = normalize_data(df)
    ocv = predict_gp(kf, dfn.s, :ocv)
    r0 = predict_gp(kf, dfn.s, :r0)
    μ = @. ocv.μ + u.i * r0.μ
    σ = @. ocv.σ + u.i^2 * r0.σ
    (; μ, σ)
end

function build_kf(n=21)
    b0 = collect(range(0, 1, n))
    b0n = StatsBase.transform(zt.s, b0)
    r0 = StatsBase.transform(zt.r, [0.03]) |> first

    # kernel1 = LinearKernel() + 0.02 * with_lengthscale(SEKernel(), 0.33)
    kernel1 = 0.02 * with_lengthscale(SEKernel(), 0.33)
    rgp1 = RGP(kernel1, b0n)

    kernel2 = 0.01 * with_lengthscale(SEKernel(), 0.5)
    rgp2 = RGP(r0, kernel2, b0n)

    nx = (length(rgp1.μ0) + (length(rgp2.μ0)))
    p = (; cache=(;
        A=I(nx),
        C=zeros(1, nx),
    ))
    rgps = (; ocv=rgp1, r0=rgp2)

    make_ekf(rgps, dynamics, measurement_combined, R2combined; Ajac, Cjac, p)
end

kf = build_kf()
@time run_kf!(kf, us, ys)

# @benchmark run_kf!($kf, $us, $ys)
# @profview_allocs prof_kf(kf, us, ys)




let fig = Figure(size=(600, 600))
    colors = Makie.wong_colors()
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / Ω"

    # for n in (11, 21, 51, 101)
    n = 21
    kf = build_kf(n)
    t = @timed run_kf!(kf, us, ys)
    (; time, bytes) = t
    memory = 1e-6 * bytes
    @info n time memory

    # predict new points -> mean and std
    smin, smax = df.s |> extrema
    bgp = StatsBase.transform(zt.s, smin:0.01:smax)
    ocv = predict_gp(kf, bgp, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
    ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)


    # plot results 
    # lines!(ax[1], 0:0.01:1, f1.(0:0.01:1), color=colors[1], label="f1(x)")
    lines!(ax[1], bgp, ocvμ, label=string(n))
    band!(ax[1], bgp, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha=0.8, label=string(n))
    # scatter!(ax[1], df_train.s, df.y, color=(:red, 0.5), label="Data")
    axislegend(ax[1]; merge=true, position=:lt)

    # predict new points -> mean and std
    r0 = predict_gp(kf, bgp, :r0)
    rμ = StatsBase.reconstruct(zt.r, r0.μ)
    rσ = StatsBase.reconstruct(zt.r, r0.σ)

    # plot results 
    # lines!(ax[2], 0:0.01:1, f2.(0:0.01:1), color=colors[1], label="f2(x)")
    lines!(ax[2], bgp, rμ)
    band!(ax[2], bgp, rμ + 2rσ, rμ - 2rσ, alpha=0.8)
    # axislegend(ax[2]; merge=true, position=:lt)
    # end
    fig
end

