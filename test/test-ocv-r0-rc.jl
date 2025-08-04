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
include("ecm-mtk.jl")
# begin
#     df = CSV.read("data/profile.csv", DataFrame)
#     zt = fit_zscore(df)
#     dfn = normalize_data(df, zt)

#     horizon = 24 * 3600
#     df_train = dfn[1:horizon, :]
#     df_train = dfn[1:horizon, :]

#     ys = [SA[y] for y in df_train.v]
#     us = [(; s=x.s, i=x.i) for x in eachrow(df_train)]
# end

## == RC
function dynamics_rc(vrc, i, p)
    (; ts, R, τ) = p

    exp(-ts / τ) * vrc + i * R * (1 - exp(-ts / τ))
end

function RC(μ0::Real, Σ0::Real, σ1::Real, σ2::Real, p)
    (; ts, τ) = p
    A = exp(-ts / τ)

    rc = (;
        μ0,
        Σ0,
        R1=σ1,
        R2=σ2,
        A,
        p,
    )
end

function predict_rc()
    # requires initial condition
    nothing
end

# === model
function dynamics_combined(x⁺, x⁻, u, p, t)
    (; xid) = p
    xc⁻ = ComponentVector(x⁻, xid)
    xc⁺ = ComponentVector(x⁺, xid)
    xc⁺ .= xc⁻ # previous values

    xc⁺.rc = dynamics_rc(xc⁻.rc, u.i, p.rc.p)
    nothing # IPD
end

function measurement_combined(x, u, p, t)
    (; xid) = p
    xc = ComponentVector(x, xid)

    ocv = measurement_gp(p.ocv, xc.ocv, u.s)
    r0 = measurement_gp(p.r0, xc.r0, u.s)
    vrc = xc.rc # measurement rc
    ocv + u.i * r0 + vrc |> SVector{1}
end

function R2combined(x, u, p, t)
    ocv = uncertainty_gp(p.ocv, u.s)
    r0 = uncertainty_gp(p.r0, u.s)
    vrc = p.rc.R2
    ocv + u.i^2 * r0 + vrc |> SMatrix{1,1}
end

function Cjac(x, u, p, t)
    (; C) = p.cache
    ForwardDiff.jacobian!(C, x -> measurement_combined(x, u, p, t), x)
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

##
function build_kf(n=21)
    b0 = collect(range(0, 1, n))
    b0n = StatsBase.transform(zt.s, b0)
    r0 = StatsBase.transform(zt.r, [0.03]) |> first

    # kernel1 = LinearKernel() + 0.02 * with_lengthscale(SEKernel(), 0.33)
    kernel1 = 0.02 * with_lengthscale(SEKernel(), 0.8)
    rgp1 = RGP(kernel1, b0n)

    kernel2 = 0.02 * with_lengthscale(SEKernel(), 1.0)
    rgp2 = RGP(r0, kernel2, b0n)

    R = StatsBase.transform(zt.r, [15e-3]) |> first
    Σ = StatsBase.transform(zt.r, [0.015]) |> first
    σ1 = StatsBase.transform(zt.r, [1e-3]) |> first
    σ2 = StatsBase.transform(zt.r, [sqrt(1e-3)]) |> first
    rc = RC(0.0, Σ, σ1^2, σ2, (; ts=1.0, R, τ=60))

    nx = (length(rc.μ0) + length(rgp1.μ0) + (length(rgp2.μ0)))
    p = (; cache=(;
        A=1.0I(nx),
        C=zeros(1, nx),
    ))
    p.cache.A[end, end] = rc.A # 
    rgps = (; ocv=rgp1, r0=rgp2, rc)

    make_ekf(rgps, dynamics_combined, measurement_combined, R2combined; Ajac, Cjac, p)
end

kf = build_kf()
@time run_kf!(kf, us, ys)

# @benchmark run_kf!($kf, $us, $ys)
# @profview_allocs prof_kf(kf, us, ys)



let fig = Figure(size=(600, 600)), df = df_out
    colors = Makie.wong_colors()
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / Ω"

    # for n in (11, 21, 51, 101)
    # n = 21
    # kf = build_kf(n)
    # t = @timed run_kf!(kf, us, ys)
    # (; time, bytes) = t
    # memory = 1e-6 * bytes
    # @info n time memory

    # predict new points -> mean and std
    # smin, smax = df.s |> extrema
    smin = 0.0
    smax = 1.0
    bgp = smin:0.01:smax
    bgpn = StatsBase.transform(zt.s, bgp)
    ocv = predict_gp(kf, bgpn, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
    ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)


    # plot results 
    # lines!(ax[1], 0:0.01:1, f1.(0:0.01:1), color=colors[1], label="f1(x)")
    lines!(ax[1], bgp, ocvμ)
    band!(ax[1], bgp, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha=0.8)
    # scatter!(ax[1], df_train.s, df.y, color=(:red, 0.5), label="Data")
    # axislegend(ax[1]; merge=true, position=:lt)

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


sol = forward_trajectory(kf, us, ys)
sol = smooth(kf, us, ys)

e = LLPF.sse(kf, us, ys)

ye = zeros(length(ys))
LLPF.prediction_errors!(ye, kf, us, ys)



let df = df_out
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]
    y = reduce(vcat, sol.y)
    v = StatsBase.reconstruct(zt.v, y)
    lines!(ax[1], df.t / 3600, df.v)
    lines!(ax[1], df.t / 3600, v)
    lines!(ax[2], df.t / 3600, df.v - v)
    fig
end

