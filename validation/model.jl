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

include("../dual/synthetic-data.jl")

## === dataset
begin
    # read OCV look-up-table and current profile, define internal resistance function
    df_ocv = CSV.File("data/ocv.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc, extrapolation=ExtrapolationType.Constant)

    df = CSV.File("data/profile.csv") |> DataFrame
    fi = ConstantInterpolation(df.i, df.t)

    fR0(s) = 0.01 + 0.005 * s + 0.005 * sinpi(-0.2 + s * 1.5)

    # simulate battery operation
    tspan = (0, 24 * 3600) # one day
    Ts = 1.0 # # time sampling

    df = generate_timeseries(; Ts, tspan, fi, focv, fR0)
    plot_timeseries(df) |> display

    # create input / output data 
    zt = fit_zscore(df)
    dfn = normalize_data(df, zt)

    us = [(; i, î) for (i, î) in zip(df.i, dfn.i)]
    ys = [SA[y] for y in dfn.v]
end


# === model
function SOC(soc0, σ0, σ1)
    x0 = ComponentVector(
        soc=soc0,
    )
    Σ0 = false .* x0 * x0'
    Σ0 .= σ0

    R1 = [σ1;;]

    return (; μ0=x0, Σ0, R1)
end

# dynamics(x, u, p, t) = x
function dynamics!(x⁺, x, u, p, t)
    (; q, Ts, xid) = p # params
    (; i) = u # control
    x = ComponentVector(x, xid)
    x⁺ = ComponentVector(x⁺, xid)
    x⁺ .= x # previous values

    x⁺.soc.soc = x.soc.soc + i * Ts / (q * 3600)
    # dx[2] = v1 * exp(-Ts / τ1) + i * R1 * (1 - exp(-Ts / τ1))
end

function measurement(x, u, p, t)
    (; xid) = p
    xc = ComponentVector(x, xid)
    ocv = measurement_gp(p.ocv, xc.ocv, xc.soc.soc)
    r0 = measurement_gp(p.r0, xc.r0, xc.soc.soc)
    ocv + u.î * r0 |> SVector{1}
end

function R2(x, u, p, t)
    (; xid, vσ) = p
    xc = ComponentVector(x, xid)
    ocv = uncertainty_gp(p.ocv, xc.soc.soc)
    r0 = uncertainty_gp(p.r0, xc.soc.soc)
    ocv + u.î^2 * r0 + vσ |> SMatrix{1,1}
end

function Cjac(x, u, p, t)
    (; C) = p.cache
    ForwardDiff.jacobian!(C, x -> measurement(x, u, p, t), x)
    # return Cjac
end

function Ajac(x, u, p, t)
    (; A) = p.cache
    return A
end

# function predict(kf, df)
#     dfn = normalize_data(df)
#     ocv = predict_gp(kf, dfn.s, :ocv)
#     r0 = predict_gp(kf, dfn.s, :r0)
#     μ = @. ocv.μ + u.i * r0.μ
#     σ = @. ocv.σ + u.i^2 * r0.σ
#     (; μ, σ)
# end

function build_kf(θ, focv_prior, zt; n=21)
    b0 = collect(range(0, 1, n)) # basis vector
    # priors
    r0 = StatsBase.transform(zt.r, [15e-3]) |> first

    # OCV GP
    kernel1 = θ.ocv.σ * with_lengthscale(SEKernel(), θ.ocv.ℓ)
    rgp1 = RGP(focv_prior, kernel1, b0)

    # R0 GP
    kernel2 = θ.r0.σ * with_lengthscale(SEKernel(), θ.r0.ℓ)
    rgp2 = RGP(r0, kernel2, b0)

    # SOC estimation
    soc = SOC(0.5, 0.01, θ.soc.σ2)

    # model
    nx = (length(soc.μ0) + length(rgp1.μ0) + (length(rgp2.μ0)))
    p = (;
        cache=(;
            A=I(nx),
            C=zeros(1, nx),
        ),
        Ts=1.0,
        q=4.8,
        vσ=StatsBase.transform(zt.σ, [0.001^2]) |> first,
    )
    rgps = (; soc, ocv=rgp1, r0=rgp2)

    make_ekf(rgps, dynamics!, measurement, R2; Ajac, Cjac, p)
end


function plot_soc_estimation(sol, df)
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]
    s´ = sol.xt .|> first
    sσ = [sqrt(R[1, 1]) for R in sol.Rt]
    lines!(ax[1], df.t / 3600, s´)
    band!(ax[1], df.t / 3600, s´ - 2sσ, s´ + 2sσ, alpha=0.5)
    lines!(ax[1], df.t / 3600, df.s)

    Δ = s´ - df.s
    lines!(ax[2], df.t / 3600, Δ)
    band!(ax[2], df.t / 3600, Δ - 2sσ, Δ + 2sσ, alpha=0.5)

    xlims!(ax[1], df[begin, :t] / 3600, df[end, :t] / 3600)
    xlims!(ax[2], df[begin, :t] / 3600, df[end, :t] / 3600)
    linkxaxes!(ax...)
    fig
end


function plot_ecm(kf, df, zt, focv, fR0; closeup=false)
    fig = Figure(size=(600, 600))
    colors = Makie.wong_colors()
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / Ω"
    ax[2].xlabel = "SOC / p.u."
    hidexdecorations!(ax[1], ticks=false, grid=false)

    soc = 0:0.01:1

    # OCV 
    ocv = predict_gp(kf, soc, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
    ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)

    lines!(ax[1], soc, ocvμ)
    band!(ax[1], soc, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha=0.8)
    lines!(ax[1], soc, focv(soc), color=:black, linestyle=:dot)

    if closeup
        ylims!(ax[1], 3.45, 4.2)
    end

    # R0
    r0 = predict_gp(kf, soc, :r0)
    rμ = StatsBase.reconstruct(zt.r, r0.μ)
    rσ = StatsBase.reconstruct(zt.r, r0.σ)

    lines!(ax[2], soc, rμ)
    band!(ax[2], soc, rμ + 2rσ, rμ - 2rσ, alpha=0.8)
    lines!(ax[2], soc, fR0.(soc), color=:black, linestyle=:dot)

    # data - SOC window
    smin, smax = df.s |> extrema
    vlines!(ax[1], [smin, smax], color=:red)
    vlines!(ax[2], [smin, smax], color=:red)

    # xlims!(ax[1], 0, 1)
    # xlims!(ax[2], 0, 1)

    fig
end


focv´ = let # focv prior
    df_ocv = CSV.File("data/ocv.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc)
    soc´ = 0.0:0.05:1.0
    ocv´ = StatsBase.transform(zt.v, focv(soc´))
    focv´ = LinearInterpolation(ocv´, soc´; extrapolation=ExtrapolationType.Constant)
end

θ = (;
    ocv=(; σ=0.1, ℓ=0.05),
    r0=(; σ=0.001, ℓ=0.5),
    soc=(; σ2=1e-7),
)
kf = build_kf(θ, focv´, zt)

# for (u, y) in zip(us, ys)
#     LLPF.update!(kf, u, y)
# end
sol = forward_trajectory(kf, us, ys)

plot_soc_estimation(sol, df)

plot_ecm(kf, df, zt, focv, fR0)

