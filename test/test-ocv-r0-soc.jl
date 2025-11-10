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

using PreallocationTools


## === dataset
include("../dual/synthetic-data.jl")

# === model
#

function SOC(soc0, σ=0.01)
    x0 = ComponentVector(
        soc=soc0,
    )
    Σ0 = false .* x0 * x0'
    Σ0 .= σ

    R1 = [1e-7;;]
    # R1 = 0.01

    return (; μ0=x0, Σ0, R1)
end


# dynamics(x, u, p, t) = x
function dynamics!(dx, x, u, p, t)
    (; Ts, q, xid) = p
    # (; Ts, q, R1, τ1, xid) = p
    # soc = x[1]
    # v1 = x[2]
    dx .= x

    dx = ComponentVector(dx, xid)
    x = ComponentVector(x, xid)
    (; soc,) = x
    i = u.i # control
    # TODO: convert to 

    # dx[1] = soc + i * Ts / (q * 3600)
    # dx[2] = v1 * exp(-Ts / τ1) + i * R1 * (1 - exp(-Ts / τ1))
    dx.soc.soc = soc.soc + i * Ts / (q * 3600)
    # dx.v1 = v1 * exp(-Ts / τ1) + i * R1 * (1 - exp(-Ts / τ1))
end

function measurement_combined(x, u, p, t)
    (; xid) = p
    xc = ComponentVector(x, xid)
    ocv = measurement_gp(p.ocv, xc.ocv, xc.soc.soc)
    r0 = measurement_gp(p.r0, xc.r0, xc.soc.soc)
    ocv + u.î * r0 |> SVector{1}
end

function R2combined(x, u, p, t)
    (; xid, vσ) = p
    xc = ComponentVector(x, xid)
    ocv = uncertainty_gp(p.ocv, xc.soc.soc)
    r0 = uncertainty_gp(p.r0, xc.soc.soc)
    ocv + u.î^2 * r0 + vσ |> SMatrix{1,1}
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

# function predict(kf, df)
#     dfn = normalize_data(df)
#     ocv = predict_gp(kf, dfn.s, :ocv)
#     r0 = predict_gp(kf, dfn.s, :r0)
#     μ = @. ocv.μ + u.i * r0.μ
#     σ = @. ocv.σ + u.i^2 * r0.σ
#     (; μ, σ)
# end

function build_kf(zt, n=21)
    b0 = collect(range(0, 1, n))
    # b0n = StatsBase.transform(zt.s, b0)
    r0 = StatsBase.transform(zt.r, [15e-3]) |> first

    # kernel1 = LinearKernel() + 0.02 * with_lengthscale(SEKernel(), 0.33)
    df_ocv = CSV.File("data/ocv.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc)
    soc´ = 0.0:0.05:1.0
    ocv´ = StatsBase.transform(zt.v, focv(soc´))
    focv´ = LinearInterpolation(ocv´, soc´; extrapolation=ExtrapolationType.Constant)

    kernel1 = 0.1 * with_lengthscale(SEKernel(), 0.05)

    gp1 = GP(focv´, kernel1)
    rgp1 = RGP(gp1, b0)

    kernel2 = 0.001 * with_lengthscale(SEKernel(), 0.5)
    rgp2 = RGP(r0, kernel2, b0)

    soc = SOC(0.5, 0.01)

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

    make_ekf(rgps, dynamics!, measurement_combined, R2combined; Ajac, Cjac, p)
    # make_ekf(rgps, dynamics!, measurement_combined, R2combined; Cjac, p)
end

import LowLevelParticleFilters: symmetrize, extended_logpdf, SimpleMvNormal, PDMats, get_mat

kf = build_kf(zt)

sol = forward_trajectory(kf, us, ys)

let fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]
    s´ = sol.xt .|> first
    sσ = [sqrt(R[1, 1]) for R in sol.Rt]
    lines!(ax[1], df.t, s´)
    band!(ax[1], df.t, s´ - 2sσ, s´ + 2sσ, alpha=0.5)
    lines!(ax[1], df.t, df.s)

    Δ = s´ - df.s
    lines!(ax[2], df.t, Δ)
    band!(ax[2], df.t, Δ - 2sσ, Δ + 2sσ, alpha=0.5)
    fig
end


# for (u, y) in zip(us, ys)
#     LLPF.update!(kf, u, y)
# end


let fig = Figure(size=(600, 600))
    colors = Makie.wong_colors()
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / Ω"

    # for n in (11, 21, 51, 101)
    n = 21
    # kf = build_kf(n)
    # t = @timed run_kf!(kf, us, ys)
    # (; time, bytes) = t
    # memory = 1e-6 * bytes
    # @info n time memory

    # predict new points -> mean and std
    # smin, smax = df.s |> extrema
    # bgp = StatsBase.transform(zt.s, smin:0.01:smax)
    soc = 0:0.01:1
    ocv = predict_gp(kf, soc, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
    ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)


    # plot results 
    # lines!(ax[1], 0:0.01:1, f1.(0:0.01:1), color=colors[1], label="f1(x)")
    lines!(ax[1], soc, ocvμ)
    band!(ax[1], soc, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha=0.8)
    # scatter!(ax[1], df_train.s, df.y, color=(:red, 0.5), label="Data")
    lines!(ax[1], soc, focv(soc), color=:black, linestyle=:dot)
    # axislegend(ax[1]; merge=true, position=:lt)

    # predict new points -> mean and std
    r0 = predict_gp(kf, soc, :r0)
    rμ = StatsBase.reconstruct(zt.r, r0.μ)
    rσ = StatsBase.reconstruct(zt.r, r0.σ)

    # plot results 
    # lines!(ax[2], 0:0.01:1, f2.(0:0.01:1), color=colors[1], label="f2(x)")
    lines!(ax[2], soc, rμ)
    band!(ax[2], soc, rμ + 2rσ, rμ - 2rσ, alpha=0.8)
    lines!(ax[2], soc, fR0.(soc), color=:black, linestyle=:dot)
    # axislegend(ax[2]; merge=true, position=:lt)
    # end

    smin, smax = df.s |> extrema
    vlines!(ax[1], [smin, smax], color=:red)
    vlines!(ax[2], [smin, smax], color=:red)

    fig
end






function dynamics_state(x, u, p, t)
    # (; Ts, R1, τ1, xid) = p
    (; Ts, q) = p
    # soc = x[1]
    # v1 = x[2]
    # dx = ComponentArray(dx, xid)
    # xc = ComponentArray(x, xid)
    # (; soc, v1, q) = xc
    soc = x[1]
    i = u[1] # control

    # dx[1] = soc + i * Ts / (q * 3600)
    # dx[2] = v1 * exp(-Ts / τ1) + i * R1 * (1 - exp(-Ts / τ1))
    soc⁺ = soc + i * Ts / (q * 3600)
    SA[soc⁺]
end


function measurement_state(x, u, p, t)
    (; kf) = p
    (; xid, ocv, r0) = kf.p
    soc = x[1]
    kfx = ComponentVector(kf.x, xid)
    ocv = measurement_gp(ocv, kfx.ocv, soc)
    r0 = measurement_gp(r0, kfx.r0, soc)
    v = ocv + u.i * r0 |> SVector{1}
    # StatsBase.reconstruct(zt.v, v) |> SVector{1}
end

function R2_state(x, u, p, t)
    (; kf) = p
    (; xid, ocv, r0) = kf.p
    soc = x[1]
    # kfx = ComponentVector(kf.x, xid)
    ocv = uncertainty_gp(ocv, soc)
    r0 = uncertainty_gp(r0, soc)
    ocv + u.i^2 * r0 |> SMatrix{1,1}
end


x0 = SA[0.5]
θ = (; kf, Ts=1.0, q=4.8)

measurement_state(x0, us[1], θ, 0.0)
R2_state(x0, us[1], θ, 0.0)


d0 = MvNormal(x0, 0.1I)
R1 = @SMatrix [0.01;;]
kf2 = ExtendedKalmanFilter(dynamics_state, measurement_state, R1, R2_state, d0; nx=1, nu=1, ny=1, p=θ)

smoothsol = smooth(kf2, us, ys)

let fig = Figure()
    ax = Axis(fig[1, 1])
    s´ = smoothsol.xT .|> first

    horizon = 24 * 3600
    df_ = df[1:horizon, :]

    lines!(ax, df_.t, s´)
    lines!(ax, df_.t, 0.4 .+ df_.s ./ 4.8)
    fig
end


begin
    ys = [SA[y+0.01randn()+0.1] for y in df_train.v]
    us = [(; i=x.i, s) for (x, s) in zip(eachrow(df_train), s´)]
end
