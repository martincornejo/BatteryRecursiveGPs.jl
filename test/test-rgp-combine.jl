using RecursiveGPs
using AbstractGPs
using StaticArrays

using Distributions
using LowLevelParticleFilters
import LowLevelParticleFilters as LLPF

using DataFrames
using Random

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


## === dataset
f1(b) = exp(b)
f2(b) = 0.1 + 0.5 * b + 0.1 * sinpi(b * 2) # <- function to infer
df = let n = 100
    rng = Xoshiro(123)
    b = 0.1 .+ rand(rng, n) / 1.5
    i = 0.2 .* randn(rng, n)
    y = @. f1(b) + i * f2(b)
    DataFrame(; b, i, y)
end
ys = [SA[y] for y in df.y]
us = [[x.b, x.i] for x in eachrow(df)]

# === model
b0 = collect(0:0.05:1)

# m1(x) = 0.1 + 0.5 .* x
# kernel1 = LinearKernel() + 0.02 * with_lengthscale(SEKernel(), 0.1)
kernel1 = 0.02 * with_lengthscale(SEKernel(), 0.1)
# rgp1 = RGP(m1, kernel1, b0)
rgp1 = RGP(kernel1, b0)

kernel2 = 0.02 * with_lengthscale(SEKernel(), 0.1)
rgp2 = RGP(kernel2, b0)

rgps = (; a=rgp1, b=rgp2)

dynamics(x, u, p, t) = x

function measurement_combined(x, u, p, t)
    (; xid) = p
    xc = ComponentVector(x, xid)
    μ1 = measurement_gp(p.a, xc.a, u[1])
    μ2 = measurement_gp(p.b, xc.b, u[1])
    μ1 + u[2] * μ2 |> SVector{1}
end

function R2combined(x, u, p, t)
    R1 = uncertainty_gp(p.a, u[1])
    R2 = uncertainty_gp(p.b, u[1])
    R1 + u[2]^2 * R2 |> SMatrix{1,1}
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

function predict(kf, u)
    y1 = predict_gp(kf, u.b, :a)
    y2 = predict_gp(kf, u.b, :b)
    μ = @. y1.μ + u.i * y2.μ
    σ = @. y1.σ + u.i^2 * y2.σ
    (; μ, σ)
end


nx = (length(rgp1.μ0) + (length(rgp2.μ0)))
p = (; cache=(;
    A=I(nx),
    C=zeros(1, nx),
))
rgps = (; a=rgp1, b=rgp2)
kf = make_ekf(rgps, dynamics, measurement_combined, R2combined; Ajac, Cjac, p)

run_kf!(kf, us, ys)

@benchmark run_kf!($kf, $us, $ys)
# @profview_allocs prof_kf(kf, us, ys)




let fig = Figure(size=(600, 600))
    colors = Makie.wong_colors()
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]

    # predict new points -> mean and std
    bgp = 0:0.01:1
    (; μ, σ) = predict_gp(kf, bgp, :a)

    # plot results 
    lines!(ax[1], 0:0.01:1, f1.(0:0.01:1), color=colors[1], label="f1(x)")
    lines!(ax[1], bgp, μ, color=colors[2], label="GP1")
    band!(ax[1], bgp, μ + 2σ, μ - 2σ, color=(colors[2], 0.5), label="GP1")
    scatter!(ax[1], df.b, df.y, color=(:red, 0.5), label="Data")
    axislegend(ax[1]; merge=true, position=:lt)
    fig


    # predict new points -> mean and std
    bgp = 0:0.01:1
    (; μ, σ) = predict_gp(kf, bgp, :b)

    # plot results 
    lines!(ax[2], 0:0.01:1, f2.(0:0.01:1), color=colors[1], label="f2(x)")
    lines!(ax[2], bgp, μ, color=colors[2], label="GP2")
    band!(ax[2], bgp, μ + 2σ, μ - 2σ, color=(colors[2], 0.5), label="GP2")
    axislegend(ax[2]; merge=true, position=:lt)
    fig
end

