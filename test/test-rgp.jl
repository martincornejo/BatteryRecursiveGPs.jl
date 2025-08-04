using RecursiveGPs
using AbstractGPs
using StaticArrays

using Distributions
using LowLevelParticleFilters
import LowLevelParticleFilters as LLPF

using DataFrames

using BenchmarkTools

using LinearAlgebra

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


function predict(kf, rgp::RGP, b)
    (; gp, b0, Σ0⁻¹) = rgp
    H = cov(gp, b, b0) * Σ0⁻¹
    μ = H * kf.x

    R = cov(gp, b) - H * cov(gp, b0, b) #eq.7 
    Σ = R + H * kf.R * H' #eq.9
    σ = sqrt.(diag(Σ))
    (; μ, σ)
end

## === dataset
f(b) = 0.5 * b + 0.1 * sinpi(b * 2) # <- function to infer
df = let n = 100
    b = 0.1 .+ rand(n) / 1.5
    y = f.(b)
    DataFrame(; b, y)
end
ys = [SA[y] for y in df.y]
us = [SA[b] for b in df.b]

# === model
b0 = collect(0:0.05:1)

m1(x) = 0.1 + 0.5 .* x
kernel1 = 0.02 * with_lengthscale(SEKernel(), 0.1)
rgp1 = RGP(m1, kernel1, b0)

# kernel2 = LinearKernel() + 0.02 * with_lengthscale(SEKernel(), 0.1)
# rgp2 = RGP(kernel2, b0)

# begin
#     (; μ0, Σ0) = rgp1
#     d0 = MvNormal(μ0, Σ0)
#     nx = length(μ0)
#     nu = 1
#     ny = 1
#     R1 = zeros(nx, nx)
#     ExtendedKalmanFilter(RecursiveGPs.dynamics, RecursiveGPs.measurement, R1, R2f, d0; nx, nu, ny, p=rgp1)
# end

kf = RecursiveGPs.make_kf(rgp1)

# for (u, y) in zip(us, ys)
#     kf(u, y)
# end

# dynamics(x, u, p, t) = x
# measurement(x, u, p, t) = SA[measurement_gp(p, x, u[1])]

# R2(x, u, p, t) = SMatrix{1,1}(uncertainty_gp(p, u[1]))

# function make_kf(rgp::RGP)
#     (; μ0, Σ0) = rgp
#     d0 = MvNormal(μ0, Σ0)
#     nx = length(μ0)
#     nu = 1
#     ny = 1
#     R1 = Diagonal(zero(μ0))
#     ExtendedKalmanFilter(dynamics, measurement, R1, R2, d0; nx, nu, ny, p=rgp)
# end

# kf = make_kf(rgp1)



run_kf!(kf, us, ys)

# @profview_allocs prof_kf(kf, us, ys)


let fig = Figure()
    colors = Makie.wong_colors()

    # predict new points -> mean and std
    bgp = 0:0.01:1
    (; μ, σ) = predict(kf, rgp1, bgp)

    # plot results 
    ax = Makie.Axis(fig[1, 1])
    lines!(ax, 0:0.01:1, f.(0:0.01:1), color=colors[1], label="f(x)")
    scatter!(ax, df.b, df.y, color=(:red, 0.5), label="Data")
    lines!(ax, bgp, μ, color=colors[2], label="GP")
    band!(ax, bgp, μ + 2σ, μ - 2σ, color=(colors[2], 0.5), label="GP")
    axislegend(ax; merge=true, position=:lt)
    fig
end

