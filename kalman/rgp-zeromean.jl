using LowLevelParticleFilters
using Distributions
using LinearAlgebra

using DataFrames

using AbstractGPs

using CairoMakie

import ComponentArrays: ComponentVector, getaxes

## synthetic dataset

f(b) = 0.1 * sinpi(b * 2) # <- function to infer

df = let n = 30
    b = 0.1 .+ rand(n) / 1.5
    y = f.(b)
    DataFrame(; b, y)
end

##
kernel = 0.02 * with_lengthscale(SEKernel(), 0.1)
gp = GP(kernel)

b0 = collect(0:0.05:1)
μ0 = mean(gp, b0)
Σ0 = cov(gp, b0) + 1e-6I
d0 = MvNormal(μ0, Σ0)

Σ0⁻¹ = inv(Σ0)

p = (;
    gp,     # gp (mean + kernel functions)
    b0,      # basis vector
    # μ0,     # mean basis vector
    # Σ0,
    Σ0⁻¹,   # inv convariance basis vector
)

A = I(length(b0))
B = zeros(length(b0), 1)
R1 = Diagonal(zero(b0))


function Hfun(x, u, p, t)
    (; gp, b0, Σ0⁻¹) = p
    b = u
    cov(gp, b, b0) * Σ0⁻¹
end

function Cfun(x, u, p, t)
    Hfun(x, u, p, t)
end

D = [0.0;;]

function R2fun(x, u, p, t) # -> P
    (; gp, b0) = p
    b = u
    H = Hfun(x, u, p, t)
    cov(gp, b) - H * cov(gp, b0, b) #eq.7 
end

ys = [[y] for y in df.y]
bs = [[b] for b in df.b]


kf = KalmanFilter(A, B, Cfun, D, R1, R2fun, d0; nx=length(b0), ny=1, nu=1, p)

for (u, y) in zip(bs, ys)
    kf(u, y)
end


let fig = Figure()
    colors = Makie.wong_colors()

    # predict new points -> mean and std
    bgp = 0:0.01:1
    H = cov(gp, bgp, b0) * Σ0⁻¹
    μ = H * kf.x

    R = cov(gp, bgp) - H * cov(gp, b0, bgp) #eq.7 
    Σgp = R + H * kf.R * H' #eq.9
    σ = sqrt.(diag(Σgp))

    # plot results 
    ax = Makie.Axis(fig[1, 1])
    lines!(ax, 0:0.01:1, f.(0:0.01:1), color=colors[1], label="f(x)")
    scatter!(ax, df.b, df.y, color=(:red, 0.5), label="Data")
    lines!(ax, bgp, μ, color=colors[2], label="GP")
    band!(ax, bgp, μ + 2σ, μ - 2σ, color=(colors[2], 0.5), label="GP")
    axislegend(ax; merge=true, position=:lb)
    fig
end


