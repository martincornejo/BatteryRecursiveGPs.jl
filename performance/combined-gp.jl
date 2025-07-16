using LowLevelParticleFilters
using Distributions
using LinearAlgebra

using StableRNGs

using DataFrames

using AbstractGPs

using CairoMakie

import ComponentArrays: ComponentVector, ComponentMatrix, getaxes

## synthetic dataset

f1(b) = 0.1 + 0.5 * b + 0.1 * sinpi(b * 2) # <- function to infer

f2(b) = exp(b)

df = let n = 100
    rng = StableRNG(123)
    b = 0.1 .+ rand(rng, n) / 1.5
    i = 0.2 .* randn(rng, n)
    y = @. f2(b) + i * f1(b)
    DataFrame(; b, i, y)
end

let fig = Figure()
    ax = Makie.Axis(fig[1, 1], xlabel="b", ylabel="f(b)")
    lines!(ax, 0:0.01:1, f2.(0:0.01:1), label="f2")
    sc = scatter!(ax, df.b, df.y, color=abs.(df.i))
    Colorbar(fig[1, 2], sc, label="abs(i)")
    axislegend(ax; position=:lt)
    fig
end

##
dynamics(x, u, p, t) = x # identity

function measurement_gp(g, b, p)
    (; gp, b0, μ0, Σ0⁻¹) = p

    H = cov(gp, b, b0) * Σ0⁻¹
    mean(gp, b) + H * (g - μ0)
    # H * g
end

function Hfun(x, u, p, t)
    (; gp, b0, Σ0⁻¹) = p
    b = u
    cov(gp, b, b0) * Σ0⁻¹
end

function R2fun(x, u, p, t)
    (; gp, b0) = p
    b = u
    H = Hfun(x, u, p, t)
    cov(gp, b) - H * cov(gp, b0, b)
end

##
function measurement_combined(x, u, p, t)
    (; xid) = p
    c = ComponentVector(x, xid)
    μ1 = measurement_gp(c.x1, [u[1]], p.x1)
    μ2 = measurement_gp(c.x2, [u[1]], p.x2)
    @. μ2 + u[2] * μ1  # @. to broadcast (since μ1 and μ2 are matrices)
end

function R2combined(x, u, p, t)
    (; xid) = p
    c = ComponentVector(x, xid)
    R1 = R2fun(c.x1, [u[1]], p.x1, t)
    R2 = R2fun(c.x2, [u[1]], p.x2, t)
    @. R1 + u[2]^2 * R2  # @. to broadcast (since μ1 and μ2 are matrices)
end

function make_kf()
    m1(x) = 0.1 + 0.5 .* x
    kernel1 = 0.02 * with_lengthscale(SEKernel(), 0.1)
    gp1 = GP(m1, kernel1)

    kernel2 = LinearKernel() + 0.02 * with_lengthscale(SEKernel(), 0.1)
    gp2 = GP(kernel2)

    b0 = collect(0:0.05:1)
    nb = length(b0)

    # initial guess
    μ1 = mean(gp1, b0)
    Σ1 = cov(gp1, b0) + 1e-6I
    Σ1⁻¹ = inv(Σ1)


    μ2 = mean(gp2, b0)
    Σ2 = cov(gp2, b0) + 1e-6I
    Σ2⁻¹ = inv(Σ2)


    x0 = ComponentVector(; x1=μ1, x2=μ2)
    Σ0 = false .* x0 * x0'
    Σ0[:x1, :x1] = Σ1
    Σ0[:x2, :x2] = Σ2

    d0 = MvNormal(x0, Σ0)

    xid = getaxes(x0)
    Σid = getaxes(Σ0)

    p = (;
        xid,
        Σid,
        x1=(;
            f=f1, # only for validation purposes
            b0,      # basis vector
            gp=gp1,     # gp (mean + kernel functions)
            μ0=μ1,     # mean basis vector
            # Σ0,
            Σ0⁻¹=Σ1⁻¹,   # inv convariance basis vector
        ),
        x2=(;
            f=f2, # only for validation purposes
            b0,      # basis vector
            # gp=gp2,     # gp (mean + kernel functions)
            gp=gp1,     # gp (mean + kernel functions)
            # μ0=μ2,     # mean basis vector
            μ0=μ1,     # mean basis vector
            # Σ0,
            # Σ0⁻¹=Σ2⁻¹,   # inv convariance basis vector
            Σ0⁻¹=Σ1⁻¹,   # inv convariance basis vector
        ),
        Ajac=I(2nb),
        cache=(
            C=zeros(1, 2nb),
        )
    )

    R1 = Diagonal(zero(x0))

    function fCjac(x, u, p, t)
        (; cache) = p
        (; C) = cache
        ForwardDiff.jacobian!(C, x -> measurement_combined(x, u, p, t), x)
        # return C
    end
    Ajac(x, u, p, t) = p.Ajac

    kf = ExtendedKalmanFilter(dynamics, measurement_combined, R1, R2combined, d0; nx=length(x0), ny=1, nu=1, p)
    # kf = ExtendedKalmanFilter(dynamics, measurement_combined, R1, R2combined, d0; Ajac, Cjac=fCjac, nx=length(x0), ny=1, nu=1, p)
end

# ys = [[y] for y in df.y]
# us = [[x.b, x.i] for x in eachrow(df)]




function plot_gps(kf)
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]
    colors = Makie.wong_colors()
    (; xid, Σid) = p
    x = ComponentVector(kf.x, xid)
    Σx = ComponentMatrix(kf.R, Σid)

    # predict new points -> mean and std
    for (i, id) in enumerate((:x1, :x2))
        (; f, gp, b0, Σ0⁻¹) = p[id]
        xid = x[id]
        Σxid = Σx[id, id]
        bgp = 0:0.01:1
        H = cov(gp, bgp, b0) * Σ0⁻¹
        μ = H * xid
        R = cov(gp, bgp) - H * cov(gp, b0, bgp) #eq.7 
        Σgp = R + H * Σxid * H' #eq.9
        σ = sqrt.(diag(Σgp))

        # plot results 
        lines!(ax[i], 0:0.01:1, f.(0:0.01:1), color=colors[1], label="f(x)")
        # scatter!(ax, df.b, df.y, color=(:red, 0.5), label="Data")
        lines!(ax[i], bgp, μ, color=colors[2], label="GP")
        band!(ax[i], bgp, μ + 2σ, μ - 2σ, color=(colors[2], 0.5), label="GP")
        axislegend(ax[i]; merge=true, position=:lt)
        ax[i].title = "f$i(b)"
    end
    fig
end