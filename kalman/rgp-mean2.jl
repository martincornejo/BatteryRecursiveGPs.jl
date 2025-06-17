module RecursiveGPs
using LinearAlgebra
using ComponentArrays
using AbstractGPs
using DataFrames
using Distributions
using LowLevelParticleFilters
using CairoMakie
using BenchmarkTools


export RGPModel, learn!, predict, construct_kf
"""
RGP model 
"""
mutable struct RGPModel
    gp::GP
    σ::Float64
    b0::Vector{Float64}
    μ::Vector{Float64}
    Σ::Matrix{Float64}

    # Fixed
    μ0::Vector{Float64}
    Σ0⁻¹::Matrix{Float64}
end

function RGPModel(gp, σ, b0)
    μ = mean_vector(gp.mean, b0)
    Σ = cov(gp, b0) + 1e-6I

    μ0 = copy(μ)
    Σ0⁻¹ = copy(inv(Σ))

    return RGPModel(gp, σ, b0, μ, Σ, μ0, Σ0⁻¹)
end

function Base.show(io::IO, model::RGPModel)
    println(io, "RGPModel summary:")
    println(io, "  GP: ", model.gp)
    println(io, "  σ: ", model.σ)
    println(io, "  Basis size: ", length(model.b0))
end

function predict(rgp::RGPModel, b; train=true)
    """
    Inference step at batch points.
    Two modes:
        train_mode = True -> Does not include noise since is already on update step
        train_mode = False -> Includes GP noise 
    """

    H = cov(rgp.gp, b, rgp.b0) * rgp.Σ0⁻¹

    μ_predict = mean(rgp.gp, b) + H * (rgp.μ - rgp.μ0) #eq.6 +

    R = cov(rgp.gp, b) - H * cov(rgp.gp, rgp.b0, b) #eq.7 
    Σ_predict = R + H * rgp.Σ * H' #eq.9

    if train == false
        Σ_predict += rgp.σ^2 * I
    end

    return (
        μ=μ_predict,
        Σ=Σ_predict
    )

end

function update_step!(rgp::RGPModel, predict_batch, H, y)
    """
    Update rgp parameters
    """

    Gk = rgp.Σ * H' * inv(predict_batch.Σ + rgp.σ^2 * I(size(y, 1))) #eq.12

    new_μ = rgp.μ + Gk * (y - predict_batch.μ) #eq.10
    new_Σ = rgp.Σ - Gk * H * rgp.Σ #eq.11

    rgp.μ = new_μ
    rgp.Σ = new_Σ
    return
end

function learn!(rgp::RGPModel, b, y)
    """ 
    Performs RGP learning
    Inputs:
        - rgp model 
        - dataLoader: Data already structured so is fast to iterate

    Note:
        - Inference and update steps separable at the moment for future when switch between Hyp or non-Hyp
    """

    H = cov(rgp.gp, b, rgp.b0) * rgp.Σ0⁻¹
    ## Predict value
    predict_batch = predict(rgp, b, train=true)

    ## Update model by predicted value error
    update_step!(rgp, predict_batch, H, y)

end


function Hfun(x, u, p, t)
    (; gp, b0, Σ0⁻¹) = p
    b = [u[1]]
    return cov(gp, b, b0) * Σ0⁻¹
end

function Cfun(x, u, p, t)
    return Hfun(x, u, p, t)
end

function Dfun(x, u, p, t)
    (; μ0) = p
    H = Hfun(x, u, p, t)
    return hcat(zeros(1), I(1), -H * μ0)
end


function R2fun(x, u, p, t) # -> P
    (; gp, b0) = p
    b = [u[1]]
    H = Hfun(x, u, p, t)
    return cov(gp, b) - H * cov(gp, b0, b) #eq.7 
end



function construct_kf(rgp::RGPModel)
    p = (;
        gp=rgp.gp,     # gp (mean + kernel functions)
        b0=rgp.b0,      # basis vector
        μ0=rgp.μ0,     # mean basis vector
        Σ0⁻¹=rgp.Σ0⁻¹,   # inv convariance basis vector
    )

    d0 = MvNormal(rgp.μ0, copy(rgp.Σ))
    A = I(length(p.b0))
    B = zeros(1, 3)
    R1 = Diagonal(zero(p.b0))

    kf = KalmanFilter(A, B, Cfun, Dfun, R1, R2fun, d0; nx=1, ny=1, nu=1, p)
    return kf
end


function learn2!(kf, b, y)
    (; gp) = kf.p
    μ_b = mean(gp, b)
    u = vcat(b, μ_b, 1)
    kf(u, y)
end


function R2fun(x, u, p, t) # -> P
    (; gp, b0) = p
    b = [u[1]]
    H = Hfun(x, u, p, t)
    return cov(gp, b) - H * cov(gp, b0, b) #eq.7 
end



dynamics(x, u, p, t) = x # identity

function measurement(x, u, p, t)
    (; gp, b0, μ0, Σ0⁻¹) = p
    # (; g) = x
    # (; b) = u
    g = x
    b = u

    H = cov(gp, b, b0) * Σ0⁻¹
    return mean(gp, b) + H * (x - μ0)
    # H * g
end


function construct_ukf(rgp::RGPModel)
    p = (;
        gp=rgp.gp,     # gp (mean + kernel functions)
        b0=rgp.b0,      # basis vector
        μ0=rgp.μ0,     # mean basis vector
        Σ0⁻¹=rgp.Σ0⁻¹,   # inv convariance basis vector
    )
    R1 = Diagonal(zero(rgp.b0))
    d0 = MvNormal(rgp.μ0, copy(rgp.Σ))
    kf = UnscentedKalmanFilter(dynamics, measurement, R1, R2fun, d0; nx=length(rgp.b0), ny=1, nu=1, p)
    return kf
end


function learn3!(kf, b, y)
    u = b
    kf(u, y)
end



begin
    f(b) = 0.1 * sinpi(b * 2) # <- function to infer

    df = let n = 30
        b = 0.1 .+ rand(n) / 1.5
        y = f.(b)
        DataFrame(; b, y)
    end

    kernel = 0.02 * with_lengthscale(SEKernel(), 0.1)
    m(x) = 0.5 .* x
    gp_kf = GP(m, kernel)
    gp_ukf = GP(m, kernel)
    gp_og = GP(m, kernel)
    b0 = collect(0:0.05:1)
    σ = 0.01

    rgp_kf = RGPModel(gp_og, σ, b0)
    rgp_ukf = RGPModel(gp_ukf, σ, b0)
    rgp_og = RGPModel(gp_kf, σ, b0)


    kf = construct_kf(rgp_kf)
    ukf = construct_ukf(rgp_ukf)


    ys = [[y] for y in df.y]
    bs = [[b] for b in df.b]
    function run_learn2!(kf, bs, ys)
        for (b, y) in zip(bs, ys)
            learn2!(kf, b, y)
        end
    end

    function run_learn3!(ukf, bs, ys)
        for (b, y) in zip(bs, ys)
            learn3!(ukf, b, y)
        end
    end

    function run_learn!(rgp_og, bs, ys)
        for (b, y) in zip(bs, ys)
            learn!(rgp_og, b, y)
        end
    end

    println("--KalmanFilter--")
    @time run_learn2!(kf, bs, ys)

    println("--UnscentedKalmanFilter--")
    @time run_learn3!(ukf, bs, ys)

    println("--original--")
    @time run_learn!(rgp_og, bs, ys)


    let fig = Figure(size=(1200, 800))
        colors = Makie.wong_colors()
        bgp = 0:0.01:1


        # predict new points with kf method-> mean and std
        H = cov(gp_kf, bgp, rgp_kf.b0) * rgp_kf.Σ0⁻¹
        μ_kf = H * kf.x

        R = cov(gp_kf, bgp) - H * cov(gp_kf, rgp_kf.b0, bgp) #eq.7 
        Σgp = R + H * kf.R * H' #eq.9
        σ_kf = sqrt.(diag(Σgp))

        # predict new points with ukf method
        H = cov(gp_kf, bgp, rgp_kf.b0) * rgp_kf.Σ0⁻¹
        μ_ukf = H * ukf.x

        R = cov(gp_ukf, bgp) - H * cov(gp_ukf, rgp_ukf.b0, bgp) #eq.7 
        Σgp = R + H * ukf.R * H' #eq.9
        σ_ukf = sqrt.(diag(Σgp))

        # predict new points with old method
        pred = predict(rgp_og, bgp)
        μ_og = pred.μ
        σ_og = sqrt.(diag(pred.Σ))




        # plot results 
        ax = Makie.Axis(fig[1, 1])
        lines!(ax, 0:0.01:1, f.(0:0.01:1), color=colors[1], label="f(x)")
        scatter!(ax, df.b, df.y, color=(:red, 0.5), label="Data")

        ## KF
        lines!(ax, bgp, μ_kf, color=colors[2], label="GP with KF")
        band!(ax, bgp, μ_kf + 2σ_kf, μ_kf - 2σ_kf, color=(colors[2], 0.5), label="GP with KF")


        lines!(ax, bgp, μ_ukf, color=:green, label="GP with UKF")
        band!(ax, bgp, μ_ukf + 2σ_ukf, μ_ukf - 2σ_ukf, color=(:green, 0.5), label="GP with UKF")

        lines!(ax, bgp, μ_og, color=:red, label="GP with original")
        band!(ax, bgp, μ_og + 2σ_og, μ_og - 2σ_og, color=(:red, 0.5), label="GP with original")




        axislegend(ax; merge=true, position=:lb)


        ax2 = Makie.Axis(fig[2, 1])
        lines!(ax2, bgp, abs.(μ_og .- μ_kf), color=:black, label="Difference kf-Original")
        ylims!(0.0, 0.001)
        axislegend(ax2; position=:lt)
        axislegend(ax2)

        ax3 = Makie.Axis(fig[3, 1])
        lines!(ax3, bgp, abs.(μ_og .- μ_kf), color=:black, label="Difference Ukf-Original")
        ylims!(0.0, 0.001)
        axislegend(ax3; position=:lt)
        axislegend(ax3)


        ax4 = Makie.Axis(fig[4, 1])
        lines!(ax4, bgp, abs.(μ_ukf .- μ_kf), color=:black, label="Difference Ukf-kf")
        ylims!(0.0, 0.00001)
        axislegend(ax4; position=:lt)
        axislegend(ax4)


        fig
    end
end

end
