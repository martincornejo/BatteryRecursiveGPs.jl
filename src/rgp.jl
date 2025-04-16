module RecursiveGPs
using LinearAlgebra
using ComponentArrays
using AbstractGPs

export RGPModel, learn!, predict

mutable struct RGPModel
    gp::GP
    σ::Float64
    X_basis::Vector{Float64}
    μ::Vector{Float64}
    Σ::Matrix{Float64}

    prior_μ::Vector{Float64}
    inv_cov::Matrix{Float64}
    inv_cov_noise::Matrix{Float64} # For testing purposes

    mean_function::Function

    function RGPModel(gp, σ, X_basis; mean_function::Function=x -> 0.0)
        μ = mean_function.(X_basis)
        Σ = cov(gp, X_basis) + σ^2 * I(size(X_basis, 1))
        Σ_noise = cov(gp, X_basis) + σ^2 * I(size(X_basis, 1))
        prior_μ = mean_function.(X_basis)

        inv_cov = inv(Σ)
        inv_cov_noise = inv(Σ_noise)

        new(gp, σ, X_basis, μ, Σ, prior_μ, inv_cov, inv_cov_noise, mean_function)
    end
end

function predict(rgp::RGPModel, X_batch)
    """
    Inference step at batch points.
    Two modes:
        train_mode = True -> Does not include noise since is already on update step
        train_mode = False -> Includes GP noise 
    """

    H = cov(rgp.gp, X_batch, rgp.X_basis) * rgp.inv_cov


    μ_predict = rgp.mean_function.(X_batch) + H * (rgp.μ - rgp.prior_μ) #eq.6 +

    R = cov(rgp.gp, X_batch) - H * cov(rgp.gp, rgp.X_basis, X_batch) #eq.7 
    Σ_predict = R + H * rgp.Σ * H' #eq.9


    return (
        μ=μ_predict,
        Σ=Σ_predict
    )

end

function update_step!(rgp::RGPModel, predict_batch, H, Y_batch)
    """
    Update rgp parameters
    """

    Gk = rgp.Σ * H' * inv(predict_batch.Σ + rgp.σ^2 * I(size(Y_batch, 1))) #eq.12

    new_μ = rgp.μ + Gk * (Y_batch - predict_batch.μ) #eq.10
    new_Σ = rgp.Σ - Gk * H * rgp.Σ #eq.11

    rgp.μ = new_μ
    rgp.Σ = new_Σ
    return
end

function learn!(rgp::RGPModel, X_batch, Y_batch)
    """ 
    Performs RGP learning
    Inputs:
        - rgp model 
        - dataLoader: Data already structured so is fast to iterate

    Note:
        - Inference and update steps separable at the moment for future when switch between Hyp or non-Hyp
    """

    H = cov(rgp.gp, X_batch, rgp.X_basis) * rgp.inv_cov
    ## Predict value
    predict_batch = predict(rgp, X_batch)

    ## Update model by predicted value error
    update_step!(rgp, predict_batch, H, Y_batch)

end




end