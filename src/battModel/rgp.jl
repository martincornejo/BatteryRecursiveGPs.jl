module RecursiveGPs
using LinearAlgebra
using ComponentArrays
using AbstractGPs

export RGPModel, learn!, predict
"""
RGP model 
"""
mutable struct RGPModel
    gp::GP
    σ::Float64
    X_basis::Vector{Float64}
    μ::Vector{Float64}
    Σ::Matrix{Float64}

    # Fixed
    prior_μ::Vector{Float64}
    inv_cov::Matrix{Float64}

    function RGPModel(gp, σ, X_basis)
        μ = mean_vector(gp.mean, X_basis)
        Σ = cov(gp, X_basis) + 1e-6 * I

        prior_μ = μ
        inv_cov = inv(Σ)

        new(gp, σ, X_basis, μ, Σ, prior_μ, inv_cov)
    end
end

function predict(rgp::RGPModel, X_batch; train=true)
    """
    Inference step at batch points.
    Two modes:
        train_mode = True -> Does not include noise since is already on update step
        train_mode = False -> Includes GP noise 
    """

    H = cov(rgp.gp, X_batch, rgp.X_basis) * rgp.inv_cov


    μ_predict = mean_vector(rgp.gp.mean, X_batch) + H * (rgp.μ - rgp.prior_μ) #eq.6 +

    R = cov(rgp.gp, X_batch) - H * cov(rgp.gp, rgp.X_basis, X_batch) #eq.7 
    Σ_predict = R + H * rgp.Σ * H' #eq.9

    if train == false
        Σ_predict += rgp.σ^2 * I
    end

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
    predict_batch = predict(rgp, X_batch, train=true)

    ## Update model by predicted value error
    update_step!(rgp, predict_batch, H, Y_batch)

end




end