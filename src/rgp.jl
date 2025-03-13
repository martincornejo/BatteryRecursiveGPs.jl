module RegressionGP
using LinearAlgebra
using ComponentArrays
using KernelFunctions

export RGPModel, rgp_learn, predict

mutable struct RGPModel
    kernel::Kernel
    σ::Float64
    X_basis::Vector{Float64}
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    prior_μ::Vector{Float64}
    inv_cov::Matrix{Float64}
    mean_function::Function

    function RGPModel(kernel, σ, X_basis; mean_function::Function=x -> 0.0)
        μ = mean_function.(X_basis)
        Σ = kernelmatrix(kernel, X_basis)
        prior_μ = mean_function.(X_basis)
        inv_cov = inv(Σ)
        ## params, kernelc = Flux.destructure(kernel) For future with parameter hypertuning
        new(kernel, σ, X_basis, μ, Σ, prior_μ, inv_cov, mean_function)
    end
end

function inference_step(rgp::RGPModel, H, X_batch)
    """
    Inference step at batch points
    """
    μ_predict = rgp.mean_function.(X_batch) + H * (rgp.μ - rgp.prior_μ) #eq.6 

    R = kernelmatrix(rgp.kernel, X_batch) - H * kernelmatrix(rgp.kernel, rgp.X_basis, X_batch) #eq.7 
    Σ_predict = R + H * rgp.Σ * H' #eq.9

    return (
        μ=μ_predict,
        Σ=Σ_predict
    )
end

function update_step(rgp::RGPModel, predict_batch, H, Y_batch)
    """
    Update rgp parameters
    """
    Gk = rgp.Σ * H' * inv(predict_batch.Σ + rgp.σ^2 * I(size(Y_batch, 1))) #eq.12

    new_μ = rgp.μ + Gk * (Y_batch - predict_batch.μ) #eq.10
    new_Σ = rgp.Σ - Gk * H * rgp.Σ #eq.11

    rgp.μ = new_μ
    rgp.Σ = new_Σ

end

function rgp_learn(rgp::RGPModel, dataLoader)
    """ 
    Performs RGP learning
    Inputs:
        - rgp model 
        - dataLoader: Data already structured so is fast to iterate

    Note:
        - Inference and update steps separable at the moment for future when switch between Hyp or non-Hyp
    """
    for batch in dataLoader
        X_batch, Y_batch = batch

        ## Observation matrix
        H = kernelmatrix(rgp.kernel, X_batch, rgp.X_basis) * rgp.inv_cov

        ## Predict value
        predict_batch = inference_step(rgp, H, X_batch)

        ## Update model by predicted value error
        update_step(rgp, predict_batch, H, Y_batch)
    end

end

function predict(rgp, X_predict)
    """
    Does a prediction using a posterior at X_predict
    """
    H = kernelmatrix(rgp.kernel, X_predict, rgp.X_basis) * rgp.inv_cov
    μ_predict, Σ_predict = inference_step(rgp, H, X_predict)
    return μ_predict, Σ_predict
end
end

