module RecursiveGPsHYP
using LinearAlgebra
using ComponentArrays
using KernelFunctions
using Flux: destructure

export RGPModel_HYP, learn!, predict
"""
Temporal RGP with HYP, To be updated and merged to RGP_Model
"""
mutable struct RGPModel_HYP
    kernelc
    params
    X_basis
    μ_z
    Σ_z
    prior_μ
    inv_cov
    mean_function

    function RGPModel_HYP(kernel, σ, X_basis; mean_function::Function=x -> 0.0)
        params, kernelc = destructure(kernel) ## Look for alternative package
        N_basis = size(X_basis, 1)
        N_params = size(params, 1)
        N_z = N_basis + N_params + 1
        μ = mean_function.(X_basis)
        Σ = kernelmatrix(kernel, X_basis)
        prior_μ = mean_function.(X_basis)
        inv_cov = inv(Σ)

        μ_z = vcat(
            μ,
            params, ## for non-negative negative values look solution (Ej.: Log)
            σ
        )

        Σ_z = vcat(
            [Σ zeros(N_basis, N_z - N_basis)],
            [zeros(N_params + 1, N_z - N_params - 1) I(N_params + 1)],
        )

        new(kernelc, params, X_basis, μ_z, Σ_z, prior_μ, inv_cov, mean_function)
    end
end


function draw_sigma_points(μ_η, Σ_η)
    """
    Draws sigma points by standard method
    """
    N_hyp = size(μ_η, 1)
    n_ηi = 2 * N_hyp + 1
    ηi = zeros(n_ηi, N_hyp)
    wi = zeros(n_ηi)

    wi[1] = 0.5
    ηi[1, :] = μ_η
    sqrt_cov = real(sqrt(N_hyp / (1 - wi[1]) * Σ_η))
    for i in 1:1:N_hyp

        ηi[i+1, :] = μ_η + sqrt_cov[:, i]
        ηi[i+N_hyp+1, :] = μ_η - sqrt_cov[:, i]

        wi[i+1] = (1 - wi[1]) / (2 * N_hyp)
        wi[i+N_hyp+1] = (1 - wi[1]) / (2 * N_hyp)
    end

    return ηi, wi
end

function draw_sigma_points_v2(μ_η, Σ_η)
    """
    Draw Sigma points from:
    Wan, Eric A., and Rudolph Van Der Merwe. "The unscented Kalman filter for nonlinear estimation." 
    Proceedings of the IEEE 2000 adaptive systems for signal processing, 
    communications, and control symposium (Cat. No. 00EX373). Ieee, 2000.
    """
    ### HYPERPAMETERS
    α = 1e-3 #Spread of sigma points, set to small value 1e-3
    β = 2 ## prior knowledge, set as 2 since optimal for Gaussian Distributions
    𝓀 = 0.0 ## Scaling parameter, set to 0 usually

    N_hyp = size(μ_η, 1)
    λ = α^2 * (N_hyp + 𝓀) - N_hyp

    n_ηi = 2 * N_hyp + 1
    ηi = zeros(n_ηi, N_hyp)
    wm = zeros(n_ηi) # Weight for mean
    wc = zeros(n_ηi) #Weights for cov

    wm[1] = λ / (N_hyp + λ)
    wc[1] = wm[1] + (1 - α^2 + β)
    ηi[:, 1] = μ_η
    sqrt_cov = cholesky((N_hyp + λ) * Σ_η).L'

    for i in 1:1:N_hyp

        ηi[i+1, :] = μ_η + sqrt_cov[:, i]
        ηi[i+N_hyp+1, :] = μ_η - sqrt_cov[:, i]

        wm[i+1] = 1 / (2 * (N_hyp + λ))
        wm[i+N_hyp+1] = 1 / (2 * (N_hyp + λ))

        wc[i+1] = wm[i+1]
        wc[i+N_hyp+1] = wm[i+N_hyp+1]
    end

end

function inference_step(rgp, kernel, X_batch)
    """
    Inference step at batch points
    """
    N_basis = size(rgp.X_basis, 1)
    N_batch = size(X_batch, 1)
    N_params = size(rgp.params, 1)
    N_z = N_basis + N_params + 1 ## INcluding noise
    X_basis = rgp.X_basis

    ## Computing Motion model noise and observation matrix
    H_sub(n) = kernelmatrix(rgp.kernelc(n) + LinearKernel(), X_batch, X_basis) * inv(kernelmatrix(rgp.kernelc(n) + LinearKernel(), X_basis) + I * 1e-6)

    b(n) = rgp.mean_function.(X_batch) - H_sub(n) * rgp.prior_μ
    B(n) = kernelmatrix(rgp.kernelc(n) + LinearKernel(), X_batch) - H_sub(n) * kernelmatrix(rgp.kernelc(n) + LinearKernel(), X_basis, X_batch)


    μ_w(n) = vcat(
        zeros(N_basis),
        zeros(N_params + 1),
        b(n)
    )

    Σ_w(n) = vcat(
        zeros(N_basis, N_z + N_batch),
        zeros(N_params + 1, N_z + N_batch),
        [zeros(N_batch, N_z) B(n)]
    )

    H_t(n) = vcat(
        [I(N_basis) zeros(N_basis, N_params + 1)],
        [zeros(N_params + 1, N_basis) I(N_params + 1)],
        [H_sub(n) zeros(N_batch, N_params + 1)]
    )

    # Getting distributions needed from z
    μ_η = rgp.μ_z[N_basis+1:end]
    Σ_η = rgp.Σ_z[N_basis+1:end, N_basis+1:end]

    Σ_gη = rgp.Σ_z[1:N_basis, N_basis+1:end]

    μ_g = rgp.μ_z[1:N_basis]
    Σ_g = rgp.Σ_z[1:N_basis, 1:N_basis]

    # Drawing Sigma points
    ηi, wi = draw_sigma_points(μ_η, Σ_η)

    # Approximation bia Sigma points
    N_ηi = size(ηi, 1)
    S_t = Σ_gη * inv(Σ_η)
    μ_predict_i = zeros(N_ηi, N_z + N_batch)
    Σ_predict_i = zeros(N_ηi, N_z + N_batch, N_z + N_batch)

    for i in 1:1:N_ηi
        temp_μ = vcat(
            μ_g + S_t * (ηi[i, :] - μ_η),
            ηi[i, :]
        )

        ## BUilding matrices for each sigma points
        H_ti = H_t(ηi[i, 1:2])
        μ_wi = μ_w(ηi[i, 1:2])
        Σ_wi = Σ_w(ηi[i, 1:2])

        μ_predict_i[i, :] = H_ti * temp_μ + μ_wi

        temp_Σ = vcat(
            [Σ_g - S_t * Σ_gη' zeros(N_basis, N_params + 1)],
            [zeros(N_params + 1, N_basis) zeros(N_params + 1, N_params + 1)]
        )
        Σ_predict_i[i, :, :] = H_ti * temp_Σ * H_ti' + Σ_wi
    end

    μ_predict = zeros(N_z + N_batch, 1) #dropdims(sum(wi .* μ_predict_i, dims=1), dims=1)
    Σ_predict = zeros(size(μ_predict, 1), size(μ_predict, 1))

    ## Update mean
    for i in 1:1:N_ηi
        μ_predict = μ_predict + wi[i] * μ_predict_i[i, :]
    end

    ## Update cov
    for i in 1:1:N_ηi
        temporal = (μ_predict_i[i, :] - μ_predict) * (μ_predict_i[i, :] - μ_predict)' + Σ_predict_i[i, :, :]
        Σ_predict = Σ_predict + wi[i] * temporal
    end


    return (μ_p=μ_predict,
        Σ_p=Σ_predict)
end



function update_observable(obs, Y_batch)
    """
    Updates the observable state of the posterior
    """
    ## Updating
    μ_y = obs.μ_o[2:end]
    Σ_y = obs.Σ_o[2:end, 2:end] .+ obs.Σ_o[1, 1] .+ obs.μ_o[1]^2
    Gt = obs.Σ_o[:, 2:end] * inv(Σ_y)

    new_μ_o = obs.μ_o + Gt * (Y_batch - μ_y)
    new_Σ_o = obs.Σ_o - Gt * Σ_y * Gt'


    return (μ_o=new_μ_o,
        Σ_o=new_Σ_o)
end

function update_unobservable(new_obs, obs, un, L)
    """
    Updates the update unobservable state of the posterior
    """

    new_un_μ = un.μ_u + L * (
        new_obs.μ_o - obs.μ_o
    )

    new_un_Σ = un.Σ_u + L * (
                            new_obs.Σ_o - obs.Σ_o
                        ) * L'

    return (μ_u=new_un_μ,
        Σ_u=new_un_Σ)
end


function update_step!(rgp, predict_batch, Y_batch)
    """
    Update rgp parameters
    """
    N_basis = size(rgp.X_basis, 1)
    N_z = size(rgp.μ_z, 1)
    ## Dividing observable and unobserved state
    # Observable state is σ,gt
    obs = (
        μ_o=predict_batch.μ_p[N_z:end],
        Σ_o=predict_batch.Σ_p[N_z:end, N_z:end]
    )

    # unobservable state is g,kernel_parameters
    un = (
        μ_u=predict_batch.μ_p[1:N_z-1],
        Σ_u=predict_batch.Σ_p[1:N_z-1, 1:N_z-1]
    )

    Σ_uo = predict_batch.Σ_p[1:N_z-1, N_z:end]

    L = Σ_uo * inv(obs.Σ_o)

    ## Updating states
    new_obs = update_observable(obs, Y_batch)
    new_un = update_unobservable(new_obs, obs, un, L)
    ## Reconstructing
    h = zeros(size(new_obs.μ_o))
    h[1] = 1


    new_z_μ = vcat(
        new_un.μ_u,
        h' * new_obs.μ_o
    )
    new_z_Σ = vcat(
        [new_un.Σ_u L * new_obs.Σ_o * h],
        [h' * new_obs.Σ_o * L' h' * new_obs.Σ_o * h]
    )

    new_params = new_z_μ[N_basis+1:end-1]

    rgp.μ_z = new_z_μ
    rgp.Σ_z = new_z_Σ
    rgp.params = new_params

end

function learn!(rgp, X_batch, Y_batch)
    """ 
    Performs RGP learning
    """

    ## Construct the kernel function
    kernel = rgp.kernelc(rgp.params) + LinearKernel()
    ## Predict value
    predict_batch = inference_step(rgp, kernel, X_batch)
    ## Update model by predicted value error
    update_step!(rgp, predict_batch, Y_batch)

end

function predict(rgp, X_predict)
    ## Construct the kernel function
    kernel = rgp.kernelc(rgp.params) + LinearKernel()
    #N_z = size(rgp.μ_z, 1)
    N_basis = size(rgp.X_basis, 1)
    σ = rgp.μ_z[end]

    H = kernelmatrix(kernel, X_predict, rgp.X_basis) * inv(kernelmatrix(kernel, rgp.X_basis) + σ^2 * I)
    μ = rgp.mean_function.(X_predict) + H * (rgp.μ_z[1:N_basis] - rgp.prior_μ)

    R = kernelmatrix(kernel, X_predict) - H * kernelmatrix(kernel, rgp.X_basis, X_predict)
    Σ = R + H * rgp.Σ_z[1:N_basis, 1:N_basis] * H'


    ## Predict value
    #predict_batch = inference_step(rgp, kernel, X_predict)
    #μ = predict_batch.μ_p[N_z+1:end]
    #Σ = predict_batch.Σ_p[N_z+1:end, N_z+1:end]
    return μ, Σ
end
end