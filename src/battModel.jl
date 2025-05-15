module battModel

using ForwardDiff
using LinearAlgebra
using DataFrames
using AbstractGPs
using LinearAlgebra
using ComponentArrays
using ..RecursiveGPs

export BATTModel, battery_learn!, battery_learn_rc!
"""
Battery model and battery model training functions
    - BATTModel struct: Battery model with 1  rgp ocv, 1 rgp R0 and 1/2 RC, default 1
    - battery_learn!: Train battery model without RC
    - battery_learn_rc!: Train battery model with one RC (with 2 RC not implemented yet)
    - battery_learn_dual_kf!: Train battery model with one RC and RC parameters
"""
mutable struct BATTModel
    """
    Battery model struct:
        - Current implementation enables changing noise and RC parameters
        - Number of RC not tuneable once the struct has been Build
        - In case no RC parameters/set to false or dummy, or wrong specification one RC is assumed with R = 15e-3 and τ = 60
    """

    rgp_ocv::RecursiveGPs.RGPModel
    rgp_r::RecursiveGPs.RGPModel
    μ::Vector{Float64}
    Σ::Matrix{Float64}

    μ_params::Vector{Float64}
    Σ_params::Matrix{Float64}

    model_noise::ComponentArray
    param_noise::ComponentArray

    i::Float64
    dt::NamedTuple

    function BATTModel(
        rgp_ocv,
        rgp_r,
        μ_params,
        dt
    )

        ## Checking number of RCs
        if size(μ_params, 1) == 2
            N_rc = 1
        elseif size(μ_params, 1) == 4
            @warn "2 RC functions not implemented yet: Set first RC only"
            μ_params = μ_params[1:2]
            N_rc = 1
        else
            @warn "No RC specified or wrong initialization, Set one RC with R = 15e-3 and τ = 60"
            N_rc = 1
            μ_params = [15e-4, 60.0]
        end

        Σ_params = 1e-6 .* I(length(μ_params))

        ## Initializing model
        filler = zeros(size(rgp_ocv.μ, 1), size(rgp_r.μ, 1))
        filler_rc = zeros(N_rc, size(rgp_ocv.μ, 1) + size(rgp_r.μ, 1))

        μ = [rgp_ocv.μ; rgp_r.μ; zeros(N_rc)]

        Σ = vcat(
            [rgp_ocv.Σ filler zeros(size(rgp_ocv.Σ, 1), N_rc)],
            [filler' rgp_r.Σ zeros(size(rgp_r.Σ, 1), N_rc)],
            [filler_rc 1e-4 * I(N_rc)],
        )

        ### Building Model Noise
        # w is motion model noise, only affecting RC
        # v is measurement model noise, affection all
        N_state = size(μ, 1)
        N_basis = size(rgp_ocv.μ, 1) + size(rgp_r.μ, 1)
        N_out = 1

        model_noise = ComponentArray(
            w=ComponentArray(
                μ=zeros(1, N_state)),
            Σ=vcat(
                [zeros(N_basis, N_basis) zeros(N_basis, 1)],
                [zeros(N_rc, N_basis) 10e-6 .* I(N_rc)],
            ),
            v=ComponentArray(
                μ=zeros(1, N_out),
                Σ=0.1 .* I(N_out)
            ),
        )

        ## Building param_noise
        w_noise_R = 1e-6
        w_noise_τ = 1e-4

        param_noise = ComponentArray(
            w=ComponentArray(
                μ=zeros(1, N_rc),
                Σ=kron(
                    Diagonal(ones(N_rc)),
                    vcat(
                        [w_noise_R 0],
                        [0 w_noise_τ]
                    )
                )
            ),
            v=ComponentArray(
                μ=zeros(1, N_out),
                Σ=1e-3 .* I(N_out)
            )
        )

        i = 0.0
        new(rgp_ocv, rgp_r, μ, Σ, μ_params, Σ_params, model_noise, param_noise, i, dt)
    end
end



############ WITH RC
function inference_step(batt)
    ts = 1## PLaceholder for PROFILE 2 Dataset, to be changed for new data
    N_basis = size(batt.rgp_ocv.X_basis, 1) + size(batt.rgp_ocv.X_basis, 1)

    R1 = batt.μ_params[1]
    τ1 = batt.μ_params[2]

    A_batt = vcat(
        [I(N_basis) zeros(N_basis, 2)],
        [zeros(1, N_basis) exp(-ts / (τ1))],
    )

    B_batt = [
        zeros(N_basis);
        R1 * (1 - exp(-ts / (τ1)))
    ]


    μ_predict = A_batt * batt.μ + B_batt * batt.i + batt.model_noise.w.μ
    Σ_predict = A_batt * batt.Σ * A_batt' + batt.model_noise.w.Σ
    return (
        μ=μ_predict,
        Σ=Σ_predict,
    )
end


function update_step!(batt, batch, μ_predict, Σ_predict, dt)
    """
    Performs join state estimation of ocv and R0
    """
    ## Update step
    Σ = Σ_predict
    μ = μ_predict

    ## OCV
    ocv = RecursiveGPs.predict(batt.rgp_ocv, batch.x.soc)
    ocv_v = StatsBase.reconstruct(dt.v, ocv.μ)
    ocv_σ = sqrt.(diag(ocv.Σ))
    ocv_σ = StatsBase.reconstruct(dt.σ, ocv_σ)[1]

    ## R0
    R0 = RecursiveGPs.predict(batt.rgp_r, batch.x.soc)
    R0_v = StatsBase.reconstruct(dt.v, R0.μ)
    R0_σ = sqrt.(diag(R0.Σ))
    R0_σ = StatsBase.reconstruct(dt.σ, R0_σ)[1]


    e = batch.y - (ocv_v + batch.x.i .* R0_v .+ μ[end] + batt.model_noise.v.μ)

    H_ocv = dt.v.scale .* cov(batt.rgp_ocv.gp, batch.x.soc, batt.rgp_ocv.X_basis) * batt.rgp_ocv.inv_cov
    H_r0 = dt.v.scale .* batch.x.i .* cov(batt.rgp_r.gp, batch.x.soc, batt.rgp_r.X_basis) * batt.rgp_r.inv_cov
    H_rc1 = 1

    H = [H_ocv H_r0 H_rc1]

    S = H * Σ * H' + (batch.x.i .^ 2 .* R0_σ .^ 2 + ocv_σ .^ 2 .+ batt.model_noise.v.Σ) * I(size(batch.y, 1))

    Gk = Σ * H' * inv(S)

    new_μ = μ + Gk * (e)
    new_Σ = Σ - Gk * H * Σ

    ## Updating model with new parameter
    ocv_start = 1
    ocv_end = size(batt.rgp_ocv.μ)[1]

    r_start = ocv_end + 1
    r_end = ocv_end + size(batt.rgp_r.μ)[1]

    batt.rgp_ocv.μ = new_μ[
        ocv_start:ocv_end
    ]
    batt.rgp_ocv.Σ = new_Σ[
        ocv_start:ocv_end,
        ocv_start:ocv_end
    ]

    batt.rgp_r.μ = new_μ[
        r_start:r_end
    ]
    batt.rgp_r.Σ = new_Σ[
        r_start:r_end,
        r_start:r_end
    ]

    batt.Σ = new_Σ
    batt.μ = new_μ

    batt.i = batch.x.i[1]
end

function battery_learn_rc!(batt, batch)
    """
    Performs join state estimation of ocv, R0, Vrc1, Vrc2
    """
    μ_predict, Σ_predict = inference_step(batt)
    update_step!(batt, batch, μ_predict, Σ_predict, dt)

    Vrc1 = batt.μ[end-1]
    Vrc2 = batt.μ[end]
    return Vrc1, Vrc2
end


####### DUAL KF WITH RC

function predict_params(batt)
    """
    Predicts parameters of RC
    """

    μ_params_predict = batt.μ_params + batt.param_noise.w.μ
    Σ_params_predict = batt.Σ_params + batt.param_noise.w.Σ
    return μ_params_predict, Σ_params_predict

end


function update_params!(batt, batch, μ_params_predict, Σ_params_predict)
    """
    Update parameters of RC
    """
    ts = 1
    R0 = 15e-3

    ocv = RecursiveGPs.predict(batt.rgp_ocv, batch.x.soc)
    ocv_v = StatsBase.reconstruct(batt.dt.v, ocv.μ)

    e = batch.v - (ocv_v .+ R0 * batch.x.i .+ new_μ[end] + batt.param_noise.v.μ)

    R1 = μ_params_predict[1]
    τ1 = μ_params_predict[2]

    HR1 = (1 - exp(-ts / (τ1))) * i

    Hτ1 = exp(-ts / τ1) * ts / (τ1^2) * ([old_μ[end]] .- R1 * i)

    H = [HR1 Hτ1]




    S = H * Σ_params_predict * H' + (batt.param_noise.v.Σ) * I(size(batch.v, 1))
    Gk = Σ_params_predict * H' * inv(S)
    new_μ_params = μ_params_predict + Gk * (e)
    new_Σ_params = Σ_params_predict - Gk * H * Σ_params_predict


    batt.μ_params = new_μ_params
    batt.Σ_params = new_Σ_params
end


function battery_learn_dual_kf!(batt, batch, dt)
    """
    Performs dual KF of ocv,R0, Vrc1 and RC parameters
    """
    ### Updating model
    μ_predict, Σ_predict = inference_step(batt)
    update_step!(batt, batch, μ_predict, Σ_predict, dt)

    ## Updating parameters
    μ_params_predict, Σ_params_predict = predict_params(batt)
    update_params!(batt, batch, μ_params_predict, Σ_params_predict)
end




###### WITHOUT RC
function battery_learn!(batt, batch)
    """
    Performs join state estimation of ocv and R0
    """
    ## Only update step
    ## Motion model os null Extracting means and cov
    N_basis = size(batt.rgp_ocv.μ, 1) + size(batt.rgp_ocv.μ, 1)
    Σ = batt.Σ[1:N_basis, 1:N_basis]
    μ = batt.μ[1:N_basis]

    ocv = RecursiveGPs.predict(batt.rgp_ocv, batch.x.soc)
    r0 = RecursiveGPs.predict(batt.rgp_r, batch.x.soc)
    e = batch.y - (ocv.μ + batch.x.i .* r0.μ + batt.model_noise.v.μ)

    H1 = cov(batt.rgp_ocv.gp, batch.x.soc, batt.rgp_ocv.X_basis) * batt.rgp_ocv.inv_cov
    H2 = batch.x.i .* cov(batt.rgp_r.gp, batch.x.soc, batt.rgp_r.X_basis) * batt.rgp_r.inv_cov
    H = [H1 H2]

    S = H * Σ * H' + (batch.x.i .^ 2 .* r0.Σ + ocv.Σ .+ batt.model_noise.v.Σ) * I(size(batch.y, 1))

    Gk = Σ * H' * inv(S)

    new_μ = μ + Gk * (e)
    new_Σ = Σ - Gk * H * Σ

    ## Updating model
    size_ocv = size(batt.rgp_ocv.μ)[1]
    size_r = size_ocv + 1

    batt.rgp_ocv.μ = new_μ[1:size_ocv]
    batt.rgp_ocv.Σ = new_Σ[1:size_ocv, 1:size_ocv]

    batt.rgp_r.μ = new_μ[size_r:end]
    batt.rgp_r.Σ = new_Σ[size_r:end, size_r:end]

    batt.Σ[1:N_basis, 1:N_basis] = new_Σ
    batt.μ[1:N_basis] = new_μ


    return
end



end
