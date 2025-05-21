module battModel

using ForwardDiff
using LinearAlgebra
using DataFrames
using AbstractGPs
using StatsBase
using LinearAlgebra
using ComponentArrays
using ..RecursiveGPs

export BATTModel, battery_learn!, battery_learn_rc!, battery_learn_dual_kf!
"""
Battery model and battery model training functions
    - BATTModel struct: Battery model with 1  rgp ocv, 1 rgp R0 and 1/2 RC, default 1
    - battery_learn!: Train battery model without RC
    - battery_learn_rc!: Train battery model with one RC (with 2 RC not implemented yet)
    - battery_learn_dual_kf!: Train battery model with one RC and RC parameters
    - Note: update_step_R!, inference_step_R functions are temporals functions for testing, as well as model_R inputs
"""
mutable struct BATTModel
    """
    Battery model struct:
        - Current implementation enables changing noise and RC parameters
        - Number of RC not tuneable once the struct has been Build
        - In case no RC parameters/set to false or dummy, or wrong specification one RC is assumed with R = 15e-3 and τ = 60
        - model_R and rgp_ocv::Any: Temporal code for testing purposes, to be deleted once adaptive_extended_kf works with RGPs.
    """

    rgp_ocv::Any
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
        dt;
        model_R=true
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
            μ_params = [15e-3; 60.0]
        end

        Σ_params = 1e-6 .* I(length(μ_params))

        if model_R
            ## Initializing model
            filler = zeros(size(rgp_ocv.μ, 1), size(rgp_r.μ, 1))
            filler_rc = zeros(N_rc, size(rgp_ocv.μ, 1) + size(rgp_r.μ, 1))

            μ = vcat(rgp_ocv.μ, rgp_r.μ, zeros(N_rc))

            Σ = vcat(
                hcat(rgp_ocv.Σ, filler, zeros(size(rgp_ocv.Σ, 1), N_rc)),
                hcat(filler', rgp_r.Σ, zeros(size(rgp_r.Σ, 1), N_rc)),
                hcat(filler_rc, 1e-4 * I(N_rc)),
            )

            ### Building Model Noise
            # w is motion model noise, only affecting RC
            # v is measurement model noise, affection all
            N_state = size(μ, 1)
            N_basis = size(rgp_ocv.μ, 1) + size(rgp_r.μ, 1)
            N_out = 1
            N_params = size(μ_params)

        else

            μ = vcat(zeros(N_rc))

            Σ = vcat(
                hcat(1e-4 * I(N_rc)),
            )

            N_state = size(μ, 1)
            N_basis = 0
            N_out = 1
            N_params = size(μ_params)
        end

        model_noise = ComponentArray(
            w=ComponentArray(
                μ=zeros(N_state),
                Σ=vcat(
                    hcat(zeros(N_basis, N_basis), zeros(N_basis, 1)),
                    hcat(zeros(N_rc, N_basis), 10e-6 .* I(N_rc)),
                )
            ),
            v=ComponentArray(
                μ=zeros(N_out),
                Σ=0.1 .* I(N_out)
            ),
        )

        ## Building param_noise
        w_noise_R = 1e-9
        w_noise_τ = 1e-4

        param_noise = ComponentArray(
            w=ComponentArray(
                μ=zeros(N_params),
                Σ=kron(
                    Diagonal(ones(N_rc)),
                    vcat(
                        hcat(w_noise_R, 0),
                        hcat(0, w_noise_τ)
                    )
                )
            ),
            v=ComponentArray(
                μ=zeros(N_out),
                Σ=1e-5 .* I(N_out)
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
        hcat(I(N_basis), zeros(N_basis, 1)),
        hcat(zeros(1, N_basis), exp(-ts / (τ1))),
    )

    B_batt = vcat(
        zeros(N_basis),
        R1 * (1 - exp(-ts / (τ1)))
    )

    μ_predict = A_batt * batt.μ + B_batt * batt.i + batt.model_noise.w.μ
    Σ_predict = A_batt * batt.Σ * A_batt' + batt.model_noise.w.Σ
    return (
        μ=μ_predict,
        Σ=Σ_predict,
    )
end


function update_step!(batt, batch, μ_predict, Σ_predict)
    """
    Performs join state estimation of ocv and R0
    """
    dt = batt.dt

    ## OCV
    ocv = RecursiveGPs.predict(batt.rgp_ocv, batch.x.soc)
    ocv_v = StatsBase.reconstruct(dt.v, ocv.μ)
    ocv_Σ = ocv.Σ .* (dt.v.scale .^ 2)

    ## R0
    r0 = RecursiveGPs.predict(batt.rgp_r, batch.x.soc)
    r0_v = StatsBase.reconstruct(dt.σ, r0.μ)
    r0_Σ = r0.Σ .* (dt.σ.scale .^ 2)


    e = batch.y - (ocv_v + batch.x.i .* r0_v .+ μ_predict[end] + batt.model_noise.v.μ)

    H_ocv = dt.v.scale .* cov(batt.rgp_ocv.gp, batch.x.soc, batt.rgp_ocv.X_basis) * batt.rgp_ocv.inv_cov
    H_r0 = dt.σ.scale .* batch.x.i .* cov(batt.rgp_r.gp, batch.x.soc, batt.rgp_r.X_basis) * batt.rgp_r.inv_cov
    H_rc1 = 1

    H = hcat(H_ocv, H_r0, H_rc1)

    S = H * Σ_predict * H' + (batt.model_noise.v.Σ) * I(size(batch.y, 1))

    Gk = Σ_predict * H' * inv(S)

    new_μ = μ_predict + Gk * (e)
    new_Σ = Σ_predict - Gk * H * Σ_predict

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


    batt.μ = vec(new_μ)
    batt.Σ = new_Σ
    batt.i = batch.x.i[1]

end

function battery_learn_rc!(batt, batch; model_R=true)
    """
    Performs join state estimation of ocv, R0, Vrc1, Vrc2
    """

    if model_R
        μ_predict, Σ_predict = inference_step(batt)
        update_step!(batt, batch, μ_predict, Σ_predict)
    else
        μ_predict, Σ_predict = inference_step_R(batt)
        update_step_R!(batt, batch, μ_predict, Σ_predict)
    end

end
###############


############ WITH RC
function inference_step_R(batt)
    ts = 1## PLaceholder for PROFILE 2 Dataset, to be changed for new data
    N_basis = 0

    R1 = batt.μ_params[1]
    τ1 = batt.μ_params[2]

    A_batt = vcat(
        hcat(I(N_basis), zeros(N_basis, 1)),
        hcat(zeros(1, N_basis), exp(-ts / (τ1))),
    )

    B_batt = vcat(
        zeros(N_basis),
        R1 * (1 - exp(-ts / (τ1)))
    )

    μ_predict = A_batt * batt.μ + B_batt * batt.i + batt.model_noise.w.μ
    Σ_predict = A_batt * batt.Σ * A_batt' + batt.model_noise.w.Σ
    return (
        μ=μ_predict,
        Σ=Σ_predict,
    )
end


function update_step_R!(batt, batch, μ_predict, Σ_predict)
    """
    Performs join state estimation of ocv and R0
    """
    dt = batt.dt

    ## OCV
    ocv = batt.rgp_ocv(batch.x.soc)
    ocv_v = StatsBase.reconstruct(dt.v, ocv)

    ## R0
    r0 = 15e-3


    e = batch.y - (ocv_v + batch.x.i .* r0 .+ μ_predict[end] + batt.model_noise.v.μ)

    H_rc1 = 1

    H = [H_rc1]

    S = H * Σ_predict * H' + (batt.model_noise.v.Σ) * I(size(batch.y, 1))

    Gk = Σ_predict * H' * inv(S)

    new_μ = μ_predict + Gk * (e)
    new_Σ = Σ_predict - Gk * H * Σ_predict

    ## Updating model with new parameter

    batt.μ = vec(new_μ)
    batt.Σ = new_Σ
    batt.i = batch.x.i[1]

end

####### DUAL KF WITH RC PARAMETERS

function predict_params(batt)
    """
    Predicts parameters of RC
    """

    μ_params_predict = batt.μ_params + batt.param_noise.w.μ
    Σ_params_predict = batt.Σ_params + batt.param_noise.w.Σ
    return μ_params_predict, Σ_params_predict

end


function update_params!(batt, batch, old_μ, old_i, μ_params_predict, Σ_params_predict)
    """
    Update parameters of RC
    """
    ts = 1
    dt = batt.dt

    ## OCV
    #ocv = RecursiveGPs.predict(batt.rgp_ocv, batch.x.soc)
    ocv = batt.rgp_ocv(batch.x.soc)
    ocv_v = StatsBase.reconstruct(dt.v, ocv)

    ## R0
    r0 = RecursiveGPs.predict(batt.rgp_r, batch.x.soc)
    r0_v = StatsBase.reconstruct(dt.σ, r0.μ)


    e = batch.y - (ocv_v .+ r0_v .* batch.x.i .+ batt.μ[end] + batt.param_noise.v.μ)

    R1 = μ_params_predict[1]
    τ1 = μ_params_predict[2]

    HR1 = (1 - exp(-ts / (τ1))) * old_i

    Hτ1 = exp(-ts / τ1) * ts / (τ1^2) * ([old_μ[end]] .- R1 * old_i)

    H = hcat(HR1, Hτ1)


    S = H * Σ_params_predict * H' + (batt.param_noise.v.Σ) * I(size(batch.y, 1))
    Gk = Σ_params_predict * H' * inv(S)
    new_μ_params = μ_params_predict + Gk * (e)
    new_Σ_params = Σ_params_predict - Gk * H * Σ_params_predict


    batt.μ_params = new_μ_params
    batt.Σ_params = new_Σ_params

    return Gk, H, e
end


function battery_learn_dual_kf!(batt, batch; model_R=true, adaptive_noise=false)
    """
    Performs dual KF of ocv,R0, Vrc1 and RC parameters
    """
    old_μ = copy(batt.μ)
    old_i = copy(batt.i)



    if model_R == true
        μ_predict, Σ_predict = inference_step(batt)
        update_step!(batt, batch, μ_predict, Σ_predict)
    else
        μ_predict, Σ_predict = inference_step_R(batt)
        update_step_R!(batt, batch, μ_predict, Σ_predict)
    end
    ## Updating parameters
    μ_params_predict, Σ_params_predict = predict_params(batt)
    Gk, H, e = update_params!(batt, batch, old_μ, old_i, μ_params_predict, Σ_params_predict)

    if adaptive_noise == true
        adaptive_extended_kf!(batt, batch, μ_params_predict, Σ_params_predict, Gk, H, e)
    end
end

#### Adaptive noise
function adaptive_extended_kf!(batt, batch, μ_params_predict, Σ_params_predict, Gk, H, e; b=0.95)

    d = (1 - b) / (1 - b^(batch.x.t[1] + 1))
    A = I


    new_v = ComponentArray(
        μ=batt.param_noise.v.μ,
        Σ=(1 - d) * batt.param_noise.v.Σ +
          d * (e * e' - H * Σ_params_predict * H')
    )


    new_w = ComponentArray(
        μ=batt.param_noise.w.μ,
        Σ=(1 - d) * batt.param_noise.w.Σ +
          d * (Gk * e * e' * Gk')
    )

    batt.param_noise.w = new_w
    batt.param_noise.v = new_v

end



###### WITHOUT RC
function battery_learn!(batt, batch)
    """
    Performs join state estimation of ocv and R0
    Does only accept batt set up with rgp of ocv and rgp of R0
    """
    dt = batt.dt
    ## Only update step
    ## Motion model os null Extracting means and cov
    N_basis = size(batt.rgp_ocv.μ, 1) + size(batt.rgp_ocv.μ, 1)
    Σ = batt.Σ[1:N_basis, 1:N_basis]
    μ = batt.μ[1:N_basis]

    ## OCV
    ocv = RecursiveGPs.predict(batt.rgp_ocv, batch.x.soc)
    ocv_v = StatsBase.reconstruct(dt.v, ocv.μ)
    ocv_Σ = ocv.Σ .* (dt.v.scale .^ 2)

    ## R0
    r0 = RecursiveGPs.predict(batt.rgp_r, batch.x.soc)
    r0_v = StatsBase.reconstruct(dt.σ, r0.μ)
    r0_Σ = r0.Σ .* (dt.v.scale .^ 2)

    ## Error
    e = batch.y - (ocv_v + batch.x.i .* r0_v + batt.model_noise.v.μ)

    H_ocv = dt.v.scale .* cov(batt.rgp_ocv.gp, batch.x.soc, batt.rgp_ocv.X_basis) * batt.rgp_ocv.inv_cov
    H_R0 = dt.v.scale .* batch.x.i .* cov(batt.rgp_r.gp, batch.x.soc, batt.rgp_r.X_basis) * batt.rgp_r.inv_cov

    H = [H_ocv H_R0]

    S = H * Σ * H' + (batch.x.i .^ 2 .* r0_Σ + ocv_Σ .+ batt.model_noise.v.Σ) * I(size(batch.y, 1))

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
