module kf_utils

using LinearAlgebra
using DataFrames
using ComponentArrays
using StatsBase
using ..RecursiveGPs

export inference_step, inference_step_R, predict_params,
    update_step!, update_step_R!, update_step_no_rc!,
    update_params!, adaptive_extended_kf!, push_rts!, empty_rts!,
    test_inference_step, test_update_step!



########### Inference ##############


function inference_step(batt)
    """
    Inference step of Kalman filter with one RC
    """
    ts = 1## PLaceholder for PROFILE 2 Dataset, to be changed for new data
    N_basis = size(batt.rgp_ocv.X_basis, 1) + size(batt.rgp_r.X_basis, 1)

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


function test_inference_step(batt)
    """
    Inference step of Kalman filter with but  R0 exact
    """
    ts = 1## PLaceholder for PROFILE 2 Dataset, to be changed for new data
    N_basis = size(batt.rgp_ocv.X_basis, 1)

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




function predict_params(batt)
    """
    Predicts parameters of RC
    """

    μ_params_predict = batt.μ_params + batt.param_noise.w.μ
    Σ_params_predict = batt.Σ_params + batt.param_noise.w.Σ
    return μ_params_predict, Σ_params_predict

end




############# Update ###################
function update_step_no_rc!(batt, batch)
    """
    Update step without RC
    """
    dt = batt.dt
    N_basis = size(batt.rgp_ocv.μ, 1) + size(batt.rgp_r.μ, 1)
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

    S = H * Σ * H' + (batt.model_noise.v.Σ) * I(size(batch.y, 1))

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
end



function update_step!(batt, batch, μ_predict, Σ_predict)
    """
    Performs join state estimation of ocv, R0 and Vrv
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


function correct_update_step!(batt, batch, old_μ, old_i, μ_params_predict, Σ_params_predict)
    """
    This update step, updates teh RC parameters as joint, no need of Voltage really
    """
    N_state = size(batt.μ)
    N_params = size(batt.μ_params)

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
    HR1 = (1 - exp(-ts / (τ1))) * old_i
    Hτ1 = exp(-ts / τ1) * ts / (τ1^2) * ([old_μ[end]] .- R1 * old_i)


    H = hcat(H_ocv, H_r0, H_rc1, HR1, Hτ1)
    Σ_kf = vcat(
        hcat(Σ_predict, zeros(1, size(μ_params_predict, 1))),
        hcat(zeros(2, size(Σ_predict, 1)), Σ_params_predict)
    )

    μ_kf = vcat(
        μ,
        μ_params_predict)

    S = H * Σ_kf * H' + (batt.model_noise.v.Σ) * I(size(batch.y, 1))

    Gk = Σ_kf * H' * inv(S)

    new_μ_kf = μ_predict + Gk * (e)
    new_Σ_kf = Σ_kf - Gk * H * Σ_kf

    new_μ = new_μ_kf[1:N_state]
    new_μ_params = new_μ_kf[1:N_params]

    new_Σ = new_Σ_kf[1:N_state, 1:N_state]
    new_Σ_params = new_Σ_kf[1:N_params, 1:N_params]

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

    N_state = size(batt.μ)
    N_params = size(batt.μ_params)

    batt.μ = new_μ
    batt.Σ = new_Σ
    batt.μ_params = new_μ_params
    batt.Σ_params = new_Σ_params

    batt.i = batch.x.i[1]
end

function test_update_step!(batt, batch, μ_predict, Σ_predict)
    """
    Performs join state estimation of Vrc with perfect R0 and ocv
    """
    dt = batt.dt

    ## OCV
    ocv = RecursiveGPs.predict(batt.rgp_ocv, batch.x.soc)
    ocv_v = StatsBase.reconstruct(dt.v, ocv.μ)

    ## R0
    r0 = 15e-3


    e = batch.y - (ocv_v + batch.x.i .* r0 .+ μ_predict[end] + batt.model_noise.v.μ)

    H_ocv = dt.v.scale .* cov(batt.rgp_ocv.gp, batch.x.soc, batt.rgp_ocv.X_basis) * batt.rgp_ocv.inv_cov
    H_rc1 = 1

    H = hcat(H_ocv, H_rc1)

    S = H * Σ_predict * H' + (batt.model_noise.v.Σ) * I(size(batch.y, 1))

    Gk = Σ_predict * H' * inv(S)

    new_μ = μ_predict + Gk * (e)
    new_Σ = Σ_predict - Gk * H * Σ_predict


    ocv_start = 1
    ocv_end = size(batt.rgp_ocv.μ)[1]

    batt.rgp_ocv.μ = new_μ[
        ocv_start:ocv_end
    ]
    batt.rgp_ocv.Σ = new_Σ[
        ocv_start:ocv_end,
        ocv_start:ocv_end
    ]
    ## Updating model with new parameter

    batt.μ = vec(new_μ)
    batt.Σ = new_Σ
    batt.i = batch.x.i[1]

end


function update_params!(batt, batch, old_μ, old_i, μ_params_predict, Σ_params_predict; model_R=true)
    """
    Update parameters of RC
    """
    ts = 1
    dt = batt.dt

    ## OCV
    if model_R == true
        ocv = RecursiveGPs.predict(batt.rgp_ocv, batch.x.soc)
        ocv_v = ocv.μ
        r0 = RecursiveGPs.predict(batt.rgp_r, batch.x.soc)
        r0_v = StatsBase.reconstruct(dt.σ, r0.μ)
    else
        ocv_v = RecursiveGPs.predict(batt.rgp_ocv, batch.x.soc).μ
        r0_v = 15e-3
    end
    ocv_v = StatsBase.reconstruct(dt.v, ocv_v)


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





#### Adaptive noise
function adaptive_extended_kf!(batt, batch, old_Σ_params, Gk, H, e; b=0.99)

    d = (1 - b) / (1 - b^(batch.x.t[1] + 1))
    A = I


    new_v = ComponentArray(
        μ=batt.param_noise.v.μ,
        Σ=batt.param_noise.v.Σ
    )


    new_w = ComponentArray(
        μ=batt.param_noise.w.μ,
        Σ=(1 - d) * batt.param_noise.w.Σ +
          d * (Gk * e * e' * Gk')
    )

    batt.param_noise.w = new_w
    batt.param_noise.v = new_v

end

end

