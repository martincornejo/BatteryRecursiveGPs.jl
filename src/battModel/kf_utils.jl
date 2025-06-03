module kf_utils

using LinearAlgebra
using DataFrames
using ComponentArrays
using StatsBase
using ..RecursiveGPs

export inference_step, inference_step_R, predict_params,
    update_step!, update_step_R!, update_step_no_rc!,
    update_params!, adaptive_extended_kf!, push_rts!, empty_rts!,
    test_inference_step, test_update_step!, update_step_joint_state!,
    join_inference_step, update_step_joint_state!



########### Inference ##############


function inference_step(batt)
    """
    Inference step of Kalman filter with one RC
    """
    ts = 1## PLaceholder for PROFILE 2 Dataset, to be changed for new data
    N_basis = size(batt.rgp_ocv.X_basis, 1) + size(batt.rgp_r.X_basis, 1)

    R1 = batt.μ_params[1]
    τ1 = batt.μ_params[2]

    A = vcat(
        hcat(I(N_basis), zeros(N_basis, 1)),
        hcat(zeros(1, N_basis), exp(-ts / (τ1))),
    )

    B = vcat(
        zeros(N_basis),
        R1 * (1 - exp(-ts / (τ1)))
    )

    μ_predict, Σ_predict = kf_inference(A, B, batt.μ, batt.Σ, batt.i, batt.model_noise.w.μ)
    return (
        μ=μ_predict,
        Σ=Σ_predict,
    )
end


function test_inference_step(batt)
    """
    Inference step of Kalman filter with but  R0 exact
    """
    ts = batt.ts
    N_basis = size(batt.rgp_ocv.X_basis, 1)

    R1 = batt.μ_params[1]
    τ1 = batt.μ_params[2]

    A = vcat(
        hcat(I(N_basis), zeros(N_basis, 1)),
        hcat(zeros(1, N_basis), exp(-ts / (τ1))),
    )

    B = vcat(
        zeros(N_basis),
        R1 * (1 - exp(-ts / (τ1)))
    )

    μ_predict, Σ_predict = kf_inference(A, B, batt.μ, batt.Σ, batt.i, batt.model_noise.w.μ)
    return (
        μ=μ_predict,
        Σ=Σ_predict,
    )
end

function join_inference_step(batt)
    """
    Inference step for joint state
    """

    ts = batt.ts
    N_basis = size(batt.rgp_ocv.μ, 1) + size(batt.rgp_r.μ, 1)
    N_params = size(batt.μ_params, 1)
    N_rc = 1

    R1 = batt.μ_params[1]
    τ1 = batt.μ_params[2]

    A = vcat(
        hcat(I(N_basis), zeros(N_basis, N_rc), zeros(N_basis, N_params)),
        hcat(zeros(N_rc, N_basis), exp(-ts / (τ1)), zeros(N_rc, N_params)),
        hcat(zeros(N_params, N_basis + N_rc), I(N_params))
    )

    B = vcat(
        zeros(N_basis),
        R1 * (1 - exp(-ts / (τ1))),
        zeros(N_params)
    )

    μ_predict, Σ_predict = kf_inference(A, B, batt.μ, batt.Σ, batt.i, batt.model_noise.w.μ)

    return (
        μ=μ_predict,
        Σ=Σ_predict,
    )

end

function params_q_inference_step(batt)
    """
    Performs parameters estimation with q inference step
    """
    ts = batt.ts

    N_params = size(batt.μ, 1)

    A = I

    B = vcat(
        zeros(N_params, 1),
        ts
    )

    μ_params_predict, Σ_params_predict = kf_inference(A, B, batt.μ_params, batt.Σ_params, batt.i, batt.param_noise.w.μ)
    return μ_params_predict, Σ_params_predict
end



function params_inference_step(batt)
    """
    Predicts parameters of RC
    """
    A = I
    B = [0.0]
    μ_params_predict, Σ_params_predict = kf_inference(A, B, batt.μ_params, batt.Σ_params, batt.i, batt.param_noise.w.μ)
    return μ_params_predict, Σ_params_predict

end




############# Update ###################
function update_step_no_rc!(batt, batch)
    """
    Update step without RC
    """
    dt = batt.dt
    N_basis = size(batt.rgp_ocv.μ, 1) + size(batt.rgp_r.μ, 1)
    Σ_predict = batt.Σ[1:N_basis, 1:N_basis]
    μ_predict = batt.μ[1:N_basis]

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

    S = H * Σ_predict * H' + (batt.model_noise.v.Σ) * I(size(batch.y, 1))

    Gk = Σ_predict * H' * inv(S)

    new_μ = μ_predict + Gk * (e)
    new_Σ = Σ_predict - Gk * H * Σ_predict


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


function update_step_joint_state!(batt, batch, old_μ, old_i, μ_predict, Σ_predict)
    """
    This update step, updates thr RC parameters as joint
    """
    N_state = size(batt.μ, 1)
    N_params = size(batt.μ_params, 1)
    N_basis = size(batt.rgp_ocv.μ, 1) + size(batt.rgp_r.μ, 1)
    N_rc = 1

    ts = batt.ts
    dt = batt.dt

    ## OCV
    ocv = RecursiveGPs.predict(batt.rgp_ocv, batch.x.soc)
    ocv_v = StatsBase.reconstruct(dt.v, ocv.μ)
    ocv_Σ = ocv.Σ .* (dt.v.scale .^ 2)

    ## R0
    r0 = RecursiveGPs.predict(batt.rgp_r, batch.x.soc)
    r0_v = StatsBase.reconstruct(dt.σ, r0.μ)
    r0_Σ = r0.Σ .* (dt.σ.scale .^ 2)

    ## RC params
    R1 = batt.μ_params[end-1]
    τ1 = batt.μ_params[end]

    ## Building Kalman Filter
    e = batch.y - (ocv_v + batch.x.i .* r0_v .+ μ_predict[end-N_params] + batt.model_noise.v.μ)



    H_ocv = dt.v.scale .* cov(batt.rgp_ocv.gp, batch.x.soc, batt.rgp_ocv.X_basis) * batt.rgp_ocv.inv_cov
    H_r0 = dt.σ.scale .* batch.x.i .* cov(batt.rgp_r.gp, batch.x.soc, batt.rgp_r.X_basis) * batt.rgp_r.inv_cov
    H_rc1 = 1
    HR1 = (1 - exp(-ts / (τ1))) * old_i
    Hτ1 = exp(-ts / τ1) * ts / (τ1^2) * ([old_μ[end-N_params]] .- R1 * old_i)


    H = hcat(H_ocv, H_r0, H_rc1, HR1, Hτ1)



    S = H * Σ_predict * H' + (batt.model_noise.v.Σ) * I(size(batch.y, 1))

    Gk = Σ_predict * H' * inv(S)

    new_μ = μ_predict + Gk * (e)
    new_Σ = Σ_predict - Gk * H * Σ_predict


    ## Updating model
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

    ## Updating RC parameters
    batt.μ_params = new_μ[N_basis+N_rc+1:end]
    batt.Σ_params = new_Σ[N_basis+N_rc+1:end, N_basis+N_rc+1:end]

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

    batt.μ = new_μ
    batt.Σ = new_Σ
    batt.i = batch.x.i[1]

end


function update_params!(batt, batch, old_μ, old_i, μ_params_predict, Σ_params_predict; model_R=true)
    """
    Update parameters of RC
    """
    ts = batt.ts
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



### Helper functions for repeated code

function kf_inference(A, B, μ, Σ, u, w)
    """
    Standard kf inference step assuming
    """
    μ_predict = A * μ + B * u + w.μ
    Σ_predict = A * Σ * A' + w.Σ

    return μ_predict, Σ_predict
end


function kf_update(H, e, μ_predict, Σ_predict, v)
    """
    Standard kalman filter update step assuming 
    """
    S = H * Σ_predict * H' + (v.Σ) * I(size(v.μ), 1)
    Gk = Σ_predict * H' * inv(S)
    new_μ = μ_predict + Gk * (e)
    new_Σ = Σ_predict - Gk * H * Σ_predict
    return new_μ, new_Σ
end

function save_ocv!(batt, new_μ, new_Σ)
    """
    Saves the ocv parameters with a new mean and cov
    Note: For future better batt.μ and batt.Σ and rgp_ocv point to same place
    """
    ocv_start = 1
    ocv_end = size(batt.rgp_ocv.μ)[1]

    batt.rgp_ocv.μ = new_μ[
        ocv_start:ocv_end
    ]
    batt.rgp_ocv.Σ = new_Σ[
        ocv_start:ocv_end,
        ocv_start:ocv_end
    ]

end

function save_r!(batt, new_μ, new_Σ)
    """
    Saves the R parameters with a new mean
    Note: For future better batt.μ and batt.Σ and rgp_r point to same place
    """

    ocv_end = size(batt.rgp_ocv.μ)[1]
    r_start = ocv_end + 1
    r_end = ocv_end + size(batt.rgp_r.μ)[1]

    batt.rgp_r.μ = new_μ[
        r_start:r_end
    ]
    batt.rgp_r.Σ = new_Σ[
        r_start:r_end,
        r_start:r_end
    ]


end

function save_params!(batt, new_μ_params)
    """
    Save model parameters
    """
    batt.μ_params = new_μ_params
    batt.Σ_params = new_Σ_params
end

function save_model!(batt, new_μ)
    """
    Saves model state
    """
    batt.μ = new_μ
    batt.Σ = new_Σ

end
end