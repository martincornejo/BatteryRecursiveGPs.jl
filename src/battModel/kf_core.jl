
module kf_core

using LinearAlgebra
using ComponentArrays
using DataFrames
using AbstractGPs
using ..RecursiveGPs
using ..kf_utils
using ..battModel

export battery_learn!, battery_learn_rc!, battery_learn_dual_kf!, RTS_Storage, push_rts!, empty_rts!,
    battery_rts_smoother!, battery_params_rts_smoother, test_battery_learn_dual_kf!, test_battery_learn_rc!,
    battery_learn_join_kf!
###### With GP OcV and R



###### without RC
function battery_learn!(batt, batch; rts=false)
    """
    Performs join state estimation of ocv and R0
    Does only accept batt set up with rgp of ocv and rgp of R0
    """
    μ_predict = copy(batt.μ)
    Σ_predict = copy(batt.Σ)

    update_step_no_rc!(batt, batch)

    μ = copy(batt.μ)
    Σ = copy(batt.Σ)

    if rts
        push_rts!(batt.histogram_model, deepcopy(μ), deepcopy(Σ), deepcopy(μ_predict), deepcopy(Σ_predict), I)
    end

end
### RC learn
function battery_learn_rc!(batt, batch; rts=false)
    """
    Performs join state estimation of ocv, R0, Vrc1, Vrc2
    """

    μ_predict, Σ_predict = inference_step(batt)
    update_step!(batt, batch, μ_predict, Σ_predict)

    if rts
        push_rts!(batt.histogram_model, μ, Σ, μ_predict, Σ_predict, I)
    end
end

function test_battery_learn_rc!(batt, batch; rts=false)
    μ_predict, Σ_predict = test_inference_step(batt)
    test_update_step!(batt, batch, μ_predict, Σ_predict)
end



### Dual kF

function battery_learn_dual_kf!(batt, batch; adaptive_noise=false, rts=false)
    """
    Performs dual KF of ocv,R0, Vrc1 and RC parameters
    """
    old_μ = copy(batt.μ)
    old_i = copy(batt.i)
    old_Σ_params = copy(batt.Σ_params)

    ## Model Kalman Filter
    μ_predict, Σ_predict = inference_step(batt)
    update_step!(batt, batch, μ_predict, Σ_predict)

    ## Kalman filter of parameters
    μ_params_predict, Σ_params_predict = params_inference_step(batt)
    Gk, H, e = update_params!(batt, batch, old_μ, old_i, μ_params_predict, Σ_params_predict; model_R=true)

    if adaptive_noise == true
        adaptive_extended_kf!(batt, batch, old_Σ_params, Gk, H, e)
    end
end

function battery_learn_join_kf!(batt, batch; adaptive_noise=false, rts=false)
    """
    Performs single Kf  of ocv, R0, Vrc1 and RC parameters
    """
    old_μ = copy(batt.μ)
    old_i = copy(batt.i)
    old_Σ_params = copy(batt.Σ_params)

    ## Model Kalman Filter
    μ_predict, Σ_predict = join_inference_step(batt)


    ## Kalman filter
    update_step_joint_state!(batt, batch, old_μ, old_i, μ_predict, Σ_predict)




end


function test_battery_learn_dual_kf!(batt, batch; rts=false)
    """
    Testing functions for dataset where R0 is known
    """
    old_μ = copy(batt.μ)
    old_i = copy(batt.i)
    old_Σ_params = copy(batt.Σ_params)

    ## Model Kalman Filter
    μ_predict, Σ_predict = test_inference_step(batt)
    test_update_step!(batt, batch, μ_predict, Σ_predict)

    ## Kalman filter of parameters
    μ_params_predict, Σ_params_predict = params_inference_step(batt)
    Gk, H, e = update_params!(batt, batch, old_μ, old_i, μ_params_predict, Σ_params_predict; model_R=false)

    if rts
        push_rts!(
            batt.histogram_params,
            deepcopy(batt.μ_params),
            deepcopy(batt.Σ_params),
            deepcopy(μ_params_predict),
            deepcopy(Σ_params_predict),
            I
        )
    end

end


## RTS smoother functions

function push_rts!(histogram, μ, Σ, μ_predict, Σ_predict, A)
    """
    Saving the RTS parameters
    """
    push!(histogram.μ, μ)
    push!(histogram.Σ, Σ)
    push!(histogram.μ_predict, μ_predict)
    push!(histogram.Σ_predict, Σ_predict)
    push!(histogram.A, A)

end


function empty_rts!(histogram)
    """
    Resets the rts smoother
    """
    empty!(histogram.μ)
    empty!(histogram.Σ)
    empty!(histogram.μ_predict)
    empty!(histogram.Σ_predict)
    empty!(histogram.A)
end

function battery_rts_smoother!(batt, k)
    """
    RTS Smoother for kalman filter of the model
    """
    histogram = batt.histogram_model
    Gk = histogram.Σ[k] * histogram.A[k]' \ histogram.Σ_predict[k+1]

    μ_s = histogram.μ[k] + Gk * (batt.μ - histogram.μ_predict[k+1])
    Σ_s = histogram.Σ[k] + Gk * (batt.Σ - histogram.Σ_predict[k+1]) * Gk'



    ### Updating Model
    ocv_start = 1
    ocv_end = size(batt.rgp_ocv.μ)[1]

    r_start = ocv_end + 1
    r_end = ocv_end + size(batt.rgp_r.μ)[1]

    batt.rgp_ocv.μ = μ_s[
        ocv_start:ocv_end
    ]
    batt.rgp_ocv.Σ = Σ_s[
        ocv_start:ocv_end,
        ocv_start:ocv_end
    ]

    batt.rgp_r.μ = μ_s[
        r_start:r_end
    ]
    batt.rgp_r.Σ = Σ_s[
        r_start:r_end,
        r_start:r_end
    ]

    batt.μ = μ_s
    batt.Σ = Σ_s
end

function battery_params_rts_smoother(batt, k)
    """
    RTS Smoother for Kalman filter of the parameters
    """

    histogram = batt.histogram_params
    Gk = histogram.Σ[k] * histogram.A[k]' \ histogram.Σ_predict[k+1]
    μ_s = histogram.μ[k] + Gk * (batt.μ_params - histogram.μ_predict[k+1])
    Σ_s = histogram.Σ[k] + Gk * (batt.Σ_params - histogram.Σ_predict[k+1]) * Gk'

    ### Updating RC parameters
    batt.μ_params = copy(μ_s)
    batt.Σ_params = copy(Σ_s)
end

end
