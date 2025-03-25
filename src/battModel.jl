module battModel

using ForwardDiff
using LinearAlgebra
using DataFrames
using KernelFunctions
using LinearAlgebra
using ..RecursiveGPs

export BATTModel, learn_batt!, learn!

mutable struct BATTModel
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    i::Float64

    Q_batt::Matrix{Float64}
    R_batt::Float64

    Q::Float64
    R1::Float64
    τ1::Float64
    R2::Float64
    τ2::Float64

    data::DataFrame

    function BATTModel(μ, Q_batt, R_batt, R1, τ1, R2, τ2)

        Σ = 0.0001 * I(3)
        Q = 4.8 * 3600
        i = 0.0
        data = DataFrame(
            t=Float64[], i=Float64[], soc=Float64[], σ_x=Float64[],
            Vrc1=Float64[], σ_vrc1=Float64[],
            Vrc2=Float64[], σ_vrc2=Float64[],
            V̂=Float64[], R0=Float64[]
        )

        new(μ, Σ, i, Q_batt, R_batt, Q, R1, τ1, R2, τ2, data)
    end
end


function update_step_batt!(batt::BATTModel, ocv, gp_r0, predicted, X_batch, Y_batch)
    ## Compute Observation matrix
    Vr(x) = isa(gp_r0, Number) ? gp_r0 * X_batch.i[1] : RecursiveGPs.predict(gp_r0, x).μ[1] * X_batch.i[1] ## R0 * i

    dV_dx = ForwardDiff.derivative(ocv, predicted.μ[1]) .+
            ForwardDiff.gradient(Vr, [predicted.μ[1]])[1]  ## dV_dx = dV_o/dx + dV_o/dx * dV/dx   

    dV_dVrc1 = 1.0
    dV_dVrc2 = 1.0
    H = [dV_dx dV_dVrc1 dV_dVrc2]

    ## Kalman Gain
    S = H * predicted.Σ * H' .+ batt.R_batt
    Gk = predicted.Σ * H' * inv(S)

    ## Update step

    V̂ = ocv(predicted.μ[1]) + predicted.μ[2] + predicted.μ[3] + Vr([predicted.μ[1]])
    μ_new = predicted.μ + Gk * (Y_batch .- V̂)
    Σ_new = predicted.Σ - Gk * S * Gk'

    ## Updating model and storing predicted voltage
    V̂_updated = ocv(μ_new[1]) + μ_new[2] + μ_new[3] + Vr([predicted.μ[1]])
    R0_new = isa(gp_r0, Number) ? gp_r0 : Vr([predicted.μ[1]]) / X_batch.i[1]

    batt.μ = μ_new
    batt.Σ = Σ_new
    batt.i = X_batch.i[1]
    return V̂_updated, R0_new
end




function inference_step_batt(batt::BATTModel)
    """
    Does the prediction step for the Variables, NOT OF V
    """
    ts = 1 ## PLaceholder, to be changed for new data.
    A_batt = vcat(
        [1 0 0],
        [0 exp(-ts / (batt.τ1)) 0],
        [0 0 exp(-ts / (batt.τ2))]
    )

    B_batt = [ts / batt.Q; batt.R1 * (1 - exp(-ts / (batt.τ1))); batt.R2 * (1 - exp(-ts / (batt.τ2)))]

    μ_predict = A_batt * batt.μ + B_batt * batt.i
    Σ_predict = A_batt * batt.Σ * A_batt' + batt.Q_batt

    return (
        μ=μ_predict,
        Σ=Σ_predict,
    )
end

function learn_batt!(batt::BATTModel, ocv, gp_r0, X_batch, Y_batch, save_data=true)

    predicted = inference_step_batt(batt)
    V̂_updated, R0_new = update_step_batt!(batt, ocv, gp_r0, predicted, X_batch, Y_batch)

    ## Save data
    if save_data == true
        push!(batt.data, (
            t=X_batch.t[1], i=X_batch.i[1], soc=batt.μ[1], σ_x=sqrt(abs.(batt.Σ[1, 1])),
            Vrc1=batt.μ[2], σ_vrc1=sqrt(abs.(batt.Σ[2, 2])),
            Vrc2=batt.μ[3], σ_vrc2=sqrt(abs.(batt.Σ[3, 3])),
            V̂=V̂_updated, R0=R0_new
        )
        )
    end
end

end
