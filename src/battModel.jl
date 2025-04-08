module battModel

using ForwardDiff
using LinearAlgebra
using DataFrames
using AbstractGPs
using LinearAlgebra
using ..RecursiveGPs

export BATTModel, battery_learn!

mutable struct BATTModel

    rgp_ocv::RecursiveGPs.RGPModel
    rgp_r::RecursiveGPs.RGPModel
    μ::Vector{Float64}
    Σ::Matrix{Float64}

    function BATTModel(rgp_ocv, rgp_r)

        filler = zeros(size(rgp_ocv.Σ, 1), size(rgp_r.Σ, 2))
        μ = [rgp_ocv.μ; rgp_r.μ]
        Σ = vcat(
            [rgp_ocv.Σ filler],
            [filler' rgp_r.Σ]
        )

        new(rgp_ocv, rgp_r, μ, Σ)
    end
end


function battery_learn!(batt, batch)
    """
    Performs join state estimation of ocv and R0
    """
    ## Only update step
    ## Motion model
    σ_model = 0.1
    Σ = batt.Σ
    μ = batt.μ

    ocv = RecursiveGPs.predict(batt.rgp_ocv, batch.x.soc)
    r0 = RecursiveGPs.predict(batt.rgp_r, batch.x.soc)
    e = batch.y - (ocv.μ + batch.x.i .* r0.μ)

    H1 = cov(batt.rgp_ocv.gp, batch.x.soc, batt.rgp_ocv.X_basis) * batt.rgp_ocv.inv_cov
    H2 = batch.x.i .* cov(batt.rgp_r.gp, batch.x.soc, batt.rgp_r.X_basis) * batt.rgp_r.inv_cov
    H = [H1 H2]

    S = H * Σ * H' + (batch.x.i .^ 2 .* r0.Σ + ocv.Σ .+ σ_model^2) * I(size(batch.y, 1))

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

    batt.Σ = new_Σ
    batt.μ = new_μ


    return
end


"""
function update_step_batt!(batt::BATTModel, ocv, gp_r0, predicted, X_batch, Y_batch)
    ## Compute Observation matrix
    Vr(x) = isa(gp_r0, Number) ? gp_r0 * X_batch.i[1] : RecursiveGPs.predict(gp_r0, x).μ[1] * X_batch.i[1] ## R0 * i
    dOCV_dx = ForwardDiff.derivative(ocv, predicted.μ[1])
    dR0_dx = ForwardDiff.gradient(Vr, [predicted.μ[1]])[1]
    dV_dx = dOCV_dx .+ dR0_dx  ## dV_dx = dV_o/dx + dV_o/dx * dV/dx   

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
    return V̂_updated, R0_new, dOCV_dx, dR0_dx
end




function inference_step_batt(batt::BATTModel)
  
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
    V̂_updated, R0_new, dOCV_dx, dR0_dx = update_step_batt!(batt, ocv, gp_r0, predicted, X_batch, Y_batch)

    ## Save data
    if save_data == true
        push!(batt.data, (
            t=X_batch.t[1], i=X_batch.i[1], soc=batt.μ[1], σ_x=sqrt(abs.(batt.Σ[1, 1])),
            Vrc1=batt.μ[2], σ_vrc1=sqrt(abs.(batt.Σ[2, 2])),
            Vrc2=batt.μ[3], σ_vrc2=sqrt(abs.(batt.Σ[3, 3])),
            dOCV_dx=dOCV_dx, dR0_dx=dR0_dx,
            V̂=V̂_updated, R0=R0_new
        )
        )
    end
end
"""
end
