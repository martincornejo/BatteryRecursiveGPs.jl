module battModel

using ForwardDiff
using LinearAlgebra
using DataFrames
using AbstractGPs
using LinearAlgebra
using ..RecursiveGPs

export BATTModel, battery_learn!, battery_learn_rc!

mutable struct BATTModel

    rgp_ocv::RecursiveGPs.RGPModel
    rgp_r::RecursiveGPs.RGPModel
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    σ_model::Float64
    R1::Float64
    τ1::Float64
    R2::Float64
    τ2::Float64
    i::Float64

    function BATTModel(rgp_ocv, rgp_r, σ_model)

        filler = zeros(size(rgp_ocv.μ, 1), size(rgp_r.μ, 1))
        filler_rc = zeros(1, size(rgp_ocv.μ, 1) + size(rgp_r.μ, 1))

        Q_vrc = 10e-6
        μ = [rgp_ocv.μ; rgp_r.μ; 0; 0]

        Σ = vcat(
            [rgp_ocv.Σ filler zeros(size(rgp_ocv.Σ, 1), 2)],
            [filler' rgp_r.Σ zeros(size(rgp_r.Σ, 1), 2)],
            [filler_rc Q_vrc 0],
            [filler_rc 0 Q_vrc]
        )

        ### Temporal Placeholders for PROFILE 2 dataset
        R1 = 1e-4
        τ1 = 60.0
        R2 = 1e-4
        τ2 = 600.0
        i = 0.0

        new(rgp_ocv, rgp_r, μ, Σ, σ_model, R1, τ1, R2, τ2, i)
    end
end



function inference_step(batt)
    ts = 10 ## PLaceholder for PROFILE 2 Dataset, to be changed for new data
    N_basis = size(batt.rgp_ocv.X_basis, 1) + size(batt.rgp_ocv.X_basis, 1)
    A_batt = vcat(
        [I(N_basis) zeros(N_basis, 2)],
        [zeros(1, N_basis) exp(-ts / (batt.τ1)) 0],
        [zeros(1, N_basis) 0 exp(-ts / (batt.τ2))]
    )

    B_batt = [
        zeros(N_basis);
        batt.R1 * (1 - exp(-ts / (batt.τ1)));
        batt.R2 * (1 - exp(-ts / (batt.τ2)))
    ]
    ## Noise of Motion model set as 10e-6 for Vrc Kalman filter side following Huwey papers
    Q_batt = vcat(
        [zeros(N_basis, N_basis) zeros(N_basis, 2)],
        [zeros(1, N_basis) 10e-6 0],
        [zeros(1, N_basis) 0 10e-6]
    )

    μ_predict = A_batt * batt.μ + B_batt * batt.i
    Σ_predict = A_batt * batt.Σ * A_batt' + Q_batt

    return (
        μ=μ_predict,
        Σ=Σ_predict,
    )
end


function update_step!(batt, batch, μ_predict, Σ_predict)
    """
    Performs join state estimation of ocv and R0
    """
    ## Update step
    Σ = Σ_predict
    μ = μ_predict

    ocv = RecursiveGPs.predict(batt.rgp_ocv, batch.x.soc)
    r0 = RecursiveGPs.predict(batt.rgp_r, batch.x.soc)
    e = batch.y - (ocv.μ + batch.x.i .* r0.μ .+ μ[end] .+ μ[end-1])

    H_ocv = cov(batt.rgp_ocv.gp, batch.x.soc, batt.rgp_ocv.X_basis) * batt.rgp_ocv.inv_cov
    H_r0 = batch.x.i .* cov(batt.rgp_r.gp, batch.x.soc, batt.rgp_r.X_basis) * batt.rgp_r.inv_cov
    H_rc1 = 1
    H_rc2 = 1
    H = [H_ocv H_r0 H_rc1 H_rc2]

    S = H * Σ * H' + (batch.x.i .^ 2 .* r0.Σ + ocv.Σ .+ batt.σ_model^2) * I(size(batch.y, 1))

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
    update_step!(batt, batch, μ_predict, Σ_predict)
end


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
    e = batch.y - (ocv.μ + batch.x.i .* r0.μ)

    H1 = cov(batt.rgp_ocv.gp, batch.x.soc, batt.rgp_ocv.X_basis) * batt.rgp_ocv.inv_cov
    H2 = batch.x.i .* cov(batt.rgp_r.gp, batch.x.soc, batt.rgp_r.X_basis) * batt.rgp_r.inv_cov
    H = [H1 H2]

    S = H * Σ * H' + (batch.x.i .^ 2 .* r0.Σ + ocv.Σ .+ batt.σ_model^2) * I(size(batch.y, 1))

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
