using ..RecursiveGPs
"""
Dummy file for testing methods on synthetic dataset
"""

function dual_kf!(rgp_ocv, batch, i, μ, Σ, μ_params, Σ_params, dt; update_params=true)

    ## Number of RCs
    if size(μ_params, 1) == 2
        N_rc = 1
    elseif size(μ_params, 1) == 4
        N_rc = 2
    end

    ## Update model
    μ_predict, Σ_predict = predict(rgp_ocv, i, μ, Σ, μ_params, N_rc)
    new_μ, new_Σ = update_model!(rgp_ocv, dt, batch, μ_predict, Σ_predict, N_rc)

    # Update parameters
    if update_params == true
        μ_params_predict, Σ_params_predict = predict_params(μ_params, Σ_params, N_rc)
        new_μ_params, new_Σ_params = update_params!(rgp_ocv, dt, i, batch, new_μ, μ_params_predict, Σ_params_predict, μ, N_rc)
    else
        new_μ_params, new_Σ_params = μ_params, Σ_params
    end

    return new_μ, new_Σ, new_μ_params, new_Σ_params
end



function predict(rgp_ocv, i, μ, Σ, μ_params, N_rc)
    ## Motion model
    ts = 1 ## PLaceholder, to be changed for new data.

    if typeof(rgp_ocv) == RGPModel
        N_basis = size(rgp_ocv.X_basis, 1)
    else
        N_basis = 0
    end


    if N_rc == 1
        R1 = μ_params[1]
        τ1 = μ_params[2]

        A_batt = vcat(
            [I(N_basis) zeros(N_basis, 1)],
            [zeros(1, N_basis) exp(-ts / (τ1))],
        )

        B_batt = [
            zeros(N_basis, 1);
            1e-3 * R1 * (1 - exp(-ts / (τ1)))
        ]

        Q_batt = vcat(
            [zeros(N_basis, N_basis) zeros(N_basis, 1)],
            [zeros(1, N_basis) 10e-6],
        )



    elseif N_rc == 2
        R1 = μ_params[1]
        τ1 = μ_params[2]
        R2 = μ_params[3]
        τ2 = μ_params[4]

        A_batt = vcat(
            [I(N_basis) zeros(N_basis, 2)],
            [zeros(1, N_basis) exp(-ts / (τ1)) 0],
            [zeros(1, N_basis) 0 exp(-ts / (τ2))]
        )

        B_batt = [
            zeros(N_basis, 1);
            1e-3 * R1 * (1 - exp(-ts / (τ1)));
            1e-3 * R2 * (1 - exp(-ts / (τ2)))
        ]

        Q_batt = vcat(
            [zeros(N_basis, N_basis) zeros(N_basis, 2)],
            [zeros(1, N_basis) 10e-6 0],
            [zeros(1, N_basis) 0 10e-6]
        )

    end

    μ_predict = A_batt * μ + B_batt * i
    Σ_predict = A_batt * Σ * A_batt' + Q_batt

    return μ_predict, Σ_predict
end


function update_model!(rgp_ocv, dt, batch, μ_predict, Σ_predict, N_rc)
    H_rc1 = 1
    H_rc2 = 1
    R0 = 15e-3
    σ_model = 1e-3


    if typeof(rgp_ocv) == RGPModel

        ocv = RecursiveGPs.predict(rgp_ocv, batch.x.soc)
        ocv_v = StatsBase.reconstruct(dt.v, ocv.μ)
        ocv_Σ = ocv.Σ .* (dt.v.scale .^ 2)

        H_ocv = dt.v.scale .* cov(rgp_ocv.gp, batch.x.soc, rgp_ocv.X_basis) * rgp_ocv.inv_cov
        if N_rc == 1
            e = batch.v - (ocv_v .+ R0 * batch.x.i .+ μ_predict[end])
            H = [H_ocv H_rc1]
        elseif N_rc == 2
            e = batch.v - (ocv_v .+ R0 * batch.x.i .+ μ_predict[end] .+ μ_predict[end-1])
            H = [H_ocv H_rc1 H_rc2]
        end
        S = H * Σ_predict * H' + (ocv_Σ .+ σ_model^2) * I(size(batch.v, 1))

    else
        ocv_v = StatsBase.reconstruct(dt.v, rgp_ocv(batch.x.soc))

        if N_rc == 1
            e = batch.v - (ocv_v .+ R0 * batch.x.i .+ μ_predict[end])
            H = [H_rc1]
        elseif N_rc == 2
            e = batch.v - (ocv_v .+ R0 * batch.x.i .+ μ_predict[end] .+ μ_predict[end-1])
            H = [H_rc1 H_rc2]
        end

        S = H * Σ_predict * H' + (σ_model^2) * I(size(batch.v, 1))

    end

    Gk = Σ_predict * H' * inv(S)
    new_μ = μ_predict + Gk * (e)
    new_Σ = Σ_predict - Gk * H * Σ_predict

    ## Updating model
    if typeof(rgp_ocv) == RGPModel
        size_ocv = size(rgp_ocv.μ)[1]

        rgp_ocv.μ = new_μ[1:size_ocv]
        rgp_ocv.Σ = new_Σ[1:size_ocv, 1:size_ocv]
    end

    return new_μ, new_Σ
end

function predict_params(μ_params, Σ_params, N_rc)
    ## Motion model
    noise_R = 3e-3
    noise_τ = 8e-4
    Q_params = vcat(
        [noise_R 0],
        [0 noise_τ]
    )

    Q_params = kron(Diagonal(ones(N_rc)), Q_params)

    μ_params_predict = μ_params
    Σ_params_predict = Σ_params + Q_params

    return μ_params_predict, Σ_params_predict
end

function update_params!(rgp_ocv, dt, i, batch, new_μ, μ_params_predict, Σ_params_predict, old_μ, N_rc)
    ts = 1
    R0 = 15e-3

    if typeof(rgp_ocv) == RGPModel
        ocv = RecursiveGPs.predict(rgp_ocv, batch.x.soc)
        ocv_v = StatsBase.reconstruct(dt.v, ocv.μ)

        if N_rc == 1
            e = batch.v - (ocv_v .+ R0 * batch.x.i .+ new_μ[end])
        elseif N_rc == 2
            e = batch.v - (ocv_v .+ R0 * batch.x.i .+ new_μ[end] .+ new_μ[end-1])
        end

    else
        ocv_v = StatsBase.reconstruct(dt.v, rgp_ocv(batch.x.soc))
        if N_rc == 1
            e = batch.v - (ocv_v .+ R0 * batch.x.i .+ new_μ[end])
        elseif N_rc == 2
            e = batch.v - (ocv_v .+ R0 * batch.x.i .+ new_μ[end] .+ new_μ[end-1])
        end
    end

    # one_rc
    if N_rc == 1
        R1 = μ_params_predict[1]
        τ1 = μ_params_predict[2]

        HR1 = 1e-3 * (1 - exp(-ts / (τ1))) * i

        Hτ1 = exp(-ts / τ1) * ts / (τ1^2) * ([old_μ[end]] .- 1e-3 * R1 * i)

        H = [HR1 Hτ1]
    end


    # two_rc
    if N_rc == 2

        R1 = μ_params_predict[1]
        τ1 = μ_params_predict[2]
        R2 = μ_params_predict[3]
        τ2 = μ_params_predict[4]

        HR1 = 1e-3 * (1 - exp(-ts / (τ1))) * i

        HR2 = 1e-3 * (1 - exp(-ts / (τ2))) * i
        Hτ1 = exp(-ts / τ1) * ts / (τ1^2) * (old_μ[end-1] .- 1e-3 * R1 * i)

        Hτ2 = exp(-ts / τ2) * ts / (τ2^2) * (old_μ[end] .- 1e-3 * R2 * i)

        H = [HR1 HR2 Hτ1 Hτ2]
    end

    σ_model = 5e-3
    S = H * Σ_params_predict * H' + (σ_model^2) * I(size(batch.v, 1))
    Gk = Σ_params_predict * H' * inv(S)
    new_μ_params = μ_params_predict + Gk * (e)
    new_Σ_params = Σ_params_predict - Gk * H * Σ_params_predict

    return new_μ_params, new_Σ_params

end
