using ..RecursiveGPs


function dual_kf!(rgp_ocv, batch, i, μ, Σ, μ_params, Σ_params, dt, dVrc_α_acc)
    ## NUmber of RCs
    N_rc = 1
    old_μ = μ
    ## Update model
    μ_predict, Σ_predict = predict(rgp_ocv, i, μ, Σ, μ_params=μ_params, N_rc=N_rc)
    new_μ, new_Σ = update_model!(rgp_ocv, dt, batch, μ_predict, Σ_predict, N_rc=N_rc)

    # Update parameters
    μ_params_predict, Σ_params_predict = predict_params(μ_params, Σ_params)
    new_μ_params, new_Σ_params, dVrc_α_ac = update_params!(rgp_ocv, dt, i, batch, new_μ, μ_params_predict, Σ_params_predict, old_μ, dVrc_α_acc)
    return new_μ, new_Σ, new_μ_params, new_Σ_params, dVrc_α_ac
end



function predict(rgp_ocv, i, μ, Σ; μ_params=[15e-3, 60], N_rc=2)
    ## Motion model
    ts = 1 ## PLaceholder, to be changed for new data.

    if typeof(rgp_ocv) == RGPModel
        N_basis = size(rgp_ocv.X_basis, 1)
    else
        N_basis = 0
    end


    R1 = μ_params[1]
    τ1 = μ_params[2]

    A_batt = vcat(
        [I(N_basis) zeros(N_basis, 1)],
        [zeros(1, N_basis) exp(-ts / (τ1))],
    )

    B_batt = [
        zeros(N_basis, 1);
        R1 * 1e-3 * (1 - exp(-ts / (τ1)))
    ]

    Q_batt = vcat(
        [zeros(N_basis, N_basis) zeros(N_basis, 1)],
        [zeros(1, N_basis) 10e-6],
    )

    μ_predict = A_batt * μ + B_batt * i
    Σ_predict = A_batt * Σ * A_batt' + Q_batt
    return μ_predict, Σ_predict
end


function update_model!(rgp_ocv, dt, batch, μ_predict, Σ_predict; N_rc=2)
    H_rc1 = 1
    R0 = 15e-3
    σ_model = 1e-2


    if typeof(rgp_ocv) == RGPModel
        ocv = RecursiveGPs.predict(rgp_ocv, batch.x.soc)
        H_ocv = cov(rgp_ocv.gp, batch.x.soc, rgp_ocv.X_basis) * rgp_ocv.inv_cov

        e = batch.v - (ocv.μ .+ R0 * batch.x.i .+ μ_predict[end])
        H = [H_ocv H_rc1]
        S = H * Σ_predict * H' + (ocv.Σ .+ σ_model^2) * I(size(batch.v, 1))

    else
        ocv = rgp_ocv(batch.x.soc)
        e = batch.v - (ocv .+ R0 * batch.x.i .+ μ_predict[end])
        H = [H_rc1]
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

function predict_params(μ_params, Σ_params)
    ## Motion model
    Q_params = vcat(
        [1e-6 0],
        [0 1e-3]
    )

    μ_params_predict = μ_params
    Σ_params_predict = Σ_params + Q_params

    return μ_params_predict, Σ_params_predict
end

function update_params!(rgp_ocv, dt, i, batch, new_μ, μ_params_predict, Σ_params_predict, old_μ, dVrc_α_acc)
    ts = 1
    R0 = 15e-3
    if typeof(rgp_ocv) == RGPModel
        ocv = RecursiveGPs.predict(rgp_ocv, batch.x.soc)
    else
        ocv = rgp_ocv(batch.x.soc)
    end
    e = batch.v - (ocv .+ R0 * batch.x.i .+ new_μ[end])


    # one_rc
    α_curr = (
        R1=μ_params_predict[1],
        τ1=μ_params_predict[2]
    )


    dg_α(α) =
        dg_ocv(α) = 1

    df_α(α) =
        df_ocv(α) =
            dVrc_α(α) = df_α(α) + df_ocv(α) * docv_α_acc(α) + df_r(α) * dr_α_acc(α)

    H(α) = dg_α(α) + dg_ocv(α) * docv_α(α) + dg_r(α) * dr_α(α)
    println("---------")



    ## Update parameters
    σ_model = 1e-4
    S = H(α_curr) * Σ_params_predict * H(α_curr)' + (σ_model^2) * I(size(batch.v, 1))
    Gk = Σ_params_predict * H(α_curr)' * inv(S)
    new_μ_params = μ_params_predict + Gk * (e)
    new_Σ_params = Σ_params_predict - Gk * H(α_curr) * Σ_params_predict

    ## Update accumulated gradient dVrc_α_acc
    new_dVrc_α_acc(α) = dVrc_α(α) - Gk * H(α)

    return new_μ_params, new_Σ_params, new_dVrc_α_acc

end