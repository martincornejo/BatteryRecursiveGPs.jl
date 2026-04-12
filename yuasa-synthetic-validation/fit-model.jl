function fit_zscore(n = 1)
    v = StatsBase.fit(ZScoreTransform, (n * 3.3):(n * 0.01):(n * 4.1))
    σ = StatsBase.fit(ZScoreTransform, (n * 3.3):(n * 0.01):(n * 4.1), center = false)
    i = StatsBase.fit(ZScoreTransform, -50:0.1:50, center = false)
    q = StatsBase.fit(ZScoreTransform, -50:0.1:50, center = false)
    r = ZScoreTransform(1, 1, [0.0], [σ.scale[1] / i.scale[1]])
    return (; v, σ, i, q, r)
end

function cell_dataset(data::DataFrame, cell_id::Int; Ts = 1.0)
    zt = fit_zscore()

    v̂ = StatsBase.transform(zt.v, data[:, "v_cell_$cell_id"])
    î = StatsBase.transform(zt.i, data.î)
    q̂ = StatsBase.transform(zt.q, cumsum(data.î) * Ts / 3600)
    T = data.T

    u = [(; i, q, T) for (i, q, T) in zip(î, q̂, T)]
    y = [SA[v] for v in v̂]

    return (; u, y)
end

function fit_model(df::DataFrame, cell_id::Int, θ; n = 21, pad = 0.05)
    u, y = cell_dataset(df, cell_id)
    zt = fit_zscore()
    model = YuasaModel(θ, u, zt; n, pad)

    stats = @timed begin
        sol = run_kf!(model, u, y)
    end

    @info "Cell $(cell_id): complete" stats.time

    return (; model, sol)
end

function fit_models(data, ids, θ; n = 21, pad = 0.05)
    models = Dict()
    sols = Dict()

    for batch in Iterators.partition(ids, Threads.nthreads())
        tasks = Dict(id => Threads.@spawn fit_model(data, id, θ; n, pad) for id in batch)

        for (id, task) in tasks
            (; model, sol) = fetch(task)
            models[id] = model
            sols[id] = sol
        end
    end

    return (; models, sols)
end

function refine_model(data::DataFrame, cell_id::Int, model, sol; Ts = 1.0, σ1_cc = nothing)
    u, y = cell_dataset(data, cell_id; Ts)

    model_new = deepcopy(model)
    reinit_kf!(model_new.kf; x = sol.xt[end], R = sol.Rt[end])

    if σ1_cc !== nothing
        model_new.kf.R1[:cc, :cc] .= [σ1_cc^2;;]
    end

    stats = @timed begin
        sol_new = run_kf!(model_new, u, y)
    end

    @info "Cell $(cell_id): refined" stats.time
    return (; model = model_new, sol = sol_new)
end

function refine_models(data, ids, models, sols; kwargs...)
    r_models = Dict()
    r_sols = Dict()

    for batch in Iterators.partition(ids, Threads.nthreads())
        tasks = Dict(id => Threads.@spawn refine_model(data, id, models[id], sols[id]; kwargs...) for id in batch)

        for (id, task) in tasks
            (; model, sol) = fetch(task)
            r_models[id] = model
            r_sols[id] = sol
        end
    end

    return (; models = r_models, sols = r_sols)
end

function extract_posterior(model, sol; n_grid = 200)
    kf = model.kf
    zt = kf.p.zt
    xs = ComponentVector.(sol.xt, kf.p.xid)
    q̂min, q̂max = extrema([x.cc.q for x in xs])
    q̂ = collect(range(q̂min, q̂max, n_grid))
    q = StatsBase.reconstruct(zt.q, q̂)
    ocv = predict_gp(kf, q̂, :ocv)
    μ_v = StatsBase.reconstruct(zt.v, ocv.μ)
    scale_v = zt.v.scale[1]
    Σ_v = scale_v^2 .* Matrix(ocv.Σ)
    return (; q, μ = μ_v, Σ = Symmetric(Σ_v))
end


# === private helpers

# Compute absolute-error metrics (rmse, mae, max, q95, q99) from a normalised
# voltage-error vector `et`, converting to mV using the σ ZScore transform.
function _volt_metrics(et, zt)
    e = abs.(StatsBase.reconstruct(zt.σ, et)) .* 1000  # mV
    return (;
        rmse = sqrt(mean(abs2, e)), mae = mean(abs, e), max = maximum(e),
        q95 = quantile(e, 0.95), q99 = quantile(e, 0.99),
    )
end

# Generic helper for absolute-error metrics on any float vector.
function _err_metrics(e)
    ae = abs.(e)
    return (;
        rmse = sqrt(mean(abs2, e)), mae = mean(ae), max = maximum(ae),
        q95 = quantile(ae, 0.95), q99 = quantile(ae, 0.99),
    )
end


# === DataFrame reports

"""
    params_to_df(param_cells, params_real)

Build a DataFrame comparing estimated Q and s0 to ground truth, one row per cell.
Columns: cell, Q_true, Q_est, Q_unc, Q_err, s0_true_%, s0_est_%, s0_unc_%, s0_err_%.
"""
function params_to_df(param_cells, params_real)
    ids = sort(collect(keys(param_cells)))
    return DataFrame(
        cell = ids,
        Q_true = [params_real["cell_$id"]["Q"] for id in ids],
        Q_est = [Measurements.value(param_cells[id][:Q]) for id in ids],
        Q_unc = [Measurements.uncertainty(param_cells[id][:Q]) for id in ids],
        Q_err = [Measurements.value(param_cells[id][:Q]) - params_real["cell_$id"]["Q"] for id in ids],
        s0_true = [params_real["cell_$id"]["soc"] * 100 for id in ids],
        s0_est = [Measurements.value(param_cells[id][:soc]) * 100 for id in ids],
        s0_unc = [Measurements.uncertainty(param_cells[id][:soc]) * 100 for id in ids],
        s0_err = [(Measurements.value(param_cells[id][:soc]) - params_real["cell_$id"]["soc"]) * 100 for id in ids],
    )
end

"""
    ocv_residuals_to_df(models, sols, param_cells, params_real, focv)

OCV GP residuals (mV) vs the reference OCV curve, evaluated using both estimated
and true Q/s0 for the SOC mapping. One row per cell.
"""
function ocv_residuals_to_df(models, sols, param_cells, params_real, focv)
    ids = sort(collect(keys(param_cells)))
    slims = extrema(focv.t)

    rows = map(ids) do id
        (; q, μ) = gp_ocv(models[id], sols[id])

        Q_est = Measurements.value(param_cells[id][:Q])
        s0_est = Measurements.value(param_cells[id][:soc])
        soc_est = s0_est .+ q ./ Q_est

        Q_r = params_real["cell_$id"]["Q"]
        s0_r = params_real["cell_$id"]["soc"]
        soc_r = s0_r .+ q ./ Q_r

        mask_est = findall(slims[1] .<= soc_est .<= slims[2])
        mask_real = findall(slims[1] .<= soc_r .<= slims[2])

        r_est = (μ[mask_est] .- focv.(soc_est[mask_est])) .* 1000  # mV
        r_real = (μ[mask_real] .- focv.(soc_r[mask_real])) .* 1000  # mV

        (;
            cell = id,
            max_mV_est = maximum(abs, r_est),
            mean_mV_est = mean(abs, r_est),
            rmse_mV_est = sqrt(mean(abs2, r_est)),
            max_mV_true = maximum(abs, r_real),
            mean_mV_true = mean(abs, r_real),
            rmse_mV_true = sqrt(mean(abs2, r_real)),
        )
    end

    return DataFrame(rows)
end

"""
    r0_residuals_to_df(models, sols, param_cells, fR025)

R0 GP residuals (mΩ) vs the reference R0 curve at 25 °C. One row per cell.
"""
function r0_residuals_to_df(models, sols, param_cells, fR025)
    ids = sort(collect(keys(param_cells)))

    rows = map(ids) do id
        (; q, μ) = gp_r0(models[id], sols[id])

        Q_est = Measurements.value(param_cells[id][:Q])
        s0_est = Measurements.value(param_cells[id][:soc])
        soc_est = s0_est .+ q ./ Q_est

        r_ref = fR025.(soc_est)        # Ω, reference at 25 °C
        err_mΩ = (μ .- r_ref) .* 1.0e3   # mΩ

        (;
            cell = id,
            max_mΩ = maximum(abs, err_mΩ),
            mean_mΩ = mean(abs, err_mΩ),
            rmse_mΩ = sqrt(mean(abs2, err_mΩ)),
        )
    end

    return DataFrame(rows)
end

"""
    ecm_params_to_df(models, sols, params_real)

Terminal ECM parameter estimates (RC resistance, time constant, Arrhenius k)
compared to ground truth where available. One row per cell.
"""
function ecm_params_to_df(models, sols, params_real)
    ids = sort(collect(keys(models)))

    rows = map(ids) do id
        zt = models[id].kf.p.zt
        xid = models[id].kf.p.xid
        xs = ComponentVector(sols[id].xt[end], xid)
        r = StatsBase.reconstruct(zt.r, [abs(xs.rc.r)]) |> first  # Ω
        τ = abs(xs.rc.τ)                                           # s
        k = abs(xs.arr.k)
        real = params_real["cell_$id"]

        (;
            cell = id,
            r_rc_mΩ = r * 1.0e3,
            r_rc_true_mΩ = get(real, "R1", NaN) * 1.0e3,
            tau_s = τ,
            tau_true_s = get(real, "tau1", NaN),
            k_arr = k,
        )
    end

    return DataFrame(rows)
end

"""
    voltage_accuracy_to_df(models, sols, data)

Voltage prediction accuracy (mV): closed-loop (from `sols`) vs open-loop
(KF reinitialised from terminal state, re-run with tt=0). One row per cell.
"""
function voltage_accuracy_to_df(models, sols, data)
    ids = sort(collect(keys(models)))
    zt = fit_zscore()

    sols_ol = Dict()
    for batch in Iterators.partition(ids, Threads.nthreads())
        tasks = Dict(
            id => Threads.@spawn begin
                    u, y = cell_dataset(data, id)
                    m = deepcopy(models[id])
                    reinit_kf!(m.kf; x = sols[id].xt[end], R = sols[id].Rt[end])
                    run_kf!(m, u, y; tt = 0)
                end for id in batch
        )

        for (id, task) in tasks
            sols_ol[id] = fetch(task)
        end
    end

    rows = map(ids) do id
        mcl = _volt_metrics(sols[id].et, zt)
        mol = _volt_metrics(sols_ol[id].et, zt)

        (;
            cell = id,
            rmse_cl = mcl.rmse, mae_cl = mcl.mae, max_cl = mcl.max,
            q95_cl = mcl.q95, q99_cl = mcl.q99,
            rmse_ol = mol.rmse, mae_ol = mol.mae, max_ol = mol.max,
            q95_ol = mol.q95, q99_ol = mol.q99,
        )
    end

    return DataFrame(rows)
end

"""
    q_estimation_to_df(data, sols_state, param_cells_state, params_real)

Charge and SOC estimation accuracy from YuasaStateModel: KF estimate vs CMU
coulomb counting, both compared to oscilloscope ground truth. One row per cell.
Charge errors in Ah, SOC errors in %.
"""
function q_estimation_to_df(data, sols_state, param_cells_state)
    ids = sort(collect(keys(sols_state)))
    zt = fit_zscore()
    Ts = 1.0

    rows = map(ids) do id
        sol = sols_state[id]
        qμ = StatsBase.reconstruct(zt.q, first.(sol.xt))
        q_cc = cumsum(data.î) .* Ts ./ 3600
        q_ref = data.q

        mkf = _err_metrics(qμ .- q_ref)
        mcc = _err_metrics(q_cc .- q_ref)

        Q_est = Measurements.value(param_cells_state[id][:Q])
        s0_est = Measurements.value(param_cells_state[id][:soc])
        soc_kf = s0_est .+ qμ ./ Q_est
        soc_cc = s0_est .+ q_cc ./ Q_est
        soc_ref = data[:, "soc_cell_$id"]

        skf = _err_metrics((soc_kf .- soc_ref) .* 100)
        scc = _err_metrics((soc_cc .- soc_ref) .* 100)

        (;
            cell = id,
            # q_rmse_kf=mkf.rmse, q_mae_kf=mkf.mae, q_max_kf=mkf.max,
            # q_q95_kf=mkf.q95, q_q99_kf=mkf.q99,
            # q_rmse_cc=mcc.rmse, q_mae_cc=mcc.mae, q_max_cc=mcc.max,
            # q_q95_cc=mcc.q95, q_q99_cc=mcc.q99,
            soc_rmse_kf = skf.rmse, soc_mae_kf = skf.mae, soc_max_kf = skf.max,
            # soc_q95_kf=skf.q95, soc_q99_kf=skf.q99,
            soc_rmse_cc = scc.rmse, soc_mae_cc = scc.mae, soc_max_cc = scc.max,
            # soc_q95_cc=scc.q95, soc_q99_cc=scc.q99,
        )
    end

    return DataFrame(rows)
end
