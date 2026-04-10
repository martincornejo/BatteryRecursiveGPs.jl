
function fit_zscore(n=1)
    v = StatsBase.fit(ZScoreTransform, (n*3.3):(n*0.01):(n*4.1))
    σ = StatsBase.fit(ZScoreTransform, (n*3.3):(n*0.01):(n*4.1), center=false)
    i = StatsBase.fit(ZScoreTransform, -50:0.1:50, center=false)
    q = StatsBase.fit(ZScoreTransform, -50:0.1:50, center=false)
    r = ZScoreTransform(1, 1, [0.0], [σ.scale[1] / i.scale[1]])
    return (; v, σ, i, q, r)
end

function cell_dataset(data::DataFrame, cell_id::Int; Ts=1.0)
    zt = fit_zscore()

    v̂ = StatsBase.transform(zt.v, data[:, "v_cell_$cell_id"])
    î = StatsBase.transform(zt.i, data.î)
    # q̂ = StatsBase.transform(zt.q, data.q)
    q̂ = StatsBase.transform(zt.q, cumsum(data.î) * Ts / 3600)
    T = data.T

    u = [(; i, q, T) for (i, q, T) in zip(î, q̂, T)]
    y = [SA[v] for v in v̂]

    (; u, y)
end

function fit_model(df::DataFrame, cell_id::Int, θ; n=21, pad=0.05)
    u, y = cell_dataset(df, cell_id)
    zt = fit_zscore()
    model = YuasaModel(θ, u, zt; n, pad)

    stats = @timed begin
        sol = run_kf!(model, u, y)
    end

    @info "Cell $(cell_id): complete" stats.time

    (; model, sol)
end

function fit_models(data, ids, θ; n=21, pad=0.05)
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

    (; models, sols)
end

function refine_model(data::DataFrame, cell_id::Int, model, sol; Ts=1.0, σ1_cc=nothing)
    u, y = cell_dataset(data, cell_id; Ts)

    model_new = deepcopy(model)
    reinit_kf!(model_new.kf; x=sol.xt[end], R=sol.Rt[end])

    if σ1_cc !== nothing
        model_new.kf.R1[:cc, :cc] .= [σ1_cc^2;;]
    end

    stats = @timed begin
        sol_new = run_kf!(model_new, u, y)
    end

    @info "Cell $(cell_id): refined" stats.time
    (; model=model_new, sol=sol_new)
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

    (; models=r_models, sols=r_sols)
end



function plot_q_estimation_state(data, sol)
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]
    zt = fit_zscore()
    q̂ = first.(sol.xt)
    R̂ = first.(sol.Rt)
    t = sol.idx * 1.0
    qμ = StatsBase.reconstruct(zt.q, q̂)
    qσ = StatsBase.reconstruct(zt.q, sqrt.(R̂)) # use uncertainty as estimation?
    lines!(ax[1], t, qμ, label="estimated")
    band!(ax[1], t, qμ - qσ, qμ + qσ, alpha=0.5)
    lines!(ax[1], t, data.q)

    q2 = cumsum(data.î) * 1.0 / 3600

    lines!(ax[2], t, qμ - data.q)
    lines!(ax[2], t, q2 - data.q)

    rmse1 = mean(abs2, qμ - data.q) |> sqrt
    rmse2 = mean(abs2, q2 - data.q) |> sqrt
    mae1 = mean(abs, qμ - data.q)
    mae2 = mean(abs, q2 - data.q)
    @info "RMSE" rmse1 rmse2
    @info "MAE" mae1 mae2

    fig
end

# report_q_estimation(data, sol, cell_id, param_cells)
# Part 1 — relative charge tracking: KF estimate and CMU coulomb-counting vs data.q (oscilloscope)
# Part 2 — absolute SOC: SOC(t) = s0_est + q(t)/Q_est vs soc_cell_<id> ground truth
function report_q_estimation(data, sol, cell_id, param_cells)
    zt  = fit_zscore()
    Ts  = 1.0

    qμ    = StatsBase.reconstruct(zt.q, first.(sol.xt))
    q_cmu = cumsum(data.î) .* Ts ./ 3600
    q_ref = data.q

    err_kf  = qμ    .- q_ref
    err_cmu = q_cmu .- q_ref

    _row(label, e) = print(@sprintf(
        "  %-18s  RMSE=%7.4f Ah   MAE=%7.4f Ah   max|e|=%7.4f Ah   span=%7.4f Ah   final=%+7.4f Ah\n",
        label, sqrt(mean(abs2, e)), mean(abs, e), maximum(abs, e), maximum(e) - minimum(e), e[end]))

    println("=== Relative charge tracking — cell $cell_id ===")
    _row("KF estimate", err_kf)
    _row("CMU counting", err_cmu)

    if haskey(param_cells, cell_id)
        Q_est  = Measurements.value(param_cells[cell_id][:Q])
        s0_est = Measurements.value(param_cells[cell_id][:soc])

        soc_kf   = s0_est .+ qμ    ./ Q_est
        soc_cmu  = s0_est .+ q_cmu ./ Q_est
        soc_true = data[:, "soc_cell_$cell_id"]

        e_kf_soc  = (soc_kf  .- soc_true) .* 100
        e_cmu_soc = (soc_cmu .- soc_true) .* 100

        _row_soc(label, e) = print(@sprintf(
            "  %-18s  RMSE=%6.3f %%    MAE=%6.3f %%    max|e|=%6.3f %%    final=%+6.3f %%\n",
            label, sqrt(mean(abs2, e)), mean(abs, e), maximum(abs, e), e[end]))

        println()
        print(@sprintf("=== Absolute SOC — cell %d  (Q_est=%.2f Ah, s0_est=%.2f%%) ===\n",
            cell_id, Q_est, s0_est * 100))
        _row_soc("KF estimate", e_kf_soc)
        _row_soc("CMU counting", e_cmu_soc)
    end
end

function plot_ocv_diagnostics(models, sols, param_cells, params_real, focv)
    fig = Figure(size=(800, 500))
    ax = [Axis(fig[i, j]) for i in 1:2, j in 1:2]

    titles = ["Estimated params", "Real params"]
    for j in 1:2
        ax[1, j].title = titles[j]
        ax[1, j].ylabel = "OCV / V"
        ax[2, j].ylabel = "Mismatch / mV"
        ax[2, j].xlabel = "SOC / p.u."
        hidexdecorations!(ax[1, j]; ticks=false, grid=false)
    end

    s = 0.0:0.005:1.0
    for j in 1:2
        lines!(ax[1, j], s, focv.(s); color=:black, linestyle=:dash)
        hlines!(ax[2, j], [0]; color=:black, linestyle=:dash)
    end

    for (id, model) in models
        (; q, μ, σ) = gp_ocv(model, sols[id])

        Q_est = Measurements.value(param_cells[id][:Q])
        s0_est = Measurements.value(param_cells[id][:soc])
        soc_est = s0_est .+ q ./ Q_est

        Q_r = params_real["cell_$id"]["Q"]
        s0_r = params_real["cell_$id"]["soc"]
        soc_r = s0_r .+ q ./ Q_r

        for (j, soc) in enumerate((soc_est, soc_r))
            lines!(ax[1, j], soc, μ)
            band!(ax[1, j], soc, μ .- 2σ, μ .+ 2σ; alpha=0.3)
            lines!(ax[2, j], soc, (μ .- focv.(soc)) .* 1000)
        end
    end

    linkxaxes!(ax[1, 1], ax[2, 1])
    linkxaxes!(ax[1, 2], ax[2, 2])
    linkyaxes!(ax[1, 1], ax[1, 2])
    linkyaxes!(ax[2, 1], ax[2, 2])
    fig
end


function report_params(param_cells, params_real)
    ids = sort(collect(keys(param_cells)))
    header = @sprintf("%-6s  %-8s  %-8s  %-8s  %-8s  %-8s  %-8s",
        "cell", "Q_real", "Q_est", "Q_err", "s0_real", "s0_est", "s0_err")
    sep = "-"^70
    println(sep)
    println(header)
    println(sep)
    for id in ids
        Q_real = params_real["cell_$id"]["Q"]
        s0_real = params_real["cell_$id"]["soc"] * 100

        Q_est  = Measurements.value(param_cells[id][:Q])
        s0_est = Measurements.value(param_cells[id][:soc]) * 100

        Q_err  = Q_est - Q_real
        s0_err = s0_est - s0_real

        @printf("%-6d  %-8.3f  %-8.3f  %-+8.3f  %-8.2f  %-8.2f  %-+8.2f\n",
            id, Q_real, Q_est, Q_err, s0_real, s0_est, s0_err)
    end
    println(sep)
    Q_errs  = [Measurements.value(param_cells[id][:Q]) - params_real["cell_$id"]["Q"] for id in ids]
    s0_errs = [(Measurements.value(param_cells[id][:soc]) - params_real["cell_$id"]["soc"]) * 100 for id in ids]
    @printf("%-6s  %-8s  %-8s  %-+8.3f  %-8s  %-8s  %-+8.2f\n",
        "RMSE", "", "", sqrt(mean(abs2, Q_errs)), "", "", sqrt(mean(abs2, s0_errs)))
    println(sep)
end

function report_ocv_residuals(models, sols, param_cells, params_real, focv)
    ids = sort(collect(keys(param_cells)))
    header = @sprintf("%-6s  %-12s  %-12s  %-12s  %-12s",
        "cell", "max|r|_est", "mean|r|_est", "max|r|_real", "mean|r|_real")
    sep = "-"^62
    println(sep)
    println(header)
    println(" "^8 * "(mV)" * " "^9 * "(mV)" * " "^9 * "(mV)" * " "^9 * "(mV)")
    println(sep)
    for id in ids
        (; q, μ, σ) = gp_ocv(models[id], sols[id])

        Q_est  = Measurements.value(param_cells[id][:Q])
        s0_est = Measurements.value(param_cells[id][:soc])
        soc_est = s0_est .+ q ./ Q_est

        Q_r   = params_real["cell_$id"]["Q"]
        s0_r  = params_real["cell_$id"]["soc"]
        soc_r = s0_r .+ q ./ Q_r

        # Only evaluate focv within its interpolation domain
        slims = extrema(focv.t)
        mask_est  = findall(slims[1] .<= soc_est .<= slims[2])
        mask_real = findall(slims[1] .<= soc_r   .<= slims[2])

        r_est  = (μ[mask_est]  .- focv.(soc_est[mask_est]))  .* 1000
        r_real = (μ[mask_real] .- focv.(soc_r[mask_real]))   .* 1000

        @printf("%-6d  %-12.2f  %-12.2f  %-12.2f  %-12.2f\n",
            id,
            maximum(abs, r_est),  mean(abs, r_est),
            maximum(abs, r_real), mean(abs, r_real))
    end
    println(sep)
end

function report_wls_diagnostics(models, sols, fsoc, focv, params_real; ids=1:3)
    for id in ids
        Q_real = params_real["cell_$id"]["Q"]
        s0_real = params_real["cell_$id"]["soc"] * 100
        @printf("Cell %d  (Q_real=%.3f  s0_real=%.2f%%):\n", id, Q_real, s0_real)
        report_Q_profile(models[id], sols[id], fsoc, focv; Q_range=(Q_real - 5):1.0:(Q_real + 15))
        println()
    end
end

"""
    report_Q_profile(model, sol, fsoc, focv; Q_range=75:0.5:115, n_grid=200)

For each Q in Q_range, find the optimal s0 (closed-form GLS) and compute the
RMS voltage residual. Prints a profile to identify whether the objective is
unimodal and where the global minimum is.
"""
function report_Q_profile(model, sol, fsoc, focv; Q_range=75:0.5:115)
    # gp_ocv returns (q, μ, σ) in physical units (Ah, V, V)
    (; q, μ) = gp_ocv(model, sol)

    fsoc_lims = extrema(fsoc.t)
    focv_lims = extrema(focv.t)
    v_low = max(fsoc_lims[1], minimum(μ))
    v_up  = min(fsoc_lims[2], maximum(μ))
    idxs  = findall(v_low .<= μ .<= v_up)
    q_f   = q[idxs]
    μ_f   = μ[idxs]

    soc_gp = fsoc.(μ_f)  # reference SOC at each GP voltage

    println(@sprintf("  %-8s  %-8s  %-10s", "Q", "s0(%)", "rmse(mV)"))
    println("  " * "-"^30)
    for Q in Q_range
        s0 = mean(soc_gp .- q_f ./ Q)
        soc_model = s0 .+ q_f ./ Q
        valid = findall(focv_lims[1] .<= soc_model .<= focv_lims[2])
        isempty(valid) && continue
        r = μ_f[valid] .- focv.(soc_model[valid])
        rmse = sqrt(mean(abs2, r)) * 1000
        @printf("  %-8.1f  %-8.3f  %-10.3f\n", Q, s0 * 100, rmse)
    end
end

