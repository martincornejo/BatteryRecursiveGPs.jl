function run_open_loop_eval(data, ti, models, ids)
    pool = WorkerPool(workers())
    tasks = Dict(
        id => begin
            zt = models[id].kf.p.zt
            (; u, y) = cell_dataset(data, ti, zt, id.m, id.c)
            model_ol = init_open_loop(models[id], y)
            remotecall((m, u, y) -> reduce_sol(m, run_kf!(m, u, y; tt = 0)), pool, model_ol, u, y)
        end
        for id in ids
    )

    sols = Dict()
    @sync for (id, task) in tasks
        @async try
            sols[id] = fetch(task)
        catch e
            @error "id=$id open-loop eval failed" exception = e
        end
    end

    return sols
end

function calc_accuracy_metrics(sol)
    e = first.(sol.yt) .- first.(sol.yμ)
    rmse = sqrt(mean(e .^ 2))
    mae = mean(abs.(e))
    return (; rmse, mae)
end

function calc_accuracy_dict(sols, ids)
    rmse = Dict(id => calc_accuracy_metrics(sols[id]).rmse for id in ids)
    mae = Dict(id => calc_accuracy_metrics(sols[id]).mae for id in ids)
    return (; rmse, mae)
end

# Compare the KF charge state with the coulomb-counted charge from the input
# current. Divergence indicates a bias in the current measurement (the KF q is
# pulled by voltage corrections; the cumulative current is not). Returns
# physical Ah, with both anchored at q[1] = 0.
function q_drift(model, sol)
    zt = model.kf.p.zt
    t = sol.idx ./ 3600
    q_kf = StatsBase.reconstruct(zt.q, sol.qμ)
    q_cc = StatsBase.reconstruct(zt.q, [u.q for u in sol.ut])
    q_kf .-= q_kf[1]
    q_cc .-= q_cc[1]
    Δ = q_kf .- q_cc
    bias_A = (Δ[end] - Δ[1]) / (t[end] - t[1])  # Ah/h ≡ A
    return (; t, q_kf, q_cc, Δ, bias_A)
end
