function default_θ()
    return (;
        ocv = (; σ = 0.5, ℓ = 0.5),
        r0 = (; σ0 = 1.0e-5, σ1 = 0.0),
        vσ = 3.0e-3,
        Ts = 1.0,
        r0μ = 1.5e-3,
        rc = (;
            v0 = 0.0, σ0_v = 1.0e-5, σ1_v = 5.0e-5,
            r0 = 1.5e-3, σ0_r = 10.0e-6, σ1_r = 0.0,
            τ0 = 250.0, σ0_τ = 10.0, σ1_τ = 0.0,
        ),
        cc = (; σ1 = 0.1e-5),
        arr = (; T0 = 25, k0 = 2000, σ0_k = 0.0, σ1_k = 0.0),
    )
end

function default_θ_2rc()
    return (;
        ocv = (; σ = 0.5, ℓ = 0.5),
        r0 = (; σ0 = 1.0e-5, σ1 = 0.0),
        vσ = 3.0e-3,
        Ts = 1.0,
        r0μ = 1.5e-3,
        rc1 = (;
            v0 = 0.0, σ0_v = 1.0e-5, σ1_v = 5.0e-5,
            r0 = 0.75e-3, σ0_r = 10.0e-6, σ1_r = 0.0,
            τ0 = 30.0, σ0_τ = 5.0, σ1_τ = 0.0,
        ),
        rc2 = (;
            v0 = 0.0, σ0_v = 1.0e-5, σ1_v = 5.0e-5,
            r0 = 0.75e-3, σ0_r = 10.0e-6, σ1_r = 0.0,
            τ0 = 500.0, σ0_τ = 20.0, σ1_τ = 0.0,
        ),
        cc = (; σ1 = 0.1e-5),
        arr = (; T0 = 25, k0 = 2000, σ0_k = 0.0, σ1_k = 0.0),
    )
end

function fit_cells(data, ti, ids; model_type = FeneconModel, θ = default_θ(), zt = fit_zscore(), distributed = true)
    make_uy = id -> cell_dataset(data, ti, zt, id.m, id.c)
    runner = distributed ? fit_models_distributed : fit_models_threaded
    return runner(model_type, make_uy, ids, θ, zt)
end

state_model(m::FeneconModel; q0, Ts, θ) = FeneconStateModel(m; q0, Ts, θ)
state_model(m::Fenecon2RCModel; q0, Ts, θ) = Fenecon2RCStateModel(m; q0, Ts, θ)

function estimate_states_distributed(data, ti, models, ids; θ, zt = fit_zscore())
    pool = WorkerPool(workers())

    tasks = Dict(
        id => begin
                (; u, y) = cell_dataset(data, ti, zt, id.m, id.c)
                idx_first = findfirst(yᵢ -> !any(yᵢ .=== missing), y)
                q0 = estimate_initial_charge(models[id], first(y[idx_first]))
                sm = state_model(models[id]; q0, Ts = 1.0, θ)
                remotecall(
                    (m, u, y) -> begin
                        sol = run_kf!(m, u, y)
                        (; model = m, sol)
                    end, pool, sm, u, y
                )
            end
            for id in ids
    )

    states = Dict()
    @sync for (id, task) in tasks
        @async try
            (; model, sol) = fetch(task)
            states[id] = (; model, sol)
            @info "id=$id state estimation complete"
        catch e
            @error "id=$id state estimation failed" exception = e
        end
    end

    return states
end

function smooth_states_distributed(data, ti, models, ids; θ, zt = fit_zscore())
    pool = WorkerPool(workers())

    tasks = Dict(
        id => begin
                (; u, y) = cell_dataset(data, ti, zt, id.m, id.c)
                idx_first = findfirst(yᵢ -> !any(yᵢ .=== missing), y)
                q0 = estimate_initial_charge(models[id], first(y[idx_first]))
                sm = state_model(models[id]; q0, Ts = 1.0, θ)
                remotecall(
                    (m, u, y) -> begin
                        smoothed = smooth_kf!(m, u, y)
                        (; model = m, smoothed)
                    end, pool, sm, u, y
                )
            end
            for id in ids
    )

    states = Dict()
    @sync for (id, task) in tasks
        @async try
            (; model, smoothed) = fetch(task)
            states[id] = (; model, smoothed)
            @info "id=$id smoothing complete"
        catch e
            @error "id=$id smoothing failed" exception = e
        end
    end

    return states
end

function refine_cells_distributed(data, ti, models, ids; zt = fit_zscore())
    pool = WorkerPool(workers())
    make_uy = id -> cell_dataset(data, ti, zt, id.m, id.c)

    tasks = Dict(
        id => begin
                (; u, y) = make_uy(id)
                model_init = init_refine(models[id], y)
                remotecall(
                    (m, u, y) -> begin
                        sol = run_kf!(m, u, y)
                        (; model = m, sol = reduce_sol(m, sol))
                    end, pool, model_init, u, y
                )
            end
            for id in ids
    )

    models_refined = Dict()
    sols_refined = Dict()

    @sync for (id, task) in tasks
        @async try
            (; model, sol) = fetch(task)
            models_refined[id] = model
            sols_refined[id] = sol
            @info "id=$id refinement complete"
        catch e
            @error "id=$id refinement failed" exception = e
        end
    end

    return (; models_refined, sols_refined)
end
