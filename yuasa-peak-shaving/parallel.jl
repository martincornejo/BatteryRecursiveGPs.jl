function spawn_fit(data, ti, θ, zt, id)
    (; u, y) = cell_dataset(data, ti, zt, id.m, id.c)
    return Threads.@spawn fit_model(u, y, θ, zt, id)
end


function fit_models_threaded(data, ti, θ, zt, ids)
    models = Dict()
    sols = Dict()

    for batch in Iterators.partition(ids, Threads.nthreads())
        tasks = Dict(id => spawn_fit(data, ti, θ, zt, id) for id in batch)
        for (id, task) in tasks
            try
                (; model, sol) = fetch(task)
                models[id] = model
                sols[id] = sol
                @info "Cell m:$(id.m), c:$(id.c) complete"
            catch e
                @error "Cell m:$(id.m), c:$(id.c) failed" exception = e
            end
        end
    end

    return (; models, sols)
end

function remotecall_fit(pool::WorkerPool, data, ti, θ, zt, id)
    (; u, y) = cell_dataset(data, ti, zt, id.m, id.c)
    return remotecall(fit_model, pool, u, y, θ, zt, id)
end

function fit_models_distributed(data, ti, θ, zt, ids)
    pool = WorkerPool(workers())
    tasks = Dict(id => remotecall_fit(pool, data, ti, θ, zt, id) for id in ids)

    models = Dict()
    sols = Dict()

    @sync for (id, task) in tasks
        @async try
            (; model, sol) = fetch(task)
            models[id] = model
            sols[id] = sol
            @info "Cell m:$(id.m), c:$(id.c) complete"
        catch e
            @error "Cell m:$(id.m), c:$(id.c) failed" exception = e
        end
    end

    return (; models, sols)
end


function fit_models_distributed(data, ti, ids)
    zt = fit_zscore()
    θ = (;
        ocv = (; σ = 0.5, ℓ = 0.5),
        r0 = (; σ = 0.01, ℓ = 2.0),
        vσ = 3.0e-3,
        Ts = 1.0,
        r0μ = 1.5e-3,
        rc = (;
            v0 = 0.0, σ0_v = 1.0e-5, σ1_v = 5.0e-5,
            r0 = 1.5e-3, σ0_r = 5.0e-6, σ1_r = 0.0,
            τ0 = 250.0, σ0_τ = 1.0, σ1_τ = 0.0,
        ),
        cc = (; σ1 = 0.1e-5),
        arr = (; T0 = 25, k0 = 20, σ0_k = 0.0, σ1_k = 0.0),
    )
    return fit_models_distributed(data, ti, θ, zt, ids)
end

function fit_models_threaded(data, ti, ids)
    zt = fit_zscore()
    θ = (;
        ocv = (; σ = 0.5, ℓ = 0.5),
        r0 = (; σ = 0.01, ℓ = 2.0),
        vσ = 3.0e-3,
        Ts = 1.0,
        r0μ = 1.5e-3,
        rc = (;
            v0 = 0.0, σ0_v = 1.0e-5, σ1_v = 5.0e-5,
            r0 = 1.5e-3, σ0_r = 5.0e-6, σ1_r = 0.0,
            τ0 = 250.0, σ0_τ = 1.0, σ1_τ = 0.0,
        ),
        cc = (; σ1 = 0.1e-5),
        arr = (; T0 = 25, k0 = 20, σ0_k = 0.0, σ1_k = 0.0),
    )

    return fit_models_threaded(data, ti, θ, zt, ids)
end
