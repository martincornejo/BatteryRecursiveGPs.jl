"""
    fit_model(make_model, u, y, θ, zt) -> (; model, sol, time)

Build a model with `make_model(θ, u, zt)` and run it once over `u` and `y`, closed loop, so
the returned `model` carries the identified posterior. `sol` is the reduced solution and
`time` the elapsed seconds of the filter run.
"""
function fit_model(make_model, u, y, θ, zt)
    model = make_model(θ, u, zt)
    stats = @timed begin
        sol = run_kf!(model, u, y)
    end
    sol = reduce_sol(model, sol)
    return (; model, sol, time = stats.time)
end

"""
    fit_ocv_curve(make_model, u, y, θ, zt) -> (; q, μ)

Fit a model at a fully specified `θ` and return only the GP-OCV posterior mean over charge.
The slim payload for distributed hyperparameter sweeps: no solution object travels back, and
the worker needs no configuration beyond its arguments.
"""
function fit_ocv_curve(make_model, u, y, θ, zt)
    (; model, sol) = fit_model(make_model, u, y, θ, zt)
    ocv = gp_ocv(model, sol)
    return (; q = collect(ocv.q), μ = collect(ocv.μ))
end

"""
    eval_model(model, sol) -> (; sol_eval, time)

Replay a fitted `model` over its own inputs open loop (`tt = 0`), giving the voltage it
predicts from the identified parameters alone. Calls [`reinit_kf!`](@ref) first, which resets
the charge and RC states while keeping the posterior — full models only, since the
`*StateModel` types define no `reinit_kf!` method.
"""
function eval_model(model, sol)
    (; u, y) = sol
    reinit_kf!(model)
    stats = @timed begin
        sol_eval = run_kf!(model, u, y; tt = 0)
    end
    sol_eval = reduce_sol(model, sol_eval)
    return (; sol_eval, time = stats.time)
end

"""
    fit_models_threaded(make_model, make_uy, ids, θ, zt) -> (; models, sols)

Fit one model per id on the available threads, `make_uy(id)` supplying that id's `(; u, y)`.
Failures are logged and the id dropped, so both dictionaries can be shorter than `ids`.
"""
function fit_models_threaded(make_model, make_uy, ids, θ, zt)
    models = Dict()
    sols = Dict()

    for batch in Iterators.partition(ids, Threads.nthreads())
        tasks = Dict(
            id => Threads.@spawn begin
                    (; u, y) = make_uy(id)
                    fit_model(make_model, u, y, θ, zt)
                end for id in batch
        )
        for (id, task) in tasks
            try
                (; model, sol) = fetch(task)
                models[id] = model
                sols[id] = sol
                @info "id=$id complete"
            catch e
                @error "id=$id failed" exception = e
            end
        end
    end

    return (; models, sols)
end


"""
    fit_models_distributed(make_model, make_uy, ids, θ, zt) -> (; models, sols)

Fit one model per id across the available worker processes, `make_uy(id)` supplying that id's
`(; u, y)` on the master. Failures are logged and the id dropped, so both dictionaries can be
shorter than `ids`. Workers need the model type loaded, e.g. via
`@everywhere using BatteryRecursiveGPs`.
"""
function fit_models_distributed(make_model, make_uy, ids, θ, zt)
    pool = WorkerPool(workers())
    tasks = Dict(
        id => begin
                (; u, y) = make_uy(id)
                remotecall(fit_model, pool, make_model, u, y, θ, zt)
            end for id in ids
    )

    models = Dict()
    sols = Dict()

    @sync for (id, task) in tasks
        @async try
            (; model, sol) = fetch(task)
            models[id] = model
            sols[id] = sol
            @info "id=$id complete"
        catch e
            @error "id=$id failed" exception = e
        end
    end

    return (; models, sols)
end
