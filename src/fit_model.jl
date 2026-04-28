function fit_model(make_model, u, y, θ, zt)
    model = make_model(θ, u, zt)
    stats = @timed begin
        sol = run_kf!(model, u, y)
    end
    sol = reduce_sol(model, sol)
    return (; model, sol, time = stats.time)
end


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
