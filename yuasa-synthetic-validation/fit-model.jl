
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

function fit_model(df::DataFrame, cell_id::Int, θ)
    u, y = cell_dataset(df, cell_id)
    zt = fit_zscore()
    model = YuasaModel(θ, u, zt)

    stats = @timed begin
        sol = run_kf!(model, u, y)
    end

    @info "Cell $(cell_id): complete" stats.time

    (; model, sol)
end

function fit_models(data, ids, θ)
    models = Dict()
    sols = Dict()

    for batch in Iterators.partition(ids, Threads.nthreads())
        tasks = Dict(id => Threads.@spawn fit_model(data, id, θ) for id in batch)

        for (id, task) in tasks
            (; model, sol) = fetch(task)
            models[id] = model
            sols[id] = sol
        end
    end

    (; models, sols)
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
