
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

