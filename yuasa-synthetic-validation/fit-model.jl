
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

function fit_model(df::DataFrame, cell_id::Int)
    θ0 = ComponentVector(; # tunable (hyper)params
        ocv=(; σ=0.5, ℓ=0.5),
        r0=(; σ=0.01, ℓ=2.0),
        vσ=3e-3,
    )
    ϑ = ComponentVector(; # non-tunable params
        Ts=1.0,
        r0μ=1.5e-3,
        rc=(;
            v0=0.0, σ0_v=1.0e-5, σ1_v=5.0e-5,
            r0=1.5e-3, σ0_r=5.0e-6, σ1_r=0.0,
            τ0=250.0, σ0_τ=1.0, σ1_τ=0.0,
        ),
        cc=(; σ1=0.1e-5),
        arr=(; T0=25, k0=20, σ0_k=0.0, σ1_k=0.0),
    )
    θ = ComponentVector(; θ0..., ϑ...)

    u, y = cell_dataset(df, cell_id)
    zt = fit_zscore()
    kf = build_kf(θ, u, zt)

    stats = @timed begin
        sol = run_kf!(kf, u, y)
    end

    @info "Cell $(cell_id): complete" stats.time

    (; kf, sol)
end

function fit_models(data, ids)
    kfs = Dict()
    sols = Dict()

    for batch in Iterators.partition(ids, Threads.nthreads())
        tasks = Dict(id => Threads.@spawn fit_model(data, id) for id in batch)

        for (id, task) in tasks
            (; kf, sol) = fetch(task)
            kfs[id] = kf
            sols[id] = sol
        end
    end

    (; kfs, sols)
end