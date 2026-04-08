
function fit_zscore(n=1)
    v = StatsBase.fit(ZScoreTransform, (n*3.3):(n*0.01):(n*4.1))
    σ = StatsBase.fit(ZScoreTransform, (n*3.3):(n*0.01):(n*4.1), center=false)
    i = StatsBase.fit(ZScoreTransform, -50:0.1:50, center=false)
    q = StatsBase.fit(ZScoreTransform, -50:0.1:50, center=false)
    r = ZScoreTransform(1, 1, [0.0], [σ.scale[1] / i.scale[1]])
    return (; v, σ, i, q, r)
end

function resample(df::DataFrame, period::TimePeriod; time_col=:time, val_cols=Not(time_col))
    df[!, time_col] = floor.(df[!, time_col], period)
    df = combine(groupby(df, time_col), val_cols .=> mean, renamecols=false)
    return df
end

function fill_missings(df::DataFrame, ts::TimePeriod=Second(1); time_col=:time)
    t0 = first(df[:, time_col])
    t1 = last(df[:, time_col])

    df_time = DataFrame(time_col => t0:ts:t1)
    leftjoin!(df_time, df, on=time_col)
    sort!(df_time, time_col)

    return df_time
end

function interpolate(df; Ts=1, time_col=:time)
    t0 = first(df[:, time_col])
    t1 = last(df[:, time_col])
    t = Dates.value.(df[!, time_col] - t0) .÷ 1000
    tr = 0:Ts:t[end]
    f = LinearInterpolation(df.value, t)

    dt = t0:Second(Ts):t1

    DataFrame(; time=dt, t=tr, value=f(tr))
end

function cell_dataset(data, ti, m, c; Ts=1.0)
    zt = fit_zscore()

    df = subset(data, :timestamp_utc => ByRow(∈(ti)))
    df = resample(df, Second(Ts); time_col=:timestamp_utc)

    # voltage
    cell = @sprintf("%02d", c)
    df_v = select(df, :timestamp_utc, Symbol("m$(m)_cell$(cell)") => :value)
    dropmissing!(df_v, :value)
    df_v[!, :value] = StatsBase.transform(zt.v, df_v.value)
    df_v = fill_missings(df_v, Second(Ts); time_col=:timestamp_utc)

    # current
    df_i = select(df, :timestamp_utc, Symbol("m$(m)_current") => ByRow(x -> -x) => :value)
    df_i = interpolate(df_i; Ts, time_col=:timestamp_utc)

    # coulomb counting
    q = cumsum(df_i.value) * Ts / 3600

    # temperature
    df_T = select!(df, :timestamp_utc, Symbol("m$(m)_temp01") => :value) # also temp02 available
    df_T = interpolate(df_T; Ts, time_col=:timestamp_utc)

    î = StatsBase.transform(zt.i, df_i.value)
    q̂ = StatsBase.transform(zt.q, q)
    T = df_T.value
    v̂ = df_v.value

    u = [(; i, q, T) for (i, q, T) in zip(î, q̂, T)]
    y = [SA[v] for v in v̂]

    (; u, y)
end

function fit_model(data, ti, id)
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

    (; m, c) = id
    u, y = cell_dataset(data, ti, m, c)
    zt = fit_zscore()
    model = YuasaModel(θ, u, zt)

    try
        stats = @timed begin
            sol = run_kf!(model, u, y)
        end
    catch e
        @error "Cell m:$(m), c:$(c) failed: $e"
    end

    @info "Cell m:$(m), c:$(c) complete" stats.time

    (; model, sol)
end

function fit_models_spawn(data, ti, ids)
    models = Dict()
    sols = Dict()

    for batch in Iterators.partition(ids, Threads.nthreads())
        tasks = Dict(id => Threads.@spawn fit_model(data, ti, id) for id in batch)

        for (id, task) in tasks
            (; model, sol) = fetch(task)
            models[id] = model
            sols[id] = sol
        end
    end

    (; models, sols)
end