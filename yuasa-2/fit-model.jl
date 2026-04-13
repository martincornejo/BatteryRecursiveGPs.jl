replace_missing!(v) = accumulate!((n0, n1) -> ismissing(n1) ? n0 : n1, v, v, init = zero(eltype(v)))

function ffill(df, ts = Second(1))
    t0 = first(df._time)
    t1 = last(df._time)

    df2 = DataFrame(_time = t0:ts:t1)
    leftjoin!(df2, df, on = :_time)
    sort!(df2, :_time)


    for col in names(df2[:, Not(:_time)])
        replace_missing!(df2[!, col])
    end

    disallowmissing!(df2)

    return df2
end


function fill_missings(df, ts = Second(1); time = :time)
    t0 = first(df[:, time])
    t1 = last(df[:, time])

    df2 = DataFrame(time => t0:ts:t1)
    leftjoin!(df2, df, on = time)
    sort!(df2, time)

    return df2
end

function interpolate(df, ti; Ts = 1)
    t = Dates.value.(df.time - first(ti)) .÷ 1000
    tr = 0:Ts:(Dates.value(last(ti) - first(ti)) .÷ 1000)
    f = LinearInterpolation(df.value, t)

    dt = first(ti):Second(Ts):last(ti)

    return DataFrame(; time = dt, t = tr, value = f(tr))
end


function cell_dataset(data, ti, p, m, c; Ts = 1.0)
    # df_i = ffill(data[:module_current], Second(1))
    # df_v = ffill(data[:cell_voltage], Second(10))
    zt = fit_zscore()

    # voltage
    df_v = copy(data[:cell_voltage])
    select!(df_v, "_time" => "time", "cell_voltage_$(p)_$(m)_1_$(c)" => "value")
    subset!(df_v, :time => ByRow(∈(ti)))
    df_v[!, :value] = StatsBase.transform(zt.v, df_v.value)
    df_v = fill_missings(df_v, Second(Ts))

    # current
    df_i = copy(data[:module_current])
    select!(df_i, "_time" => "time", "module_average_current_$(p)_$(m)" => ByRow(x -> -x) => "value")
    df_i = interpolate(df_i, ti; Ts)

    # coulomb counting
    q = cumsum(df_i.value) * Ts / 3600

    # temperature
    df_T = copy(data[:battery_temperature])
    select!(df_T, "_time" => "time", "battery_sensor_temperature_$(p)_$(m)_1" => "value")
    df_T = interpolate(df_T, ti; Ts)

    î = StatsBase.transform(zt.i, df_i.value)
    q̂ = StatsBase.transform(zt.q, q)
    T = df_T.value
    v̂ = df_v.value

    u = [(; i, q, T) for (i, q, T) in zip(î, q̂, T)]
    y = [SA[v] for v in v̂]

    return (; u, y)
end


function module_dataset(data, ti, p, m; Ts = 1.0)
    zt = fit_zscore(12)

    # voltage
    df_v = copy(data[:module_voltage])
    select!(df_v, "_time" => "time", "module_voltage_$(p)_$(m)" => "value")
    df_v[!, :value] = StatsBase.transform(zt.v, df_v.value)
    df_v = fill_missings(df_v, Second(Ts))
    subset!(df_v, :time => ByRow(∈(ti)))

    # current
    df_i = copy(data[:module_current])
    select!(df_i, "_time" => "time", "module_average_current_$(p)_$(m)" => ByRow(x -> -x) => "value")
    df_i = interpolate(df_i, ti; Ts = 1.0)

    # coulomb counting
    q = cumsum(df_i.value) * Ts / 3600

    # temperature
    df_T = copy(data[:battery_temperature])
    select!(df_T, "_time" => "time", "battery_sensor_temperature_$(p)_$(m)_1" => "value")
    df_T = interpolate(df_T, ti; Ts)

    î = StatsBase.transform(zt.i, df_i.value)
    q̂ = StatsBase.transform(zt.q, q)
    T = df_T.value
    v̂ = df_v.value

    u = [(; i, q, T) for (i, q, T) in zip(î, q̂, T)]
    y = [SA[v] for v in v̂]

    return (; u, y)
end


function cell_dataset_osci(data, ti, c)
    # df_i = ffill(data[:module_current], Second(1))
    p = 1
    m = 9
    zt = fit_zscore()

    df_v = copy(data[:cell_voltage])
    select!(df_v, "_time" => "time", "cell_voltage_$(p)_$(m)_1_$(c)" => "value")
    # df_v[!, :time] .= df_v[!, :time] .- Second(10)
    subset!(df_v, :time => ByRow(∈(ti)))
    df_v[!, :value] = StatsBase.transform(zt.v, df_v.value)
    df_v = fill_missings(df_v, Second(Ts))

    df_î = CSV.File(datadir * "oscilloscope_p1_m9.csv"; dateformat = dateformat"y-m-dTH:M:S.sss+00:00") |> DataFrame
    df_î.timestamp_utc = floor.(df_î.timestamp_utc, Second(1))
    df_î = combine(groupby(df_î, :timestamp_utc), :MEAS1 => mean => :MEAS1)
    select!(df_î, :timestamp_utc => :_time, :MEAS1 => ByRow(x -> -x) => :i)
    subset!(df_î, :_time => ByRow(∈(ti)))
    df_î = ffill(df_î, Second(1))
    # select!(df_î, :_time => :time, :i => ByRow(x -> -x) => :i)

    Ts = 1.0
    df_î.q = cumsum(df_î.i) * Ts / 3600

    î = StatsBase.transform(zt.i, df_î.i)
    q̂ = StatsBase.transform(zt.q, df_î.q)

    # v̂ = StatsBase.transform(zt.v, df_v.v)
    v̂ = df_v.value

    u = [(; i, q) for (i, q) in zip(î, q̂)]
    y = [SA[v] for v in v̂]

    return (; u, y)
end

function fit_zscore(n = 1)
    v = StatsBase.fit(ZScoreTransform, (n * 3.3):(n * 0.01):(n * 4.1))
    σ = StatsBase.fit(ZScoreTransform, (n * 3.3):(n * 0.01):(n * 4.1), center = false)
    i = StatsBase.fit(ZScoreTransform, -50:0.1:50, center = false)
    q = StatsBase.fit(ZScoreTransform, -50:0.1:50, center = false)
    r = ZScoreTransform(1, 1, [0.0], [σ.scale[1] / i.scale[1]])
    return (; v, σ, i, q, r)
end


function fit_model(data, ti, id)
    θ0 = ComponentVector(;
        # tunable (hyper)params
        ocv = (; σ = 0.5, ℓ = 0.5),
        r0 = (; σ = 0.01, ℓ = 2.0),
        vσ = 3.0e-3,
    )
    ϑ = ComponentVector(;
        # non-tunable params
        Ts = 1.0,
        r0μ = 1.5e-3,
        rc = (;
            v0 = 0.0, σ0_v = 1.0e-5, σ1_v = 5.0e-5,
            r0 = 1.5e-3, σ0_r = 5.0e-6, σ1_r = 0.0,
            τ0 = 250.0, σ0_τ = 1.0, σ1_τ = 0.0,
        ),
        cc = (; σ1 = 0.1e-5),
        arr = (; T0 = 25, k0 = 2000, σ0_k = 0.0, σ1_k = 0.0),
    )
    θ = ComponentVector(; θ0..., ϑ...)

    (; p, m, c) = id
    u, y = cell_dataset(data, ti, p, m, c)
    zt = fit_zscore()
    model = YuasaModel(θ, u, zt)

    stats = @timed begin
        sol = run_kf!(model, u, y)
    end

    @info "Cell p:$(p), m:$(m), c:$(c) complete" stats.time

    return (; model, sol)
end

function fit_module(data, ti, id)
    n = 12
    θ0 = ComponentVector(;
        # tunable (hyper)params
        ocv = (; σ = 0.5, ℓ = 0.5),
        r0 = (; σ = 0.01, ℓ = 2.0),
        vσ = n * 3.0e-3,
    )
    ϑ = ComponentVector(;
        # non-tunable params
        Ts = 1.0,
        r0μ = n * 1.5e-3,
        rc = (;
            v0 = n * 0.0, σ0_v = n * 1.0e-5, σ1_v = n * 5.0e-5,
            r0 = n * 1.5e-3, σ0_r = n * 5.0e-6, σ1_r = n * 0.0,
            τ0 = 250.0, σ0_τ = 1.0, σ1_τ = 0.0,
        ),
        cc = (; σ1 = 0.1e-5),
        arr = (; T0 = 25, k0 = 20, σ0_k = 0.0, σ1_k = 0.0),
    )
    θ = ComponentVector(; θ0..., ϑ...)

    (; p, m) = id
    zt = fit_zscore(n)
    u, y = module_dataset(data, ti, p, m)
    model = YuasaModel(θ, u, zt)

    stats = @timed begin
        sol = run_kf!(model, u, y)
    end

    @info "Module p:$(p), m:$(m), complete" stats.time

    return (; model, sol)
end

function fit_modules(data, ti, ids)
    models = Dict()
    sols = Dict()

    for id in ids
        (; model, sol) = fit_module(data, ti, id)
        models[id] = model
        sols[id] = sol
    end

    return (; models, sols)
end

function fit_models(data, ti, ids)
    models = Dict()
    sols = Dict()

    for id in ids
        (; model, sol) = fit_model(data, ti, id)
        models[id] = model
        sols[id] = sol
    end

    return (; models, sols)
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

    return (; models, sols)
end


function extract_ocv(model::YuasaModel)
    kf = model.kf
    zt = kf.p.zt
    q̂min, q̂max = extrema(kf.p.r0.b0)
    q̂ = q̂min:0.01:q̂max
    q = StatsBase.reconstruct(zt.q, q̂)

    # Q = last(q) - first(q)
    # soc = (q .- first(q)) ./ Q

    soc = range(0.15, 0.9, length = length(q))
    # soc = range(0.05, 0.95, length=length(q))

    ocv = predict_gp(kf, q̂, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)

    focv = LinearInterpolation(ocvμ, soc)
    focv⁻¹ = LinearInterpolation(soc, ocvμ)

    return (; ocv = focv, ocv⁻¹ = focv⁻¹)
end
