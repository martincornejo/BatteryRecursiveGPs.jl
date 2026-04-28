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


function fit_zscore(n = 1)
    v = StatsBase.fit(ZScoreTransform, (n * 3.3):(n * 0.01):(n * 4.1))
    σ = StatsBase.fit(ZScoreTransform, (n * 3.3):(n * 0.01):(n * 4.1), center = false)
    i = StatsBase.fit(ZScoreTransform, -50:0.1:50, center = false)
    q = StatsBase.fit(ZScoreTransform, -50:0.1:50, center = false)
    r = ZScoreTransform(1, 1, [0.0], [σ.scale[1] / i.scale[1]])
    return (; v, σ, i, q, r)
end


function cell_dataset(data, ti, p, m, c; Ts = 1.0, zt = fit_zscore())
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

    î = StatsBase.transform(zt.i, df_i.value)
    q̂ = StatsBase.transform(zt.q, q)
    T = df_T.value
    v̂ = df_v.value

    u = [(; i, q, T) for (i, q, T) in zip(î, q̂, T)]
    y = [SA[v] for v in v̂]

    return (; u, y)
end


function module_dataset(data, ti, p, m; Ts = 1.0, zt = fit_zscore(12))
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

    î = StatsBase.transform(zt.i, df_i.value)
    q̂ = StatsBase.transform(zt.q, q)
    T = df_T.value
    v̂ = df_v.value

    u = [(; i, q, T) for (i, q, T) in zip(î, q̂, T)]
    y = [SA[v] for v in v̂]

    return (; u, y)
end


function cell_dataset_osci(data, ti, c; Ts = 1.0, zt = fit_zscore())
    p = 1
    m = 9

    df_v = copy(data[:cell_voltage])
    select!(df_v, "_time" => "time", "cell_voltage_$(p)_$(m)_1_$(c)" => "value")
    subset!(df_v, :time => ByRow(∈(ti)))
    df_v[!, :value] = StatsBase.transform(zt.v, df_v.value)
    df_v = fill_missings(df_v, Second(Ts))

    df_î = CSV.File(datadir * "oscilloscope_p1_m9.csv"; dateformat = dateformat"y-m-dTH:M:S.sss+00:00") |> DataFrame
    df_î.timestamp_utc = floor.(df_î.timestamp_utc, Second(1))
    df_î = combine(groupby(df_î, :timestamp_utc), :MEAS1 => mean => :MEAS1)
    select!(df_î, :timestamp_utc => :_time, :MEAS1 => ByRow(x -> -x) => :i)
    subset!(df_î, :_time => ByRow(∈(ti)))
    df_î = ffill(df_î, Second(1))

    df_î.q = cumsum(df_î.i) * Ts / 3600

    î = StatsBase.transform(zt.i, df_î.i)
    q̂ = StatsBase.transform(zt.q, df_î.q)
    v̂ = df_v.value

    u = [(; i, q) for (i, q) in zip(î, q̂)]
    y = [SA[v] for v in v̂]

    return (; u, y)
end


"""
    fit_model_eval(make_model, u, y, θ, zt)

Closed-loop fit + open-loop evaluation in one go.  Returns a small payload
suitable for parallel sweeps: per-cell GP-OCV plus closed/open-loop residual
RMSEs. Open-loop run reuses the learned GP via `reinit_kf!` followed by
`run_kf!(...; tt=0)` which skips all `correct!` steps.
"""
function fit_model_eval(make_model, u, y, θ, zt)
    model = make_model(θ, u, zt)

    # Closed-loop fit: learns OCV/R0 GP
    sol_cl = run_kf!(model, u, y)
    cl_rmse = sqrt(mean(abs2, sol_cl.et))

    # Snapshot the GP-derived OCV before reinit/open-loop changes model.kf
    sol_cl_red = reduce_sol(model, sol_cl)
    cell = gp_ocv(model, sol_cl_red)

    # Open-loop sim using the learned GP (no voltage feedback)
    reinit_kf!(model)
    sol_ol = run_kf!(model, u, y; tt = 0)
    ol_rmse = sqrt(mean(abs2, sol_ol.et))

    return (; cell, cl_rmse, ol_rmse)
end


function extract_ocv(model::YuasaModel)
    kf = model.kf
    zt = kf.p.zt
    q̂min, q̂max = extrema(kf.p.r0.b0)
    q̂ = q̂min:0.01:q̂max
    q = StatsBase.reconstruct(zt.q, q̂)

    soc = range(0.15, 0.9, length = length(q))

    ocv = predict_gp(kf, q̂, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)

    focv = LinearInterpolation(ocvμ, soc)
    focv⁻¹ = LinearInterpolation(soc, ocvμ)

    return (; ocv = focv, ocv⁻¹ = focv⁻¹)
end
