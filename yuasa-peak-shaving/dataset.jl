function fit_zscore(n = 1)
    v = StatsBase.fit(ZScoreTransform, (n * 3.3):(n * 0.01):(n * 4.1))
    σ = StatsBase.fit(ZScoreTransform, (n * 3.3):(n * 0.01):(n * 4.1), center = false)
    i = StatsBase.fit(ZScoreTransform, -50:0.1:50, center = false)
    q = StatsBase.fit(ZScoreTransform, -50:0.1:50, center = false)
    r = ZScoreTransform(1, 1, [0.0], [σ.scale[1] / i.scale[1]])
    return (; v, σ, i, q, r)
end


function resample(df::DataFrame, period::TimePeriod; time_col = :time, val_cols = Not(time_col))
    df[!, time_col] = floor.(df[!, time_col], period)
    df = combine(groupby(df, time_col), val_cols .=> mean, renamecols = false)
    return df
end

function fill_missings(
        df::DataFrame, ts::TimePeriod = Second(1);
        time_col = :time,
        t0 = first(df[:, time_col]),
        t1 = last(df[:, time_col]),
    )
    df_time = DataFrame(time_col => t0:ts:t1)
    leftjoin!(df_time, df, on = time_col)
    sort!(df_time, time_col)

    return df_time
end

function interpolate(df; Ts = 1, time_col = :time)
    t0 = first(df[:, time_col])
    t1 = last(df[:, time_col])
    t = Dates.value.(df[!, time_col] - t0) .÷ 1000
    tr = 0:Ts:t[end]
    f = LinearInterpolation(df.value, t)

    dt = t0:Second(Ts):t1

    return DataFrame(; time = dt, t = tr, value = f(tr))
end

function cell_dataset(data, ti, zt, m, c; Ts = 1.0)
    df = subset(data, :timestamp_utc => ByRow(∈(ti)))
    df = resample(df, Second(Ts); time_col = :timestamp_utc)
    t0 = first(df.timestamp_utc)
    t1 = last(df.timestamp_utc)

    # voltage
    cell = @sprintf("%02d", c)
    df_v = select(df, :timestamp_utc, Symbol("m$(m)_cell$(cell)") => :value)
    dropmissing!(df_v, :value)
    df_v[!, :value] = StatsBase.transform(zt.v, df_v.value)
    df_v = fill_missings(df_v, Second(Ts); time_col = :timestamp_utc, t0, t1)

    # current
    df_i = select(df, :timestamp_utc, Symbol("m$(m)_current") => ByRow(x -> -x) => :value)
    df_i = interpolate(df_i; Ts, time_col = :timestamp_utc)

    # coulomb counting
    q = cumsum(df_i.value) * Ts / 3600

    # temperature
    df_T = select!(df, :timestamp_utc, Symbol("m$(m)_temp01") => :value) # also temp02 available
    df_T = interpolate(df_T; Ts, time_col = :timestamp_utc)

    î = StatsBase.transform(zt.i, df_i.value)
    q̂ = StatsBase.transform(zt.q, q)
    T = df_T.value
    v̂ = df_v.value

    u = [(; i, q, T) for (i, q, T) in zip(î, q̂, T)]
    y = [SA[v] for v in v̂]

    return (; u, y)
end
