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

function interpolate(df; Ts = 1, time_col = :time, tref = first(df[:, time_col]))
    t1 = last(df[:, time_col])
    t = Dates.value.(df[!, time_col] - tref) .÷ 1000
    tr = 0:Ts:t[end]
    f = LinearInterpolation(df.value, t)

    dt = tref:Second(Ts):t1

    return DataFrame(; time = dt, t = tr, value = f(tr))
end

function cell_dataset(data, ti, zt, m, c; Ts = 1.0)
    # 60-second pre-buffer so interpolation has data before t = 0
    ti_ext = Interval(first(ti) - Second(60), last(ti))
    df_ext = subset(data, :timestamp_utc => ByRow(∈(ti_ext)))
    df_ext = resample(df_ext, Second(Ts); time_col = :timestamp_utc)
    df = subset(df_ext, :timestamp_utc => ByRow(∈(ti)))
    t0 = first(df.timestamp_utc)
    t1 = last(df.timestamp_utc)

    # voltage
    cell = @sprintf("%02d", c)
    df_v = select(df, :timestamp_utc, Symbol("m$(m)_cell$(cell)") => :value)
    dropmissing!(df_v, :value)
    df_v[!, :value] = StatsBase.transform(zt.v, df_v.value)
    df_v = fill_missings(df_v, Second(Ts); time_col = :timestamp_utc, t0, t1)

    # current
    df_i = select(df_ext, :timestamp_utc, Symbol("m$(m)_current") => ByRow(x -> -x) => :value)
    df_i = interpolate(df_i; Ts, time_col = :timestamp_utc, tref = t0)

    # coulomb counting
    q = cumsum(df_i.value) * Ts / 3600

    # temperature
    df_T = select(df_ext, :timestamp_utc, Symbol("m$(m)_temp01") => :value) # also temp02 available
    df_T = interpolate(df_T; Ts, time_col = :timestamp_utc, tref = t0)

    î = StatsBase.transform(zt.i, df_i.value)
    q̂ = StatsBase.transform(zt.q, q)
    T = df_T.value
    v̂ = df_v.value

    u = [(; i, q, T) for (i, q, T) in zip(î, q̂, T)]
    y = [SA[v] for v in v̂]

    return (; u, y)
end
