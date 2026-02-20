replace_missing!(v) = accumulate!((n0, n1) -> ismissing(n1) ? n0 : n1, v, v, init=zero(eltype(v)))

function ffill(df, ts=Second(1))
    t0 = first(df._time)
    t1 = last(df._time)

    df2 = DataFrame(_time=t0:ts:t1)
    leftjoin!(df2, df, on=:_time)
    sort!(df2, :_time)


    for col in names(df2[:, Not(:_time)])
        replace_missing!(df2[!, col])
    end

    disallowmissing!(df2)

    df2
end


function fill_missings(df, ts=Second(1); time=:time)
    t0 = first(df[:, time])
    t1 = last(df[:, time])

    df2 = DataFrame(time => t0:ts:t1)
    leftjoin!(df2, df, on=time)
    sort!(df2, time)

    return df2
end

function interpolate(df, ti; Ts=1)
    t = Dates.value.(df.time - first(ti)) .÷ 1000
    tr = 0:Ts:(Dates.value(last(ti) - first(ti)).÷1000)
    f = LinearInterpolation(df.value, t)

    dt = first(ti):Second(Ts):last(ti)

    DataFrame(; time=dt, t=tr, value=f(tr))
end


function cell_dataset(data, ti, p, m, c; Ts=1.0)
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

    (; u, y)
end


function module_dataset(data, ti, p, m; Ts=1.0)
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
    df_i = interpolate(df_i, ti; Ts=1.0)

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

    (; u, y)
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

    df_î = CSV.File(datadir * "oscilloscope_p1_m9.csv"; dateformat=dateformat"y-m-dTH:M:S.sss+00:00") |> DataFrame
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

    (; u, y)
end

function fit_zscore(n=1)
    v = StatsBase.fit(ZScoreTransform, (n*3.3):(n*0.01):(n*4.1))
    σ = StatsBase.fit(ZScoreTransform, (n*3.3):(n*0.01):(n*4.1), center=false)
    i = StatsBase.fit(ZScoreTransform, -50:0.1:50, center=false)
    q = StatsBase.fit(ZScoreTransform, -50:0.1:50, center=false)
    r = ZScoreTransform(1, 1, [0.0], [σ.scale[1] / i.scale[1]])
    return (; v, σ, i, q, r)
end


function run_kf!(kf, u, y; tt=length(u))
    # check if y and u have correct lengths

    # preallocate results
    idx = map(y_ -> any(y_ .!== missing), y) |> findall # indexes with (non-missing) observations
    T = length(idx) # number of (non-missing) observations
    ut = Array{eltype(u)}(undef, T)
    yt = Array{eltype(y)}(undef, T)
    xt = Array{particletype(kf)}(undef, T)
    Rt = Array{LLPF.covtype(kf)}(undef, T)
    et = Array{eltype(particletype(kf))}(undef, T)

    yμ = Array{eltype(y)}(undef, T)
    yΣ = Array{eltype(y)}(undef, T)

    llt = zero(eltype(particletype(kf)))

    # U = length(u)
    # x = Array{particletype(kf)}(undef, U)
    # R = Array{LLPF.covtype(kf)}(undef, U)

    trange_1 = filter(<=(tt), eachindex(u))
    trange_2 = filter(>(tt), eachindex(u))

    k = 1 # 

    for i in trange_1
        if !any(y[i] .=== missing) # skip correcting step for missing values

            # x[k] = state(kf) |> copy
            # R[k] = covariance(kf) |> copy

            (; ll, e, S, Sᵪ, K) = correct!(kf, u[i], y[i])

            # from LLPF
            llt += ll
            ut[k] = u[i]
            yt[k] = y[i]
            et[k] = first(e)
            xt[k] = state(kf) |> copy
            Rt[k] = covariance(kf) |> copy

            # ouput
            v = predict_kf(kf, u[i]) # TODO: check performance
            yμ[k] = v.μ
            yΣ[k] = v.Σ

            k += 1
        end

        predict!(kf, u[i])
    end

    for i in trange_2
        if !any(y[i] .=== missing) # skip correcting step for missing values
            v = predict_kf(kf, u[i]) # TODO: check performance
            e = y[i] - v.μ
            ut[k] = u[i]
            yt[k] = y[i]
            et[k] = first(e)
            xt[k] = state(kf) |> copy
            Rt[k] = covariance(kf) |> copy

            yμ[k] = v.μ
            yΣ[k] = v.Σ

            k += 1
        end

        predict!(kf, u[i])
    end

    (; idx, u, y, ut, yt, xt, Rt, et, yμ, yΣ, ll=llt, tt)
end

function loss(θ̂, p)
    (; ϑ, zt, u, y) = p

    θ⁺ = softplus.(θ̂) # transform to force positive values
    θ = ComponentVector(; θ⁺..., ϑ...)
    kf = build_kf(θ, u, zt)

    sol = run_kf!(kf, u, y)
    return sum(abs2, sol.e)
end

function residuals(θ̂, p)
    (; ϑ, zt, u, y) = p

    θ⁺ = softplus.(θ̂) # transform to force positive values
    θ = ComponentVector(; θ⁺..., ϑ...)
    kf = build_kf(θ, u, zt)

    sol = run_kf!(kf, u, y)
    return sol.e
end

function tune_hyperparams(θ0, p)
    u0 = invsoftplus.(θ0)

    adtype = AutoForwardDiff()
    f = OptimizationFunction(loss, adtype)
    prob = OptimizationProblem(f, u0, p)

    alg = LBFGS(linesearch=LineSearches.BackTracking())
    sol = solve(prob,
        alg,
        reltol=1e-4,
        show_trace=true
    )

    θ = softplus.(sol.u)
    return θ
end

function tune_hyperparams_nlls(θ0, p; maxiters=10)
    u0 = invsoftplus.(θ0)

    nlls_prob = NonlinearLeastSquaresProblem(residuals, u0, p)

    sol = solve(nlls_prob, LevenbergMarquardt();
        maxiters, show_trace=Val(true),
        # trace_level=TraceWithJacobianConditionNumber(25)
    )

    θ = softplus.(sol.u)
    return θ
end






function extract_ocv(kf)
    zt = kf.p.zt
    q̂min, q̂max = extrema(kf.p.r0.b0)
    q̂ = q̂min:0.01:q̂max
    q = StatsBase.reconstruct(zt.q, q̂)

    # Q = last(q) - first(q)
    # soc = (q .- first(q)) ./ Q

    soc = range(0.15, 0.9, length=length(q))
    # soc = range(0.05, 0.95, length=length(q))

    ocv = predict_gp(kf, q̂, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)

    focv = LinearInterpolation(ocvμ, soc)
    focv⁻¹ = LinearInterpolation(soc, ocvμ)

    (; ocv=focv, ocv⁻¹=focv⁻¹)
end


function plot_rc_params(kfs)

    fig = Figure()
    ax = Axis(fig[1, 1])

    zt = kf.p.zt

    for (id, kf) in kfs
        x = ComponentVector(kf.x, kf.p.xid)
        r = StatsBase.reconstruct(zt.r, [abs(x.rc.r)]) |> first
        τ = x.rc.τ
        scatter!(ax, τ, r)
    end

    fig
end


function plot_arrhenius(kfs)

    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]


    T = 15:30
    T0 = 25

    for (i, (id, kf)) in enumerate(kfs)
        x = ComponentVector(kf.x, kf.p.xid)
        # r = StatsBase.reconstruct(zt.r, [abs(x.rc.r)]) |> first
        # τ = x.rc.τ
        # scatter!(ax, τ, r)
        k = x.arr.k
        kT = @. exp(k * (1 / T - 1 / T0))
        lines!(ax[1], T, kT)
        scatter!(ax[2], i, k)
    end

    fig
end



# fig7 = let id = (; p=3, m=5)
