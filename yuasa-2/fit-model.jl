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


function module_dataset(data, ti, p, m)

    df_v = copy(data[:module_voltage])
    select!(df_v, "_time" => "time", "module_voltage_$(p)_$(m)" => "value")
    df_v = interpolate(df_v, ti; Ts=1.0)

    df_i = copy(data[:module_current])
    select!(df_i, "_time" => "time", "module_average_current_$(p)_$(m)" => ByRow(x -> -x) => "value")
    df_i = interpolate(df_i, ti; Ts=1.0)

    Ts = 1.0
    q = cumsum(df_i.value) * Ts / 3600

    zt = fit_zscore(12)
    î = StatsBase.transform(zt.i, df_i.value)
    q̂ = StatsBase.transform(zt.q, q)

    v̂ = StatsBase.transform(zt.v, df_v.value)

    u = [(; i, q) for (i, q) in zip(î, q̂)]
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

            k += 1
        end

        predict!(kf, u[i])
    end

    for i in trange_2
        if !any(y[i] .=== missing) # skip correcting step for missing values
            v = measure_kf(kf, u[i]) # TODO: check performance
            e = y[i] - v.μ
            ut[k] = u[i]
            yt[k] = y[i]
            et[k] = first(e)
            xt[k] = state(kf) |> copy
            Rt[k] = covariance(kf) |> copy

            k += 1
        end

        predict!(kf, u[i])
    end

    (; idx, u, y, ut, yt, xt, Rt, et, ll=llt, tt)
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


function plot_rc_param_trajectory(kf, sol)
    (; xid, Σid, zt) = kf.p
    xs = ComponentVector.(sol.xt, xid)
    Σs = [ComponentMatrix(R, Σid) for R in sol.Rt]

    rμ = StatsBase.reconstruct(zt.r, abs.([x.rc.r for x in xs])) * 1e3
    rσ = StatsBase.reconstruct(zt.r, sqrt.([Σ[:rc, :rc][:r, :r] for Σ in Σs])) * 1e3

    τμ = [x.rc.τ for x in xs]
    τσ = sqrt.([Σ[:rc, :rc][:τ, :τ] for Σ in Σs])

    vμ = StatsBase.reconstruct(zt.σ, [x.rc.v for x in xs]) * 1e3
    vσ = StatsBase.reconstruct(zt.r, sqrt.([Σ[:rc, :rc][:v, :v] for Σ in Σs])) * 1e3


    t = (1:length(rμ)) / 3600 * 10
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:3]
    ax[1].ylabel = "R / mΩ"
    ax[2].ylabel = "τ / s"
    ax[2].xlabel = "Time / h"
    lines!(ax[1], t, rμ)
    band!(ax[1], t, rμ - 2rσ, rμ + 2rσ; alpha=0.5)
    lines!(ax[2], t, τμ)
    band!(ax[2], t, τμ - 2τσ, τμ + 2τσ; alpha=0.5)


    lines!(ax[3], t, vμ)
    band!(ax[3], t, vμ - 2vσ, vμ + 2vσ; alpha=0.5)

    fig
end


function plot_sim(kf, sol; Ts=1.0, plot_Δv=true)
    zt = kf.p.zt
    (; idx, u, xt, Rt, ut, yt) = sol

    μ = zeros(length(idx))
    σ = zeros(length(idx))
    t = (0:length(u)-1) * Ts / 3600 |> collect

    for i in eachindex(idx, ut, xt, Rt)
        v = measure_kf(kf, ut[i], xt[i], Rt[i])
        μ[i] = StatsBase.reconstruct(zt.v, v.μ) |> first
        σ[i] = StatsBase.reconstruct(zt.σ, sqrt.(v.Σ)) |> first
    end


    # plot
    fig = Figure()
    # ax = [Axis(fig[i, 1]) for i in 1:3]
    ax = [Axis(fig[i, 1]) for i in 1:2]
    colors = Makie.wong_colors()

    # terminal voltage 
    v = StatsBase.reconstruct(zt.v, first.(yt))
    lines!(ax[1], t[idx], v)
    lines!(ax[1], t[idx], μ, color=colors[2])
    band!(ax[1], t[idx], μ - 2σ, μ + 2σ, color=(colors[2], 0.5))

    # voltage error
    e = v - μ
    lines!(ax[2], t[idx], e * 1e3, color=colors[2])
    band!(ax[2], t[idx], (e - 2σ) * 1e3, (e + 2σ) * 1e3, color=(colors[2], 0.5))

    # input current
    # i = StatsBase.reconstruct(zt.i, [x.i for x in u])
    # scatterlines!(ax[3], t, i, color=(:red, 0.5))

    # if plot_Δv
    #     Δv = abs.(diff(v))
    #     Δv_idx = idx[findall(>(0.01), Δv).+1]  # +1 because diff() reduces length by 1
    #     vlines!(ax[1], t[Δv_idx], color=(:gray, 0.5), linestyle=:dash)
    #     vlines!(ax[2], t[Δv_idx], color=(:gray, 0.5), linestyle=:dash)
    #     vlines!(ax[3], t[Δv_idx], color=(:gray, 0.5), linestyle=:dash)
    # end

    if sol.tt != length(sol.u)
        vlines!(ax[1], t[sol.tt]; color=:red)
        vlines!(ax[2], t[sol.tt]; color=:red)
    end

    linkxaxes!(ax...)

    fig
end


function plot_ecm(kf)
    fig = Figure(size=(600, 600))
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / mΩ"
    ax[2].xlabel = "ΔQ / Ah"
    hidexdecorations!(ax[1], ticks=false, grid=false)

    zt = kf.p.zt

    q̂min, q̂max = extrema(kf.p.r0.b0)
    q̂ = q̂min:0.01:q̂max
    q = StatsBase.reconstruct(zt.q, q̂)

    # OCV 
    ocv = predict_gp(kf, q̂, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
    ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)

    lines!(ax[1], q, ocvμ)
    band!(ax[1], q, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha=0.8)

    # R0
    r0 = predict_gp(kf, q̂, :r0)
    rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1e3
    rσ = StatsBase.reconstruct(zt.r, r0.σ) * 1e3

    lines!(ax[2], q, rμ)
    band!(ax[2], q, rμ + 2rσ, rμ - 2rσ, alpha=0.8)

    ylims!(ax[1], 3.4, 4.2)
    ylims!(ax[2], 0.0, 3.0)

    linkxaxes!(ax...)
    fig
end





function plot_ecms(kfs)
    fig = Figure(size=(600, 600))
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / mΩ"
    # ax[2].xlabel = "SOC / p.u."
    ax[2].xlabel = "ΔQ / Ah"
    hidexdecorations!(ax[1], ticks=false, grid=false)

    for (id, kf) in kfs

        zt = kf.p.zt

        q̂min, q̂max = extrema(kf.p.r0.b0)
        q̂ = q̂min:0.01:q̂max
        q = StatsBase.reconstruct(zt.q, q̂)

        # OCV 
        ocv = predict_gp(kf, q̂, :ocv)
        ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
        ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)

        lines!(ax[1], q, ocvμ)
        band!(ax[1], q, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha=0.8)

        # R0
        r0 = predict_gp(kf, q̂, :r0)
        rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1e3
        rσ = StatsBase.reconstruct(zt.r, r0.σ) * 1e3

        lines!(ax[2], q, rμ)
        band!(ax[2], q, rμ + 2rσ, rμ - 2rσ, alpha=0.8)

    end

    # xlims!(ax[1], 0, 1)
    # xlims!(ax[2], 0, 1)
    ylims!(ax[1], 3.4, 4.1)
    ylims!(ax[2], 0.0, 3.0)
    linkxaxes!(ax...)
    fig
end

function plot_ecms2(kfs, fsoc, focv; vlim=(3.5, 3.95))
    fig = Figure(size=(600, 600))
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / mΩ"
    ax[2].xlabel = "SOC / p.u."
    hidexdecorations!(ax[1], ticks=false, grid=false)

    s = 0.0:0.01:1.0
    lines!(ax[1], s, focv(s); color=:black, linestyle=:dash)

    for (id, kf) in kfs

        zt = kf.p.zt

        soc0 = calc_soc0(kf, fsoc; v=vlim) |> Measurements.value
        Q = calc_Q(kf, fsoc; v=vlim) |> Measurements.value

        q̂min, q̂max = extrema(kf.p.r0.b0)
        q̂ = q̂min:0.01:q̂max
        q = StatsBase.reconstruct(zt.q, q̂)

        soc = soc0 .+ q ./ Q

        # OCV 
        ocv = predict_gp(kf, q̂, :ocv)
        ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
        ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)

        lines!(ax[1], soc, ocvμ)
        band!(ax[1], soc, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha=0.8)

        # R0
        r0 = predict_gp(kf, q̂, :r0)
        rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1e3
        rσ = StatsBase.reconstruct(zt.r, r0.σ) * 1e3

        lines!(ax[2], soc, rμ)
        band!(ax[2], soc, rμ + 2rσ, rμ - 2rσ, alpha=0.8)

    end

    # xlims!(ax[1], 0, 1)
    # xlims!(ax[2], 0, 1)
    ylims!(ax[1], 3.4, 4.15)
    ylims!(ax[2], 0.2, 3.0)
    linkxaxes!(ax...)
    fig
end


function plot_ecms3(kfs; v1=3.7)
    fig = Figure(size=(600, 600))
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / mΩ"
    # ax[2].xlabel = "SOC / p.u."
    ax[2].xlabel = "ΔQ / Ah"
    hidexdecorations!(ax[1], ticks=false, grid=false)

    for (id, kf) in kfs

        zt = kf.p.zt

        q̂min, q̂max = extrema(kf.p.r0.b0)
        q̂ = q̂min:0.01:q̂max
        q = StatsBase.reconstruct(zt.q, q̂)

        q0 = calc_ΔQ0(kf, v1) |> Measurements.value

        # OCV 
        ocv = predict_gp(kf, q̂, :ocv)
        ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
        ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)

        lines!(ax[1], q .- q0, ocvμ)
        band!(ax[1], q .- q0, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha=0.8)

        # R0
        r0 = predict_gp(kf, q̂, :r0)
        rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1e3
        rσ = StatsBase.reconstruct(zt.r, r0.σ) * 1e3

        lines!(ax[2], q .- q0, rμ)
        band!(ax[2], q .- q0, rμ + 2rσ, rμ - 2rσ, alpha=0.8)

    end

    # xlims!(ax[1], 0, 1)
    # xlims!(ax[2], 0, 1)
    ylims!(ax[1], 3.4, 4.1)
    ylims!(ax[2], 0.0, 3.0)
    linkxaxes!(ax...)
    fig
end

function animate_ecm_evolution(file, kf, sol)
    fig = Figure(size=(600, 600))
    ax = [Makie.Axis(fig[i, 1]) for i in 1:3]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / mΩ"
    ax[2].xlabel = "ΔQ / Ah"
    hidexdecorations!(ax[1], ticks=false, grid=false)

    zt = kf.p.zt

    q̂min, q̂max = extrema(kf.p.r0.b0)
    q̂ = q̂min:0.01:q̂max
    q = StatsBase.reconstruct(zt.q, q̂)

    (; xt, Rt) = sol

    # OCV 
    ocv = predict_gp(kf, q̂, xt[1], Rt[1], :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
    ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)

    l1 = lines!(ax[1], q, ocvμ)
    b1 = band!(ax[1], q, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha=0.8)

    # R0
    r0 = predict_gp(kf, q̂, xt[1], Rt[1], :r0)
    rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1e3
    rσ = StatsBase.reconstruct(zt.r, r0.σ) * 1e3

    l2 = lines!(ax[2], q, rμ)
    b2 = band!(ax[2], q, rμ + 2rσ, rμ - 2rσ, alpha=0.8)


    # 
    v = StatsBase.reconstruct(zt.v, first.(sol.yt))
    # v = StatsBase.reconstruct(zt.i, first.(sol.ut))
    lines!(ax[3], v)
    v1 = vlines!(ax[3], 1; color=:red)

    ylims!(ax[1], 3.4, 4.2)
    ylims!(ax[2], 0.0, 3.0)
    # for i in eachindex(xt, Rt)
    # linkxaxes!(ax...)
    linkxaxes!(ax[1], ax[2])


    record(fig, file, 1:10:length(xt)) do i
        # OCV
        ocv = predict_gp(kf, q̂, xt[i], Rt[i], :ocv)
        ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
        ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)

        # R0
        r0 = predict_gp(kf, q̂, xt[i], Rt[i], :r0)
        rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1e3
        rσ = StatsBase.reconstruct(zt.r, r0.σ) * 1e3

        Makie.update!(l1, arg2=ocvμ)
        Makie.update!(b1, arg2=ocvμ + 2ocvσ, arg3=ocvμ - 2ocvσ)

        Makie.update!(l2, arg2=rμ)
        Makie.update!(b2, arg2=rμ + 2rσ, arg3=rμ - 2rσ)

        Makie.update!(v1, arg1=i)
    end

end


function extract_ocv(kf)
    q̂min, q̂max = extrema(kf.p.r0.b0)
    q̂ = q̂min:0.01:q̂max
    q = StatsBase.reconstruct(zt.q, q̂)

    # Q = last(q) - first(q)
    # soc = (q .- first(q)) ./ Q

    soc = range(0.015, 0.85, length=length(q))

    ocv = predict_gp(kf, q̂, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)

    focv = LinearInterpolation(ocvμ, soc)
    focv⁻¹ = LinearInterpolation(soc, ocvμ)

    (; ocv=focv, ocv⁻¹=focv⁻¹)
end

