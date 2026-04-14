run_kf!(model::AbstractBatteryModel, u, y; tt = length(u)) = run_kf!(model.kf, u, y; tt)

function run_kf!(kf, u, y; tt = length(u))
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

    trange_1 = filter(<=(tt), eachindex(u))
    trange_2 = filter(>(tt), eachindex(u))

    k = 1

    for i in trange_1
        if !any(y[i] .=== missing) # skip correcting step for missing values

            (; ll, e, S, Sᵪ, K) = correct!(kf, u[i], y[i])

            # from LLPF
            llt += ll
            ut[k] = u[i]
            yt[k] = y[i]
            et[k] = first(e)
            xt[k] = state(kf) |> copy
            Rt[k] = covariance(kf) |> copy

            # output
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

    return (; idx, u, y, ut, yt, xt, Rt, et, yμ, yΣ, ll = llt, tt)
end

function reduce_sol(model::AbstractBatteryModel, sol)
    kf = model.kf
    (; xid, Σid) = kf.p
    (; xt, Rt) = sol

    T = length(xt)
    qμ = Vector{Float64}(undef, T)
    qσ = Vector{Float64}(undef, T)
    rc_vμ = Vector{Float64}(undef, T)
    rc_rμ = Vector{Float64}(undef, T)
    rc_τμ = Vector{Float64}(undef, T)
    rc_vσ = Vector{Float64}(undef, T)
    rc_rσ = Vector{Float64}(undef, T)
    rc_τσ = Vector{Float64}(undef, T)
    arr_kμ = Vector{Float64}(undef, T)
    arr_kσ = Vector{Float64}(undef, T)

    for i in 1:T
        x = ComponentVector(xt[i], xid)
        Σ = ComponentMatrix(Rt[i], Σid)

        qμ[i] = x.cc.q
        rc_vμ[i] = x.rc.v
        rc_rμ[i] = x.rc.r
        rc_τμ[i] = x.rc.τ
        arr_kμ[i] = x.arr.k

        qσ[i] = Σ[:cc, :cc][:q, :q]
        rc_vσ[i] = Σ[:rc, :rc][:v, :v]
        rc_rσ[i] = Σ[:rc, :rc][:r, :r]
        rc_τσ[i] = Σ[:rc, :rc][:τ, :τ]
        arr_kσ[i] = Σ[:arr, :arr][:k, :k]
    end

    x_end = xt[end]
    R_end = Rt[end]

    return (;
        sol.idx, sol.u, sol.y, sol.ut, sol.yt, sol.et, sol.yμ, sol.yΣ, sol.ll, sol.tt,
        qμ, qσ, rc_vμ, rc_rμ, rc_τμ, rc_vσ, rc_rσ, rc_τσ, arr_kμ, arr_kσ,
        x_end, R_end,
    )
end
