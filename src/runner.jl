run_kf!(model::AbstractBatteryModel, u, y; tt=length(u)) = run_kf!(model.kf, u, y; tt)

function run_kf!(kf, u, y; tt=length(u))
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

    (; idx, u, y, ut, yt, xt, Rt, et, yμ, yΣ, ll=llt, tt)
end
