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


# Shared body for the per-model `reinit_kf!` methods. Re-anchoring `cc.q` to 0 makes its
# accumulated correlations stale, so its covariance block goes too; the RC voltages are
# zeroed but keep theirs, since they relax on their own.
function _reinit_kf!(model, rc_branches...; x, R)
    kf = model.kf
    (; xid, Σid) = kf.p

    x_new = ComponentVector(copy(x), xid)
    x_new.cc.q = 0.0
    for branch in rc_branches
        getproperty(x_new, branch).v = 0.0
    end
    kf.x .= x_new

    Σ_new = ComponentMatrix(copy(R), Σid)
    Σ_new[:cc, :] .= 0
    Σ_new[:, :cc] .= 0
    Σ_new[:cc, :cc] .= 0.0
    kf.R .= Σ_new

    return model
end


function reduce_sol(model::AbstractBatteryStateModel, sol)
    (; xt, Rt) = sol
    qμ = [x[1] for x in xt]
    qσ = [R[1, 1] for R in Rt]
    vrc_μ = [x[2] for x in xt]
    vrc_σ = [R[2, 2] for R in Rt]
    x_end = xt[end]
    R_end = Rt[end]
    return (;
        sol.idx, sol.u, sol.y, sol.ut, sol.yt, sol.xt, sol.et, sol.yμ, sol.yΣ, sol.ll, sol.tt,
        qμ, qσ, vrc_μ, vrc_σ,
        x_end, R_end,
    )
end

function run_kf_smoother!(kf, u, y)
    T = length(u)
    x = Vector{particletype(kf)}(undef, T)
    xt = Vector{particletype(kf)}(undef, T)
    R = Vector{LLPF.covtype(kf)}(undef, T)
    Rt = Vector{LLPF.covtype(kf)}(undef, T)

    llt = zero(eltype(particletype(kf)))

    for i in 1:T
        x[i] = state(kf) |> copy
        R[i] = covariance(kf) |> copy

        if !any(y[i] .=== missing)
            (; ll) = correct!(kf, u[i], y[i])
            llt += ll
        end

        xt[i] = state(kf) |> copy
        Rt[i] = covariance(kf) |> copy

        predict!(kf, u[i])
    end

    return (; x, xt, R, Rt, u, y, ll = llt)
end

run_kf_smoother!(model::AbstractBatteryModel, u, y) = run_kf_smoother!(model.kf, u, y)

function smooth_kf!(kf, u, y)
    sol = run_kf_smoother!(kf, u, y)
    return LLPF.smooth(sol, kf, u, y)
end

smooth_kf!(model::AbstractBatteryModel, u, y) = smooth_kf!(model.kf, u, y)
