function ColoumbCounting(; q0=0.0, σ0=0.0, σ1)
    μ0 = ComponentVector(q=q0)
    Σ0 = false .* μ0 * μ0'
    Σ0[:q, :q] = σ0

    R1 = [σ1^2;;]

    return (; μ0, Σ0, R1)
end

function dynamics_cc(x, u, p)
    (; Ts) = p
    (; q) = x.cc
    (; i) = u
    q + i * Ts / (3600)
end

function Arrhenius(; T0, k0, σ0_k, σ1_k)
    μ0 = ComponentVector(;
        k=k0
    )

    Σ0 = false .* μ0 * μ0'
    Σ0[:k, :k] = σ0_k^2

    R1 = diagm([σ1_k]) .^ 2

    p = (; T0)

    return (; μ0, Σ0, R1, p)
end

function arrhenius_factor(x, T, p)
    (; T0) = p
    (; k) = x
    exp(k * (1 / T - 1 / T0))
end


function RC(; v0, r0, τ0, σ0_v, σ0_r, σ0_τ, σ1_v, σ1_r, σ1_τ)
    μ0 = ComponentVector(
        v=v0,
        r=r0,
        τ=τ0,
    )
    Σ0 = false .* μ0 * μ0'
    Σ0[:v, :v] = σ0_v^2
    Σ0[:τ, :τ] = σ0_τ^2
    Σ0[:r, :r] = σ0_r^2

    R1 = diagm([σ1_v, σ1_r, σ1_τ]) .^ 2

    return (; μ0, Σ0, R1) # R2
end


function dynamics_rc(x, u, p)
    (; Ts) = p
    (; i, T) = u
    (; v, r, τ) = x.rc
    kT = arrhenius_factor(x.arr, T, p.arr)

    r = abs(r) * kT # force positive values
    τ = abs(τ)
    exp(-Ts / τ) * v + i * r * (1 - exp(-Ts / τ))
end

# === model
function dynamics!(x⁺, x⁻, u, p, t)
    (; xid) = p
    xc⁻ = ComponentVector(x⁻, xid)
    xc⁺ = ComponentVector(x⁺, xid)
    xc⁺ .= xc⁻ # forward previous values

    xc⁺.rc.v = dynamics_rc(xc⁻, u, p)
    xc⁺.cc.q = dynamics_cc(xc⁻, u, p)
    nothing # IPD
end

function measurement(x, u, p, t)
    (; xid) = p
    (; i, T) = u
    xc = ComponentVector(x, xid)
    (; q) = xc.cc
    # (; q) = u <- plain coulmb counting

    kT = arrhenius_factor(xc.arr, T, p.arr)

    ocv = measurement_gp(p.ocv, xc.ocv, q)
    r0 = measurement_gp(p.r0, xc.r0, q) * kT
    vrc = xc.rc.v # measurement rc
    ocv + i * r0 + vrc |> SVector{1}
end

function R2(x, u, p, t)
    (; vσ², xid) = p
    (; i, T) = u
    xc = ComponentVector(x, xid)
    (; q) = xc.cc
    # (; q) = u <- plain coulmb counting
    kT = arrhenius_factor(xc.arr, T, p.arr)
    ocv = uncertainty_gp(p.ocv, q)
    r0 = uncertainty_gp(p.r0, q) * kT
    ocv + i^2 * r0 + vσ² |> SMatrix{1,1}
end


# === system
function build_kf(θ, u, zt; n=21)
    # basis vectors
    qmin, qmax = extrema([x.q for x in u])
    Δq = qmax - qmin
    b0 = range(qmin + 0.05Δq, qmax + 0.05Δq, n) |> collect
    # b0 = range(qmin, qmax, n) |> collect

    # OCV GP
    kernel1 = θ.ocv.σ * with_lengthscale(SEKernel(), θ.ocv.ℓ)
    rgp1 = RGP(kernel1, b0)

    # R0 GP
    r0 = StatsBase.transform(zt.r, [θ.r0μ]) |> first
    kernel2 = θ.r0.σ * with_lengthscale(SEKernel(), θ.r0.ℓ)
    rgp2 = RGP(r0, kernel2, b0)

    # RC
    rc = RC(;
        # v1    
        v0=StatsBase.transform(zt.σ, [θ.rc.v0]) |> first,
        σ0_v=StatsBase.transform(zt.σ, [θ.rc.σ0_v]) |> first,
        σ1_v=StatsBase.transform(zt.σ, [θ.rc.σ1_v]) |> first,
        # r1
        r0=StatsBase.transform(zt.r, [θ.rc.r0]) |> first,
        σ0_r=StatsBase.transform(zt.r, [θ.rc.σ0_r]) |> first,
        σ1_r=StatsBase.transform(zt.r, [θ.rc.σ1_r]) |> first,
        # τ1
        τ0=θ.rc.τ0,
        σ0_τ=θ.rc.σ0_τ,
        σ1_τ=θ.rc.σ1_τ,
    )

    # Arrhenius
    arr = Arrhenius(; T0=25, k0=10, σ0_k=5, σ1_k=0.0)

    # coloumb counting
    cc = ColoumbCounting(σ1=1e-4)

    # measurement / model noise
    vσ² = StatsBase.transform(zt.σ, [θ.vσ]) |> first |> abs2 # ^2

    # model
    nx = (length(rc.μ0) + length(rgp1.μ0) + (length(rgp2.μ0)))
    p = (;
        arr=arr.p,
        Ts=θ.Ts,
        vσ²,
        zt
    )
    rgps = (; ocv=rgp1, r0=rgp2, rc, arr, cc)

    make_ekf(rgps, dynamics!, measurement, R2; p)
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
