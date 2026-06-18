"""
OCV as RGP, R0 as a scalar constant (random walk),
with Arrhenius temperature dependence and RC circuit dynamics.
"""
struct FeneconModel <: AbstractBatteryModel
    kf::ExtendedKalmanFilter
end

FeneconModel(θ, u, zt; n = 21) = FeneconModel(_build_fenecon_kf(θ, u, zt; n))


# === private dynamics / measurement / R2

function _cr0_dynamics!(x⁺, x⁻, u, p, t)
    (; xid, Ts) = p
    (; i, T) = u
    xc⁻ = ComponentVector(x⁻, xid)
    xc⁺ = ComponentVector(x⁺, xid)
    xc⁺ .= xc⁻ # forward previous values

    kT = arrhenius_factor(xc⁻.arr, T, p.arr)
    xc⁺.rc.v = dynamics_rc(xc⁻.rc, i, Ts; kT)
    xc⁺.cc.q = dynamics_cc(xc⁻.cc, i, Ts)
    return nothing # IPD
end

function _cr0_measurement(x, u, p, t)
    (; xid) = p
    (; i, T) = u
    xc = ComponentVector(x, xid)
    (; q) = xc.cc

    kT = arrhenius_factor(xc.arr, T, p.arr)

    ocv = measurement_gp(p.ocv, xc.ocv, q)
    r0 = abs(xc.r0.r) * kT    # scalar R0, no GP
    vrc = xc.rc.v
    return ocv + i * r0 + vrc |> SVector{1}
end

function _cr0_R2(x, u, p, t)
    (; vσ², xid) = p
    xc = ComponentVector(x, xid)
    (; q) = xc.cc
    ocv = uncertainty_gp(p.ocv, q)
    return ocv + vσ² |> SMatrix{1, 1}   # no i^2*r0 uncertainty term
end


# === builder

function _build_fenecon_kf(θ, u, zt; n = 21)
    # basis vectors (OCV only)
    qmin, qmax = extrema([x.q for x in u])
    Δq = qmax - qmin
    b0 = range(qmin + 0.05Δq, qmax + 0.05Δq, n) |> collect

    # OCV GP
    kernel1 = θ.ocv.σ * with_lengthscale(SEKernel(), θ.ocv.ℓ)
    rgp1 = RGP(kernel1, b0)

    # R0 scalar component
    r0 = R0(;
        r0 = StatsBase.transform(zt.r, [θ.r0μ]) |> first,
        σ0 = StatsBase.transform(zt.r, [θ.r0.σ0]) |> first,
        σ1 = StatsBase.transform(zt.r, [θ.r0.σ1]) |> first,
    )

    # RC
    rc = RC(;
        v0 = StatsBase.transform(zt.σ, [θ.rc.v0]) |> first,
        σ0_v = StatsBase.transform(zt.σ, [θ.rc.σ0_v]) |> first,
        σ1_v = StatsBase.transform(zt.σ, [θ.rc.σ1_v]) |> first,
        r0 = StatsBase.transform(zt.r, [θ.rc.r0]) |> first,
        σ0_r = StatsBase.transform(zt.r, [θ.rc.σ0_r]) |> first,
        σ1_r = StatsBase.transform(zt.r, [θ.rc.σ1_r]) |> first,
        τ0 = θ.rc.τ0,
        σ0_τ = θ.rc.σ0_τ,
        σ1_τ = θ.rc.σ1_τ,
    )

    # Arrhenius
    arr = Arrhenius(; θ.arr...)

    # coulomb counting
    cc = ColoumbCounting(; θ.cc...)

    # measurement noise
    vσ² = StatsBase.transform(zt.σ, [θ.vσ]) |> first |> abs2

    p = (; arr = arr.p, Ts = θ.Ts, vσ², zt)
    components = (; ocv = rgp1, r0, rc, arr, cc)

    return ExtendedKalmanFilter(components, _cr0_dynamics!, _cr0_measurement, _cr0_R2; p)
end


"""
    reinit_kf!(model::FeneconModel; x=model.kf.x, R=model.kf.R)

Reinitialize the KF for a second pass on the same data, warm-starting the
posterior from a previous run.

Keeps: GP OCV state and covariance, scalar R0 state, RC parameters (r, τ),
Arrhenius state.
Resets: CC charge to q=0, RC voltage to 0, CC cross-correlations to 0.
"""
function reinit_kf!(model::FeneconModel; x = model.kf.x, R = model.kf.R)
    kf = model.kf
    (; xid, Σid) = kf.p

    x_new = ComponentVector(copy(x), xid)
    x_new.cc.q = 0.0
    x_new.rc.v = 0.0
    kf.x .= x_new

    Σ_new = ComponentMatrix(copy(R), Σid)
    Σ_new[:cc, :] .= 0
    Σ_new[:, :cc] .= 0
    Σ_new[:cc, :cc] .= 0.0
    kf.R .= Σ_new

    return model
end


# === reduce_sol

function reduce_sol(model::FeneconModel, sol)
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
    r0_μ = Vector{Float64}(undef, T)
    r0_σ = Vector{Float64}(undef, T)
    arr_kμ = Vector{Float64}(undef, T)
    arr_kσ = Vector{Float64}(undef, T)

    for i in 1:T
        x = ComponentVector(xt[i], xid)
        Σ = ComponentMatrix(Rt[i], Σid)

        qμ[i] = x.cc.q
        rc_vμ[i] = x.rc.v
        rc_rμ[i] = x.rc.r
        rc_τμ[i] = x.rc.τ
        r0_μ[i] = x.r0.r
        arr_kμ[i] = x.arr.k

        qσ[i] = Σ[:cc, :cc][:q, :q]
        rc_vσ[i] = Σ[:rc, :rc][:v, :v]
        rc_rσ[i] = Σ[:rc, :rc][:r, :r]
        rc_τσ[i] = Σ[:rc, :rc][:τ, :τ]
        r0_σ[i] = Σ[:r0, :r0][:r, :r]
        arr_kσ[i] = Σ[:arr, :arr][:k, :k]
    end

    x_end = xt[end]
    R_end = Rt[end]

    return (;
        sol.idx, sol.u, sol.y, sol.ut, sol.yt, sol.et, sol.yμ, sol.yΣ, sol.ll, sol.tt,
        qμ, qσ, rc_vμ, rc_rμ, rc_τμ, rc_vσ, rc_rσ, rc_τσ, r0_μ, r0_σ, arr_kμ, arr_kσ,
        x_end, R_end,
    )
end
