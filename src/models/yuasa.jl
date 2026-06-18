"""
OCV and R0 modeled as Recursive Gaussian Processes (RGPs),
with Arrhenius temperature dependence and RC circuit dynamics.
"""
struct YuasaModel <: AbstractBatteryModel
    kf::ExtendedKalmanFilter
end

YuasaModel(θ, u, zt; n = 21, pad = 0.05) = YuasaModel(_build_yuasa_kf(θ, u, zt; n, pad))


# === private dynamics / measurement / R2

function _yuasa_dynamics!(x⁺, x⁻, u, p, t)
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

function _yuasa_measurement(x, u, p, t)
    (; xid) = p
    (; i, T) = u
    xc = ComponentVector(x, xid)
    (; q) = xc.cc

    kT = arrhenius_factor(xc.arr, T, p.arr)

    ocv = measurement_gp(p.ocv, xc.ocv, q)
    r0 = measurement_gp(p.r0, xc.r0, q) * kT
    vrc = xc.rc.v
    return ocv + i * r0 + vrc |> SVector{1}
end

function _yuasa_R2(x, u, p, t)
    (; vσ², xid) = p
    (; i, T) = u
    xc = ComponentVector(x, xid)
    (; q) = xc.cc
    kT = arrhenius_factor(xc.arr, T, p.arr)
    ocv = uncertainty_gp(p.ocv, q)
    r0 = uncertainty_gp(p.r0, q) * kT
    return ocv + i^2 * r0 + vσ² |> SMatrix{1, 1}
end


# === builder

function _build_yuasa_kf(θ, u, zt; n = 21, pad = 0.05)
    # basis vectors — extend pad*Δq past each observed edge so boundary
    # basis points are inside the observed range, not at its edge.
    qmin, qmax = extrema([x.q for x in u])
    Δq = qmax - qmin
    b0 = range(qmin - pad * Δq, qmax + pad * Δq, n) |> collect

    # OCV GP
    kernel1 = θ.ocv.σ * with_lengthscale(SEKernel(), θ.ocv.ℓ)
    rgp1 = RGP(kernel1, b0)

    # R0 GP
    r0 = StatsBase.transform(zt.r, [θ.r0μ]) |> first
    kernel2 = θ.r0.σ * with_lengthscale(SEKernel(), θ.r0.ℓ)
    rgp2 = RGP(r0, kernel2, b0)

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
    rgps = (; ocv = rgp1, r0 = rgp2, rc, arr, cc)

    return ExtendedKalmanFilter(rgps, _yuasa_dynamics!, _yuasa_measurement, _yuasa_R2; p)
end


"""
    reinit_kf!(model::YuasaModel; x=model.kf.x, R=model.kf.R)

Reinitialize the KF for a second pass on the same data, warm-starting the GP
posterior from a previous run.

Keeps: GP (ocv, r0) state and covariance, RC parameters (r, τ), Arrhenius state.
Resets: CC charge to q=0, RC voltage to 0, CC cross-correlations to 0.
"""
function reinit_kf!(model::YuasaModel; x = model.kf.x, R = model.kf.R)
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

function reduce_sol(model::YuasaModel, sol)
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
