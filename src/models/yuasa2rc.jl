"""
OCV and R0 modeled as Recursive Gaussian Processes (RGPs),
with Arrhenius temperature dependence and two parallel RC branches.
"""
struct Yuasa2RCModel <: AbstractBatteryModel
    kf::ExtendedKalmanFilter
end

Yuasa2RCModel(θ, u, zt; n = 21, pad = 0.05) = Yuasa2RCModel(_build_yuasa2rc_kf(θ, u, zt; n, pad))


# === private dynamics / measurement / R2

function _yuasa2rc_dynamics!(x⁺, x⁻, u, p, t)
    (; xid, Ts) = p
    (; i, T) = u
    xc⁻ = ComponentVector(x⁻, xid)
    xc⁺ = ComponentVector(x⁺, xid)
    xc⁺ .= xc⁻ # forward previous values

    kT = arrhenius_factor(xc⁻.arr, T, p.arr)
    xc⁺.rc1.v = dynamics_rc(xc⁻.rc1, i, Ts; kT)
    xc⁺.rc2.v = dynamics_rc(xc⁻.rc2, i, Ts; kT)
    xc⁺.cc.q = dynamics_cc(xc⁻.cc, i, Ts)
    return nothing # IPD
end

function _yuasa2rc_measurement(x, u, p, t)
    (; xid) = p
    (; i, T) = u
    xc = ComponentVector(x, xid)
    (; q) = xc.cc

    kT = arrhenius_factor(xc.arr, T, p.arr)

    ocv = measurement_gp(p.ocv, xc.ocv, q)
    r0 = measurement_gp(p.r0, xc.r0, q) * kT
    vrc = xc.rc1.v + xc.rc2.v
    return ocv + i * r0 + vrc |> SVector{1}
end

function _yuasa2rc_R2(x, u, p, t)
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

function _build_yuasa2rc_kf(θ, u, zt; n = 21, pad = 0.05)
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

    # RC (two branches)
    rc1 = _build_rc(θ.rc1, zt)
    rc2 = _build_rc(θ.rc2, zt)

    # Arrhenius
    arr = Arrhenius(; θ.arr...)

    # coulomb counting
    cc = ColoumbCounting(; θ.cc...)

    # measurement noise
    vσ² = StatsBase.transform(zt.σ, [θ.vσ]) |> first |> abs2

    p = (; arr = arr.p, Ts = θ.Ts, vσ², zt)
    components = (; ocv = rgp1, r0 = rgp2, rc1, rc2, arr, cc)

    return ExtendedKalmanFilter(components, _yuasa2rc_dynamics!, _yuasa2rc_measurement, _yuasa2rc_R2; p)
end

function _build_rc(θrc, zt)
    return RC(;
        v0 = StatsBase.transform(zt.σ, [θrc.v0]) |> first,
        σ0_v = StatsBase.transform(zt.σ, [θrc.σ0_v]) |> first,
        σ1_v = StatsBase.transform(zt.σ, [θrc.σ1_v]) |> first,
        r0 = StatsBase.transform(zt.r, [θrc.r0]) |> first,
        σ0_r = StatsBase.transform(zt.r, [θrc.σ0_r]) |> first,
        σ1_r = StatsBase.transform(zt.r, [θrc.σ1_r]) |> first,
        τ0 = θrc.τ0,
        σ0_τ = θrc.σ0_τ,
        σ1_τ = θrc.σ1_τ,
    )
end


"""
    reinit_kf!(model::Yuasa2RCModel; x=model.kf.x, R=model.kf.R)

Reinit the 2-RC KF for a second pass: reset charge and both RC voltages to 0,
clear CC cross-correlations, keep GP/RC/Arrhenius posteriors.
"""
function reinit_kf!(model::Yuasa2RCModel; x = model.kf.x, R = model.kf.R)
    kf = model.kf
    (; xid, Σid) = kf.p

    x_new = ComponentVector(copy(x), xid)
    x_new.cc.q = 0.0
    x_new.rc1.v = 0.0
    x_new.rc2.v = 0.0
    kf.x .= x_new

    Σ_new = ComponentMatrix(copy(R), Σid)
    Σ_new[:cc, :] .= 0
    Σ_new[:, :cc] .= 0
    Σ_new[:cc, :cc] .= 0.0
    kf.R .= Σ_new

    return model
end


# === reduce_sol

function reduce_sol(model::Yuasa2RCModel, sol)
    kf = model.kf
    (; xid, Σid) = kf.p
    (; xt, Rt) = sol

    T = length(xt)
    qμ = Vector{Float64}(undef, T)
    qσ = Vector{Float64}(undef, T)
    rc1_vμ = Vector{Float64}(undef, T)
    rc1_rμ = Vector{Float64}(undef, T)
    rc1_τμ = Vector{Float64}(undef, T)
    rc1_vσ = Vector{Float64}(undef, T)
    rc1_rσ = Vector{Float64}(undef, T)
    rc1_τσ = Vector{Float64}(undef, T)
    rc2_vμ = Vector{Float64}(undef, T)
    rc2_rμ = Vector{Float64}(undef, T)
    rc2_τμ = Vector{Float64}(undef, T)
    rc2_vσ = Vector{Float64}(undef, T)
    rc2_rσ = Vector{Float64}(undef, T)
    rc2_τσ = Vector{Float64}(undef, T)
    arr_kμ = Vector{Float64}(undef, T)
    arr_kσ = Vector{Float64}(undef, T)

    for i in 1:T
        x = ComponentVector(xt[i], xid)
        Σ = ComponentMatrix(Rt[i], Σid)

        qμ[i] = x.cc.q
        rc1_vμ[i] = x.rc1.v
        rc1_rμ[i] = x.rc1.r
        rc1_τμ[i] = x.rc1.τ
        rc2_vμ[i] = x.rc2.v
        rc2_rμ[i] = x.rc2.r
        rc2_τμ[i] = x.rc2.τ
        arr_kμ[i] = x.arr.k

        qσ[i] = Σ[:cc, :cc][:q, :q]
        rc1_vσ[i] = Σ[:rc1, :rc1][:v, :v]
        rc1_rσ[i] = Σ[:rc1, :rc1][:r, :r]
        rc1_τσ[i] = Σ[:rc1, :rc1][:τ, :τ]
        rc2_vσ[i] = Σ[:rc2, :rc2][:v, :v]
        rc2_rσ[i] = Σ[:rc2, :rc2][:r, :r]
        rc2_τσ[i] = Σ[:rc2, :rc2][:τ, :τ]
        arr_kσ[i] = Σ[:arr, :arr][:k, :k]
    end

    x_end = xt[end]
    R_end = Rt[end]

    return (;
        sol.idx, sol.u, sol.y, sol.ut, sol.yt, sol.et, sol.yμ, sol.yΣ, sol.ll, sol.tt,
        qμ, qσ,
        rc1_vμ, rc1_rμ, rc1_τμ, rc1_vσ, rc1_rσ, rc1_τσ,
        rc2_vμ, rc2_rμ, rc2_τμ, rc2_vσ, rc2_rσ, rc2_τσ,
        arr_kμ, arr_kσ,
        x_end, R_end,
    )
end
