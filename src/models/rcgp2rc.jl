"""
OCV as RGP, R0 as a scalar constant (random walk), R1 and R2 (RC resistances)
as RGPs, with Arrhenius temperature dependence and two parallel RC branches.
"""
struct RCGP2RCModel <: AbstractBatteryModel
    kf::ExtendedKalmanFilter
end

RCGP2RCModel(θ, u, zt; n = 21, pad = 0.05) = RCGP2RCModel(_build_rcgp2rc_kf(θ, u, zt; n, pad))


# === private dynamics / measurement / R2

function _rcgp2rc_dynamics!(x⁺, x⁻, u, p, t)
    (; xid, Ts) = p
    (; i, T) = u
    xc⁻ = ComponentVector(x⁻, xid)
    xc⁺ = ComponentVector(x⁺, xid)
    xc⁺ .= xc⁻ # forward previous values

    kT = arrhenius_factor(xc⁻.arr, T, p.arr)
    q = xc⁻.cc.q
    r1 = measurement_gp(p.r1, xc⁻.r1, q)
    r2 = measurement_gp(p.r2, xc⁻.r2, q)
    xc⁺.rc1.v = dynamics_rc_vτ(xc⁻.rc1, r1, i, Ts; kT)
    xc⁺.rc2.v = dynamics_rc_vτ(xc⁻.rc2, r2, i, Ts; kT)
    xc⁺.cc.q = dynamics_cc(xc⁻.cc, i, Ts)
    return nothing # IPD
end

function _rcgp2rc_measurement(x, u, p, t)
    (; xid) = p
    (; i, T) = u
    xc = ComponentVector(x, xid)
    (; q) = xc.cc

    kT = arrhenius_factor(xc.arr, T, p.arr)

    ocv = measurement_gp(p.ocv, xc.ocv, q)
    r0 = abs(xc.r0.r) * kT
    vrc = xc.rc1.v + xc.rc2.v
    return ocv + i * r0 + vrc |> SVector{1}
end

function _rcgp2rc_R2(x, u, p, t)
    (; vσ², xid) = p
    xc = ComponentVector(x, xid)
    (; q) = xc.cc
    ocv = uncertainty_gp(p.ocv, q)
    return ocv + vσ² |> SMatrix{1, 1}
end


# === builder

function _build_rcgp2rc_kf(θ, u, zt; n = 21, pad = 0.05)
    qmin, qmax = extrema([x.q for x in u])
    Δq = qmax - qmin
    b0 = range(qmin - pad * Δq, qmax + pad * Δq, n) |> collect

    # OCV GP
    kernel_ocv = θ.ocv.σ * with_lengthscale(SEKernel(), θ.ocv.ℓ)
    rgp_ocv = RGP(kernel_ocv, b0)

    # R1 GP
    r1μ̂ = StatsBase.transform(zt.r, [θ.r1μ]) |> first
    kernel_r1 = θ.r1.σ * with_lengthscale(SEKernel(), θ.r1.ℓ)
    rgp_r1 = RGP(r1μ̂, kernel_r1, b0)

    # R2 GP
    r2μ̂ = StatsBase.transform(zt.r, [θ.r2μ]) |> first
    kernel_r2 = θ.r2.σ * with_lengthscale(SEKernel(), θ.r2.ℓ)
    rgp_r2 = RGP(r2μ̂, kernel_r2, b0)

    # R0 scalar component
    r0 = R0(;
        r0 = StatsBase.transform(zt.r, [θ.r0μ]) |> first,
        σ0 = StatsBase.transform(zt.r, [θ.r0.σ0]) |> first,
        σ1 = StatsBase.transform(zt.r, [θ.r0.σ1]) |> first,
    )

    # RC branches (no r state)
    rc1 = _build_rc_vτ(θ.rc1, zt)
    rc2 = _build_rc_vτ(θ.rc2, zt)

    # Arrhenius
    arr = Arrhenius(; θ.arr...)

    # coulomb counting
    cc = ColoumbCounting(; θ.cc...)

    # measurement noise
    vσ² = StatsBase.transform(zt.σ, [θ.vσ]) |> first |> abs2

    p = (; arr = arr.p, Ts = θ.Ts, vσ², zt)
    components = (; ocv = rgp_ocv, r1 = rgp_r1, r2 = rgp_r2, r0, rc1, rc2, arr, cc)

    return ExtendedKalmanFilter(components, _rcgp2rc_dynamics!, _rcgp2rc_measurement, _rcgp2rc_R2; p)
end

function _build_rc_vτ(θrc, zt)
    return RC_VTau(;
        v0 = StatsBase.transform(zt.σ, [θrc.v0]) |> first,
        σ0_v = StatsBase.transform(zt.σ, [θrc.σ0_v]) |> first,
        σ1_v = StatsBase.transform(zt.σ, [θrc.σ1_v]) |> first,
        τ0 = θrc.τ0,
        σ0_τ = θrc.σ0_τ,
        σ1_τ = θrc.σ1_τ,
    )
end


"""
    reinit_kf!(model::RCGP2RCModel; x=model.kf.x, R=model.kf.R)

Reinit the 2-RC KF for a second pass: reset charge and both RC voltages to 0,
clear CC cross-correlations, keep GP/RC/Arrhenius/scalar-R0 posteriors.
"""
function reinit_kf!(model::RCGP2RCModel; x = model.kf.x, R = model.kf.R)
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

function reduce_sol(model::RCGP2RCModel, sol)
    kf = model.kf
    (; xid, Σid) = kf.p
    (; xt, Rt) = sol

    T = length(xt)
    qμ = Vector{Float64}(undef, T)
    qσ = Vector{Float64}(undef, T)
    rc1_vμ = Vector{Float64}(undef, T)
    rc1_τμ = Vector{Float64}(undef, T)
    rc1_vσ = Vector{Float64}(undef, T)
    rc1_τσ = Vector{Float64}(undef, T)
    rc2_vμ = Vector{Float64}(undef, T)
    rc2_τμ = Vector{Float64}(undef, T)
    rc2_vσ = Vector{Float64}(undef, T)
    rc2_τσ = Vector{Float64}(undef, T)
    r0_μ = Vector{Float64}(undef, T)
    r0_σ = Vector{Float64}(undef, T)
    arr_kμ = Vector{Float64}(undef, T)
    arr_kσ = Vector{Float64}(undef, T)

    for i in 1:T
        x = ComponentVector(xt[i], xid)
        Σ = ComponentMatrix(Rt[i], Σid)

        qμ[i] = x.cc.q
        rc1_vμ[i] = x.rc1.v
        rc1_τμ[i] = x.rc1.τ
        rc2_vμ[i] = x.rc2.v
        rc2_τμ[i] = x.rc2.τ
        r0_μ[i] = x.r0.r
        arr_kμ[i] = x.arr.k

        qσ[i] = Σ[:cc, :cc][:q, :q]
        rc1_vσ[i] = Σ[:rc1, :rc1][:v, :v]
        rc1_τσ[i] = Σ[:rc1, :rc1][:τ, :τ]
        rc2_vσ[i] = Σ[:rc2, :rc2][:v, :v]
        rc2_τσ[i] = Σ[:rc2, :rc2][:τ, :τ]
        r0_σ[i] = Σ[:r0, :r0][:r, :r]
        arr_kσ[i] = Σ[:arr, :arr][:k, :k]
    end

    x_end = xt[end]
    R_end = Rt[end]

    return (;
        sol.idx, sol.u, sol.y, sol.ut, sol.yt, sol.et, sol.yμ, sol.yΣ, sol.ll, sol.tt,
        qμ, qσ,
        rc1_vμ, rc1_τμ, rc1_vσ, rc1_τσ,
        rc2_vμ, rc2_τμ, rc2_vσ, rc2_τσ,
        r0_μ, r0_σ, arr_kμ, arr_kσ,
        x_end, R_end,
    )
end


# === model-specific plots

function plot_ecm!(ax, model::RCGP2RCModel, sol = nothing)
    kf = model.kf
    zt = kf.p.zt

    if sol === nothing
        q̂min, q̂max = extrema(kf.p.r1.b0)
    else
        q̂min, q̂max = extrema(sol.qμ)
    end
    q̂ = collect(q̂min:0.01:q̂max)
    q = StatsBase.reconstruct(zt.q, q̂)

    # OCV
    ocv = predict_gp(kf, q̂, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
    ocvσ = StatsBase.reconstruct(zt.σ, sqrt.(diag(ocv.Σ)))
    lines!(ax[1], q, ocvμ)
    band!(ax[1], q, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha = 0.8)

    # R0 scalar: horizontal band from final state estimate
    x_last = sol === nothing ? state(kf) : sol.x_end
    xc = ComponentVector(x_last, kf.p.xid)
    R_last = sol === nothing ? covariance(kf) : sol.R_end
    Σ = ComponentMatrix(R_last, kf.p.Σid)
    rμ = StatsBase.reconstruct(zt.r, [abs(xc.r0.r)]) |> first
    rσ = StatsBase.reconstruct(zt.r, [sqrt(Σ[:r0, :r0][:r, :r])]) |> first
    hlines!(ax[2], rμ * 1.0e3; color = Cycled(1))
    hspan!(ax[2], (rμ - 2rσ) * 1.0e3, (rμ + 2rσ) * 1.0e3; color = (Cycled(1), 0.3))

    # R1 + R2 GPs overlaid
    for (sym, idx) in ((:r1, 2), (:r2, 3))
        r = predict_gp(kf, q̂, sym)
        rμ_gp = StatsBase.reconstruct(zt.r, r.μ) * 1.0e3
        rσ_gp = StatsBase.reconstruct(zt.r, sqrt.(diag(r.Σ))) * 1.0e3
        lines!(ax[2], q, rμ_gp; color = Cycled(idx))
        band!(ax[2], q, rμ_gp + 2rσ_gp, rμ_gp - 2rσ_gp; color = (Cycled(idx), 0.3))
    end
    return ax[2]
end
