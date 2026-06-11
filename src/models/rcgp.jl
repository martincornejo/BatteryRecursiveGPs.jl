"""
OCV as RGP, R0 as a scalar constant (random walk), R1 (RC resistance) as RGP,
with Arrhenius temperature dependence and RC circuit dynamics.
"""
struct RCGPModel <: AbstractBatteryModel
    kf::ExtendedKalmanFilter
end

RCGPModel(θ, u, zt; n = 21, pad = 0.05) = RCGPModel(_build_rcgp_kf(θ, u, zt; n, pad))


# === private dynamics / measurement / R2

function _rcgp_dynamics!(x⁺, x⁻, u, p, t)
    (; xid, Ts) = p
    (; i, T) = u
    xc⁻ = ComponentVector(x⁻, xid)
    xc⁺ = ComponentVector(x⁺, xid)
    xc⁺ .= xc⁻ # forward previous values

    kT = arrhenius_factor(xc⁻.arr, T, p.arr)
    q = xc⁻.cc.q
    r1 = measurement_gp(p.r1, xc⁻.r1, q)
    xc⁺.rc.v = dynamics_rc_vτ(xc⁻.rc, r1, i, Ts; kT)
    xc⁺.cc.q = dynamics_cc(xc⁻.cc, i, Ts)
    return nothing # IPD
end

function _rcgp_measurement(x, u, p, t)
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

function _rcgp_R2(x, u, p, t)
    (; vσ², xid) = p
    xc = ComponentVector(x, xid)
    (; q) = xc.cc
    ocv = uncertainty_gp(p.ocv, q)
    return ocv + vσ² |> SMatrix{1, 1}
end


# === builder

function _build_rcgp_kf(θ, u, zt; n = 21, pad = 0.05)
    qmin, qmax = extrema([x.q for x in u])
    Δq = qmax - qmin
    b0 = range(qmin - pad * Δq, qmax + pad * Δq, n) |> collect

    # OCV GP — absolute ℓ (no Δv scaling)
    kernel1 = θ.ocv.σ * with_lengthscale(SEKernel(), θ.ocv.ℓ)
    rgp1 = RGP(kernel1, b0)

    # R1 GP — absolute ℓ (no * Δq)
    r1μ̂ = StatsBase.transform(zt.r, [θ.r1μ]) |> first
    kernel2 = θ.r1.σ * with_lengthscale(SEKernel(), θ.r1.ℓ)
    rgp2 = RGP(r1μ̂, kernel2, b0)

    r0 = R0(;
        r0 = StatsBase.transform(zt.r, [θ.r0μ]) |> first,
        σ0 = StatsBase.transform(zt.r, [θ.r0.σ0]) |> first,
        σ1 = StatsBase.transform(zt.r, [θ.r0.σ1]) |> first,
    )

    rc = RC_VTau(;
        v0 = StatsBase.transform(zt.σ, [θ.rc.v0]) |> first,
        σ0_v = StatsBase.transform(zt.σ, [θ.rc.σ0_v]) |> first,
        σ1_v = StatsBase.transform(zt.σ, [θ.rc.σ1_v]) |> first,
        τ0 = θ.rc.τ0,
        σ0_τ = θ.rc.σ0_τ,
        σ1_τ = θ.rc.σ1_τ,
    )

    arr = Arrhenius(; θ.arr...)
    cc = ColoumbCounting(; θ.cc...)
    vσ² = StatsBase.transform(zt.σ, [θ.vσ]) |> first |> abs2

    p = (; arr = arr.p, Ts = θ.Ts, vσ², zt)
    components = (; ocv = rgp1, r1 = rgp2, r0, rc, arr, cc)

    return ExtendedKalmanFilter(components, _rcgp_dynamics!, _rcgp_measurement, _rcgp_R2; p)
end


"""
    reinit_kf!(model::RCGPModel; x=model.kf.x, R=model.kf.R)

Reinitialize the KF for a second pass on the same data, warm-starting GP and
scalar posteriors from a previous run.

Keeps: GP (ocv, r1) state and covariance, scalar R0 state, RC parameters (τ),
Arrhenius state.
Resets: CC charge to q=0, RC voltage to 0, CC cross-correlations to 0.
"""
function reinit_kf!(model::RCGPModel; x = model.kf.x, R = model.kf.R)
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

function reduce_sol(model::RCGPModel, sol)
    kf = model.kf
    (; xid, Σid) = kf.p
    (; xt, Rt) = sol

    T = length(xt)
    qμ = Vector{Float64}(undef, T)
    qσ = Vector{Float64}(undef, T)
    rc_vμ = Vector{Float64}(undef, T)
    rc_τμ = Vector{Float64}(undef, T)
    rc_vσ = Vector{Float64}(undef, T)
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
        rc_τμ[i] = x.rc.τ
        r0_μ[i] = x.r0.r
        arr_kμ[i] = x.arr.k

        qσ[i] = Σ[:cc, :cc][:q, :q]
        rc_vσ[i] = Σ[:rc, :rc][:v, :v]
        rc_τσ[i] = Σ[:rc, :rc][:τ, :τ]
        r0_σ[i] = Σ[:r0, :r0][:r, :r]
        arr_kσ[i] = Σ[:arr, :arr][:k, :k]
    end

    x_end = xt[end]
    R_end = Rt[end]

    return (;
        sol.idx, sol.u, sol.y, sol.ut, sol.yt, sol.et, sol.yμ, sol.yΣ, sol.ll, sol.tt,
        qμ, qσ, rc_vμ, rc_vσ, rc_τμ, rc_τσ, r0_μ, r0_σ, arr_kμ, arr_kσ,
        x_end, R_end,
    )
end


# === model-specific plots

function plot_ecm!(ax, model::RCGPModel, sol = nothing; color = nothing)
    kw = isnothing(color) ? (;) : (; color)
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
    lines!(ax[1], q, ocvμ; kw...)
    band!(ax[1], q, ocvμ + 2ocvσ, ocvμ - 2ocvσ; alpha = 0.8, kw...)

    # R0 scalar: horizontal band from final state estimate
    x_last = sol === nothing ? state(kf) : sol.x_end
    xc = ComponentVector(x_last, kf.p.xid)
    R_last = sol === nothing ? covariance(kf) : sol.R_end
    Σ = ComponentMatrix(R_last, kf.p.Σid)
    r0μ = StatsBase.reconstruct(zt.r, [abs(xc.r0.r)]) |> first
    r0σ = StatsBase.reconstruct(zt.r, [sqrt(Σ[:r0, :r0][:r, :r])]) |> first
    # hlines!(ax[2], r0μ * 1.0e3; color = Cycled(1))
    # hspan!(ax[2], (r0μ - 2r0σ) * 1.0e3, (r0μ + 2r0σ) * 1.0e3)

    # R1 GP curve overlaid on the R0 panel
    r1 = predict_gp(kf, q̂, :r1)
    r1μ = StatsBase.reconstruct(zt.r, r1.μ) * 1.0e3
    r1σ = StatsBase.reconstruct(zt.r, sqrt.(diag(r1.Σ))) * 1.0e3

    rμ = r0μ .+ r1μ
    rσ = r0σ .+ r1σ

    lines!(ax[2], q, rμ; kw...)
    band!(ax[2], q, rμ + 2rσ, rμ - 2rσ; alpha = 0.8, kw...)

    return nothing
end
