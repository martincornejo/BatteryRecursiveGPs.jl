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
    (; xid) = p
    xc⁻ = ComponentVector(x⁻, xid)
    xc⁺ = ComponentVector(x⁺, xid)
    xc⁺ .= xc⁻ # forward previous values

    xc⁺.rc.v = dynamics_rc(xc⁻, u, p)
    xc⁺.cc.q = dynamics_cc(xc⁻, u, p)
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


# === model-specific plots

function plot_ecm!(ax, model::FeneconModel, sol = nothing)
    kf = model.kf
    zt = kf.p.zt

    if sol === nothing
        q̂min, q̂max = extrema(kf.p.ocv.b0)
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
    return hspan!(ax[2], (rμ - 2rσ) * 1.0e3, (rμ + 2rσ) * 1.0e3; color = (Cycled(1), 0.3))
end
