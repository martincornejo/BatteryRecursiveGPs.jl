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


# === model-specific plots

function plot_ecm!(ax, model::Yuasa2RCModel, sol = nothing)
    kf = model.kf
    zt = kf.p.zt

    if sol === nothing
        q̂min, q̂max = extrema(kf.p.r0.b0)
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

    # R0
    r0 = predict_gp(kf, q̂, :r0)
    rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1.0e3
    rσ = StatsBase.reconstruct(zt.r, sqrt.(diag(r0.Σ))) * 1.0e3
    lines!(ax[2], q, rμ)
    return band!(ax[2], q, rμ + 2rσ, rμ - 2rσ, alpha = 0.8)
end


function plot_rc_param_trajectory(model::Yuasa2RCModel, sol; r1 = nothing, τ1 = nothing, r2 = nothing, τ2 = nothing)
    kf = model.kf
    (; zt, Ts) = kf.p

    t = (1:length(sol.qμ)) / 3600 * Ts
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:3]

    for (tag, rμ_raw, rσ_raw, τμ, τσ_raw, vμ_raw, vσ_raw, rref, τref) in (
            ("rc1", sol.rc1_rμ, sol.rc1_rσ, sol.rc1_τμ, sol.rc1_τσ, sol.rc1_vμ, sol.rc1_vσ, r1, τ1),
            ("rc2", sol.rc2_rμ, sol.rc2_rσ, sol.rc2_τμ, sol.rc2_τσ, sol.rc2_vμ, sol.rc2_vσ, r2, τ2),
        )
        rμ = StatsBase.reconstruct(zt.r, abs.(rμ_raw)) * 1.0e3
        rσ = StatsBase.reconstruct(zt.r, sqrt.(rσ_raw)) * 1.0e3
        τσ = sqrt.(τσ_raw)
        vμ = StatsBase.reconstruct(zt.σ, vμ_raw) * 1.0e3
        vσ = StatsBase.reconstruct(zt.σ, sqrt.(vσ_raw)) * 1.0e3

        lines!(ax[1], t, rμ; label = tag)
        band!(ax[1], t, rμ - 2rσ, rμ + 2rσ; alpha = 0.5)
        lines!(ax[2], t, τμ; label = tag)
        band!(ax[2], t, τμ - 2τσ, τμ + 2τσ; alpha = 0.5)
        lines!(ax[3], t, vμ; label = tag)
        band!(ax[3], t, vμ - 2vσ, vμ + 2vσ; alpha = 0.5)

        if rref !== nothing
            hlines!(ax[1], rref * 1.0e3; color = :black, linestyle = :dash)
        end
        if τref !== nothing
            hlines!(ax[2], τref; color = :black, linestyle = :dash)
        end
    end

    ax[1].ylabel = "R / mΩ"
    ax[2].ylabel = "τ / s"
    ax[3].ylabel = "RC voltage / mV"
    ax[3].xlabel = "Time / h"

    for _ax in ax
        xlims!(_ax, t[1], t[end])
    end
    axislegend(ax[1]; position = :rt)

    linkxaxes!(ax...)
    return fig
end


function plot_arrhenius_param_trajectory(model::Yuasa2RCModel, sol; k = nothing)
    kf = model.kf
    (; Ts) = kf.p

    kμ = abs.(sol.arr_kμ)
    kσ = sqrt.(sol.arr_kσ)

    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]

    # trajectory
    t = (1:length(kμ)) / 3600 * Ts
    lines!(ax[1], t, kμ)
    band!(ax[1], t, kμ - 2kσ, kμ + 2kσ; alpha = 0.5)

    # arrhenius curve
    T = 15:0.1:35
    T_K = T .+ 273.15
    T0_K = 25 + 273.15
    k0μ = abs(sol.arr_kμ[end])
    k0σ = sqrt(sol.arr_kσ[end])
    k0 = k0μ ± k0σ
    kT = @. exp(k0 * (1 / T_K - 1 / T0_K))
    kTμ = kT .|> Measurements.value
    kTσ = kT .|> Measurements.uncertainty
    lines!(ax[2], T, kTμ)
    band!(ax[2], T, kTμ - 2kTσ, kTμ + 2kTσ; alpha = 0.5)

    if k !== nothing
        kT_ = @. exp(k * (1 / T_K - 1 / T0_K))
        hlines!(ax[1], k; color = :black, linestyle = :dash)
        lines!(ax[2], T, kT_; color = :black, linestyle = :dash)
    end

    xlims!(ax[1], t[1], t[end])
    xlims!(ax[2], T[1], T[end])

    ax[1].xlabel = "Time / h"
    ax[1].ylabel = "k"
    ax[2].xlabel = "Temperature / °C"
    ax[2].ylabel = "kT"

    return fig
end
