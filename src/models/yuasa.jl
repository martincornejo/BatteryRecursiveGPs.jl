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


# === model-specific plots

function plot_ecm!(ax, model::YuasaModel, sol = nothing)
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


function plot_ecms_norm(models::AbstractDict, sols, fsoc, focv, fR0 = nothing; vlim = (3.5, 3.95))
    fig = Figure(size = (600, 600))
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / mΩ"
    ax[2].xlabel = "SOC / p.u."
    hidexdecorations!(ax[1], ticks = false, grid = false)

    s = 0.0:0.01:1.0
    lines!(ax[1], s, focv(s); color = :black, linestyle = :dash)
    if fR0 !== nothing
        lines!(ax[2], s, fR0.(s) * 1.0e3; color = :black, linestyle = :dash)
    end

    for (id, model) in models
        kf = model.kf
        zt = kf.p.zt

        soc0 = calc_soc0(model, sols[id], fsoc; v = vlim) |> Measurements.value
        Q = calc_Q(model, sols[id], fsoc; v = vlim) |> Measurements.value

        q̂min, q̂max = extrema(sols[id].qμ)
        q̂ = q̂min:0.01:q̂max
        q = StatsBase.reconstruct(zt.q, q̂)

        soc = soc0 .+ q ./ Q

        # OCV
        ocv = predict_gp(kf, q̂, :ocv)
        ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
        ocvσ = StatsBase.reconstruct(zt.σ, sqrt.(diag(ocv.Σ)))

        lines!(ax[1], soc, ocvμ)
        band!(ax[1], soc, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha = 0.8)

        # R0
        r0 = predict_gp(kf, q̂, :r0)
        rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1.0e3
        rσ = StatsBase.reconstruct(zt.r, sqrt.(diag(r0.Σ))) * 1.0e3

        lines!(ax[2], soc, rμ)
        band!(ax[2], soc, rμ + 2rσ, rμ - 2rσ, alpha = 0.8)
    end

    ylims!(ax[1], 3.4, 4.15)
    ylims!(ax[2], 0.2, 3.0)
    linkxaxes!(ax...)
    return fig
end


function plot_rc_param_trajectory(model::YuasaModel, sol; r1 = nothing, τ1 = nothing)
    kf = model.kf
    (; zt, Ts) = kf.p

    rμ = StatsBase.reconstruct(zt.r, abs.(sol.rc_rμ)) * 1.0e3
    rσ = StatsBase.reconstruct(zt.r, sqrt.(sol.rc_rσ)) * 1.0e3

    τμ = copy(sol.rc_τμ)
    τσ = sqrt.(sol.rc_τσ)

    vμ = StatsBase.reconstruct(zt.σ, sol.rc_vμ) * 1.0e3
    vσ = StatsBase.reconstruct(zt.σ, sqrt.(sol.rc_vσ)) * 1.0e3

    t = (1:length(rμ)) / 3600 * Ts
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:3]
    lines!(ax[1], t, rμ)
    band!(ax[1], t, rμ - 2rσ, rμ + 2rσ; alpha = 0.5)
    lines!(ax[2], t, τμ)
    band!(ax[2], t, τμ - 2τσ, τμ + 2τσ; alpha = 0.5)

    lines!(ax[3], t, vμ)
    band!(ax[3], t, vμ - 2vσ, vμ + 2vσ; alpha = 0.5)

    if r1 !== nothing
        hlines!(ax[1], r1 * 1.0e3; color = :black, linestyle = :dash)
    end
    if τ1 !== nothing
        hlines!(ax[2], τ1; color = :black, linestyle = :dash)
    end

    ax[1].ylabel = "R / mΩ"
    ax[2].ylabel = "τ / s"
    ax[3].ylabel = "RC voltage / mV"
    ax[3].xlabel = "Time / h"

    for _ax in ax
        xlims!(_ax, t[1], t[end])
    end

    linkxaxes!(ax...)
    return fig
end


function plot_arrhenius_param_trajectory(model::YuasaModel, sol; k = nothing)
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


function animate_ecm_evolution(file, model::YuasaModel, sol)
    kf = model.kf
    fig = Figure(size = (600, 600))
    ax = [Makie.Axis(fig[i, 1]) for i in 1:3]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / mΩ"
    ax[2].xlabel = "ΔQ / Ah"
    hidexdecorations!(ax[1], ticks = false, grid = false)

    zt = kf.p.zt

    x = ComponentVector.(sol.xt, kf.p.xid)
    q̂min, q̂max = extrema([_x.cc.q for _x in x])

    q̂ = q̂min:0.01:q̂max
    q = StatsBase.reconstruct(zt.q, q̂)

    (; xt, Rt) = sol

    # OCV
    ocv = predict_gp(kf, q̂, xt[1], Rt[1], :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
    ocvσ = StatsBase.reconstruct(zt.σ, sqrt.(diag(ocv.Σ)))

    l1 = lines!(ax[1], q, ocvμ)
    b1 = band!(ax[1], q, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha = 0.8)

    # R0
    r0 = predict_gp(kf, q̂, xt[1], Rt[1], :r0)
    rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1.0e3
    rσ = StatsBase.reconstruct(zt.r, sqrt.(diag(r0.Σ))) * 1.0e3

    l2 = lines!(ax[2], q, rμ)
    b2 = band!(ax[2], q, rμ + 2rσ, rμ - 2rσ, alpha = 0.8)

    v = StatsBase.reconstruct(zt.v, first.(sol.yt))
    lines!(ax[3], v)
    v1 = vlines!(ax[3], 1; color = :red)

    ylims!(ax[1], 3.4, 4.2)
    ylims!(ax[2], 0.0, 3.0)
    linkxaxes!(ax[1], ax[2])

    return record(fig, file, 1:10:length(xt)) do i
        # OCV
        ocv = predict_gp(kf, q̂, xt[i], Rt[i], :ocv)
        ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
        ocvσ = StatsBase.reconstruct(zt.σ, sqrt.(diag(ocv.Σ)))

        # R0
        r0 = predict_gp(kf, q̂, xt[i], Rt[i], :r0)
        rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1.0e3
        rσ = StatsBase.reconstruct(zt.r, sqrt.(diag(r0.Σ))) * 1.0e3

        Makie.update!(l1, arg2 = ocvμ)
        Makie.update!(b1, arg2 = ocvμ + 2ocvσ, arg3 = ocvμ - 2ocvσ)

        Makie.update!(l2, arg2 = rμ)
        Makie.update!(b2, arg2 = rμ + 2rσ, arg3 = rμ - 2rσ)

        Makie.update!(v1, arg1 = i)
    end
end


function animate_model(file, sol)
    fig = Figure(size = (900, 400))
    gl1 = GridLayout(fig[1, 2])
    ax1 = [Axis(gl1[i, 1]) for i in 1:2]
    rowsize!(gl1, 1, Relative(0.65))

    gl2 = GridLayout(fig[1, 1])
    ax2 = [Axis(gl2[i, 1]) for i in 1:2]

    colors = Makie.wong_colors()

    θ0 = ComponentVector(;
        # tunable (hyper)params
        ocv = (; σ = 0.5, ℓ = 0.7),
        r0 = (; σ = 0.05, ℓ = 0.5),
        vσ = 3.0e-3,
    )
    ϑ = ComponentVector(;
        # non-tunable params
        Ts = 1.0,
        r0μ = 1.0e-3,
        rc = (;
            v0 = 0.0, σ0_v = 1.0e-3, σ1_v = 1.0e-4,
            r0 = 1.0e-3, σ0_r = 0.5e-3, σ1_r = 0.0,
            τ0 = 300.0, σ0_τ = 30.0, σ1_τ = 0.0,
        ),
    )
    θ = ComponentVector(; θ0..., ϑ...)

    Ts = 1.0
    (; u, y) = sol
    tt = 0.0
    zt = fit_zscore()
    model = YuasaModel(θ, u, zt)
    sol = run_kf!(model, u, y; tt)
    kf = model.kf

    t = sol.idx * Ts

    ## output prediction
    vμ = StatsBase.reconstruct(zt.v, first.(sol.yμ))
    vσ = StatsBase.reconstruct(zt.σ, sqrt.(first.(sol.yΣ)))
    ve = StatsBase.reconstruct(zt.σ, sol.et)
    v̂ = StatsBase.reconstruct(zt.v, first.(sol.yt))

    lines!(ax1[1], t / 3600, v̂, color = :gray, label = "Measurement")
    l1 = lines!(ax1[1], t / 3600, vμ; color = colors[6], label = "Model")
    b1 = band!(ax1[1], t / 3600, vμ - 2vσ, vμ + 2vσ; color = (colors[2], 0.5), label = "Model")
    v1 = vlines!(ax1[1], tt; color = :red, linestyle = :dash)
    l2 = lines!(ax1[2], t / 3600, ve * 1.0e3, color = colors[6])
    b2 = band!(ax1[2], t / 3600, (ve - 2vσ) * 1.0e3, (ve + 2vσ) * 1.0e3; color = (colors[2], 0.5))

    xlims!(ax1[1], 0, length(sol.u) / 3600)
    xlims!(ax1[2], 0, length(sol.u) / 3600)
    ylims!(ax1[1], 3.45, 4.2)
    ylims!(ax1[2], -50, 50)

    ax1[1].ylabel = "Terminal voltage (V)"
    ax1[2].ylabel = "Error (mV)"
    ax1[2].xlabel = "Time (h)"

    ax1[1].xminorticks = IntervalsBetween(5)
    ax1[1].xminorticksvisible = true
    ax1[2].xminorticks = IntervalsBetween(5)
    ax1[2].xminorticksvisible = true

    ax1[1].yminorticks = IntervalsBetween(2)
    ax1[1].yminorticksvisible = true
    ax1[2].yminorticks = IntervalsBetween(5)
    ax1[2].yminorticksvisible = true

    axislegend(ax1[1]; merge = true, position = :ct, orientation = :horizontal, framevisible = false)
    hidexdecorations!(ax1[1], ticks = false, minorticks = false, grid = false, minorgrid = false)

    ## ecm
    x = ComponentVector.(sol.xt, kf.p.xid)
    q̂min, q̂max = extrema([_x.cc.q for _x in x])

    q̂ = q̂min:0.01:q̂max
    q = StatsBase.reconstruct(zt.q, q̂)

    # OCV
    ocv = predict_gp(kf, q̂, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
    ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)

    l3 = lines!(ax2[1], q, ocvμ)
    b3 = band!(ax2[1], q, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha = 0.8)

    # R0
    r0 = predict_gp(kf, q̂, :r0)
    rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1.0e3
    rσ = StatsBase.reconstruct(zt.r, r0.σ) * 1.0e3

    l4 = lines!(ax2[2], q, rμ)
    b4 = band!(ax2[2], q, rμ + 2rσ, rμ - 2rσ, alpha = 0.8)

    ylims!(ax2[1], 3.2, 4.2)
    ylims!(ax2[2], 0.0, 3.0)
    ax2[1].yticks = 3.2:0.3:4.2
    ax2[2].yticks = 0:3

    ax2[1].ylabel = "OCV (V)"
    ax2[2].ylabel = "R₀ (mΩ)"
    ax2[2].xlabel = "Charge (Ah)"
    hidexdecorations!(ax2[1], ticks = false, grid = false)

    ax2[1].title = "ECM reconstruction"
    ax1[1].title = "Model prediction"
    colgap!(fig.layout, 30)

    record(fig, file, 1:60:length(sol.y)) do i
        model = YuasaModel(θ, u, zt)
        sol = run_kf!(model, u, y; tt = i)
        kf = model.kf

        vμ = StatsBase.reconstruct(zt.v, first.(sol.yμ))
        vσ = StatsBase.reconstruct(zt.σ, sqrt.(first.(sol.yΣ)))
        ve = StatsBase.reconstruct(zt.σ, sol.et)

        Makie.update!(l1, arg2 = vμ)
        Makie.update!(b1, arg2 = vμ + 2vσ, arg3 = vμ - vσ)

        Makie.update!(l2, arg2 = ve * 1.0e3)
        Makie.update!(b2, arg2 = (ve + 2vσ) * 1.0e3, arg3 = (ve - 2vσ) * 1.0e3)

        Makie.update!(v1, arg1 = i / 3600)

        # OCV
        ocv = predict_gp(kf, q̂, :ocv)
        ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
        ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)

        # R0
        r0 = predict_gp(kf, q̂, :r0)
        rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1.0e3
        rσ = StatsBase.reconstruct(zt.r, r0.σ) * 1.0e3

        Makie.update!(l3, arg2 = ocvμ)
        Makie.update!(b3, arg2 = ocvμ + 2ocvσ, arg3 = ocvμ - 2ocvσ)

        Makie.update!(l4, arg2 = rμ)
        Makie.update!(b4, arg2 = rμ + 2rσ, arg3 = rμ - 2rσ)
    end

    return fig
end
