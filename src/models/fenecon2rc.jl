"""
    Fenecon2RCModel(θ, u, zt; n = 21)

As [`FeneconModel`](@ref), but with two RC branches in series — a fast and a slow one.

| axis        | value               |
|-------------|---------------------|
| GP curves   | OCV                 |
| R0          | scalar random walk  |
| R1, R2      | scalar random walks |
| temperature | Arrhenius           |
| GP domain   | charge (Ah)         |
| RC branches | 2                   |

`n` sets the number of GP basis points.

`θ` is as [`FeneconModel`](@ref) but with `rc1` and `rc2` in place of `rc`, each carrying the
same fields.
"""
struct Fenecon2RCModel <: AbstractBatteryModel
    kf::ExtendedKalmanFilter
end

Fenecon2RCModel(θ, u, zt; n = 21) = Fenecon2RCModel(_build_fenecon2rc_kf(θ, u, zt; n))


# === private dynamics / measurement / R2

function _fenecon2rc_dynamics!(x⁺, x⁻, u, p, t)
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

function _fenecon2rc_measurement(x, u, p, t)
    (; xid) = p
    (; i, T) = u
    xc = ComponentVector(x, xid)
    (; q) = xc.cc

    kT = arrhenius_factor(xc.arr, T, p.arr)

    ocv = measurement_gp(p.ocv, xc.ocv, q)
    r0 = abs(xc.r0.r) * kT    # scalar R0, no GP
    vrc = xc.rc1.v + xc.rc2.v
    return ocv + i * r0 + vrc |> SVector{1}
end

function _fenecon2rc_R2(x, u, p, t)
    (; vσ², xid) = p
    xc = ComponentVector(x, xid)
    (; q) = xc.cc
    ocv = uncertainty_gp(p.ocv, q)
    return ocv + vσ² |> SMatrix{1, 1}   # no i^2*r0 uncertainty term
end


# === builder

function _build_fenecon2rc_kf(θ, u, zt; n = 21)
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

    # RC (two branches)
    rc1 = _build_rc(θ.rc1, zt)
    rc2 = _build_rc(θ.rc2, zt)

    # Arrhenius
    arr = Arrhenius(; θ.arr...)

    # coulomb counting
    cc = CoulombCounting(; θ.cc...)

    # measurement noise
    vσ² = StatsBase.transform(zt.σ, [θ.vσ]) |> first |> abs2

    p = (; arr = arr.p, Ts = θ.Ts, vσ², zt)
    components = (; ocv = rgp1, r0, rc1, rc2, arr, cc)

    return ExtendedKalmanFilter(components, _fenecon2rc_dynamics!, _fenecon2rc_measurement, _fenecon2rc_R2; p)
end


"""
    reinit_kf!(model::Fenecon2RCModel; x = model.kf.x, R = model.kf.R)

Reinitialize for a second pass over the same data, warm-starting from a previous run.

Resets the Coulomb-counted charge and **both** RC voltages to 0, and clears the charge
state's cross-covariances. Keeps the OCV GP posterior, the scalar R0, both branches'
parameters (r, τ) and the Arrhenius state.
"""
reinit_kf!(model::Fenecon2RCModel; x = model.kf.x, R = model.kf.R) = _reinit_kf!(model, :rc1, :rc2; x, R)


# === reduce_sol

function reduce_sol(model::Fenecon2RCModel, sol)
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
