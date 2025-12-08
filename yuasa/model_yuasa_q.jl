
function ColoumbCounting(; q0=0.0, σ0=0.0, σ1)
    μ0 = ComponentVector(q=q0)
    Σ0 = false .* μ0 * μ0'
    Σ0[:q, :q] = σ0

    R1 = [σ1^2;;]

    return (; μ0, Σ0, R1)
end

function dynamics_cc(x, i, p, t)
    (; Ts) = p
    (; q) = x
    q + i * Ts / (3600)
end

# === model
function dynamics_2!(x⁺, x⁻, u, p, t)
    (; xid) = p
    xc⁻ = ComponentVector(x⁻, xid)
    xc⁺ = ComponentVector(x⁺, xid)
    xc⁺ .= xc⁻

    xc⁺.rc.v = dynamics_rc(xc⁻.rc, u.i, p)
    xc⁺.cc.q = dynamics_cc(xc⁻.cc, u.i, p, t)
    nothing # IPD
end

function measurement_2(x, u, p, t)
    (; xid) = p
    xc = ComponentVector(x, xid)
    (; q) = xc.cc

    ocv = measurement_gp(p.ocv, xc.ocv, q)
    r0 = measurement_gp(p.r0, xc.r0, q)
    vrc = xc.rc.v # measurement rc
    ocv + u.i * r0 + vrc |> SVector{1}
end

function R2_2(x, u, p, t)
    (; xid, vσ²) = p
    xc = ComponentVector(x, xid)
    (; q) = xc.cc

    ocv = uncertainty_gp(p.ocv, q)
    r0 = uncertainty_gp(p.r0, q)
    ocv + u.i^2 * r0 + vσ² |> SMatrix{1,1}
end

##
function build_kf_q(θ, ϑ, df, zt; n=21)
    # basis vectors
    dfn = normalize_data(zt, df)
    # qmin, qmax = extrema(dfn.q)
    Ts = ϑ.Ts
    qmin, qmax = extrema(cumsum(dfn.i) * Ts / 3600)
    Δq = qmax - qmin
    b0 = range(qmin + 0.1Δq, qmax + 0.1Δq, n) |> collect

    # OCV GP
    kernel1 = θ.ocv.σ * with_lengthscale(SEKernel(), θ.ocv.ℓ) # + LinearKernel()
    ocv = RGP(kernel1, b0)

    # R0 GP
    kernel2 = θ.r0.σ * with_lengthscale(SEKernel(), θ.r0.ℓ)
    r0μ = StatsBase.transform(zt.r, [ϑ.r0.r0]) |> first
    r0 = RGP(r0μ, kernel2, b0)

    # RC
    r1 = StatsBase.transform(zt.r, [ϑ.rc.r0]) |> first
    rc = RC(; r0=r1, τ0=ϑ.rc.τ0, v0=ϑ.rc.v0, θ.rc...)

    cc = ColoumbCounting(σ1=θ.q.σ1)

    # measurement / model noise
    vσ² = StatsBase.transform(zt.σ, [θ.vσ^2]) |> first

    p = (;
        Ts=ϑ.Ts,
        vσ²,
    )
    rgps = (; ocv, r0, rc, cc)

    make_ekf(rgps, dynamics_2!, measurement_2, R2_2; p)
end

