
function Q(;q0, σ1, σ2)
    (; 
    μ0=SA[q0],
    Σ0=[1e-5],
    R1=[σ1^2],
    R2=[σ2^2]
    )
end

function measurement_q(q⁺,u,p,t)
    q⁺
end

function dynamics_q(q⁻,i,p,t)
    (; Ts) = p
    q⁺ = q⁻ .+ i * Ts / (3600)
    q⁺
end

# === model
function dynamics_2!(x⁺, x⁻, u, p, t)
    (; xid) = p
    xc⁻ = ComponentVector(x⁻, xid)
    xc⁺ = ComponentVector(x⁺, xid)
    xc⁺ .= xc⁻ 

    xc⁺.rc.v = dynamics_rc(xc⁻.rc, u.i, p)
    xc⁺.q = dynamics_q(xc⁻.q, u.i, p, t)
    nothing # IPD
end

function measurement_2(x, u, p, t)
    (; xid) = p
    xc = ComponentVector(x, xid)

    ocv = measurement_gp(p.ocv, xc.ocv, u.q)
    r0 = measurement_gp(p.r0, xc.r0, u.q)
    vrc = xc.rc.v # measurement rc
    ocv + u.i * r0 + vrc |> SVector{1}
end

function R2_2(x, u, p, t)
    (; xid,vσ²) = p
    xc = ComponentVector(x, xid)
    ocv = uncertainty_gp(p.ocv, xc.q)
    r0 = uncertainty_gp(p.r0, xc.q)
    ocv + u.i^2 * r0 + vσ² |> SMatrix{1,1}
end

##
function build_kf_q(θ, ϑ, df, zt; n=21)
    # basis vectors
    dfn = normalize_data(zt, df)
    qmin, qmax = extrema(dfn.q)
    b0 = range(qmin, qmax, n) |> collect

    # OCV GP
    kernel1 = θ.ocv.σ * with_lengthscale(SEKernel(), θ.ocv.ℓ)
    rgp1 = RGP(kernel1, b0)

    # R0 GP
    r0 = StatsBase.transform(zt.r, [ϑ.r0.r0]) |> first
    kernel2 = θ.r0.σ * with_lengthscale(SEKernel(), θ.r0.ℓ)
    rgp2 = RGP(r0, kernel2, b0)

    # RC
    r1 = StatsBase.transform(zt.r, [ϑ.rc.r0]) |> first
    rc = RC(; r0=r1, τ0=ϑ.rc.τ0, v0=ϑ.rc.v0, θ.rc...)

    q =Q(
        q0=θ.q.q0,
        σ1=θ.q.σ1,
        σ2=1e-4,
    )

    # measurement / model noise
    vσ² = StatsBase.transform(zt.σ, [θ.vσ^2]) |> first

    p = (;
        Ts=ϑ.Ts,
        vσ²,
    )
    rgps = (; ocv=rgp1, r0=rgp2, rc=rc,q = q)

    make_ekf(rgps, dynamics_2!, measurement_2, R2; p)
end