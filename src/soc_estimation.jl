
function dynamics_state(x, u, p, t)
    (; Ts, r1, τ1, T0, k) = p
    (; i, T) = u

    kT = exp(k * (1 / T - 1 / T0))
    r1 = r1 * kT

    q = x[1]
    vrc = x[2]

    vrc⁺ = exp(-Ts / τ1) * vrc + i * r1 * (1 - exp(-Ts / τ1))
    q⁺ = q + i * Ts / (3600)
    SA[q⁺, vrc⁺]
end

function measurement_state(x, u, p, t)
    (; kf, T0, k) = p
    kfx = ComponentVector(kf.x, kf.p.xid)
    q = x[1]
    vrc = x[2]
    (; i, T) = u
    # q´ = ((soc - Δsoc) * q - zt.q.mean[1]) / zt.q.scale[1] # transform zt.q !!!
    kT = exp(k * (1 / T - 1 / T0))

    ocv = measurement_gp(kf.p.ocv, kfx.ocv, q)
    r0 = measurement_gp(kf.p.r0, kfx.r0, q) * kT
    ocv + u.i * r0 + vrc |> SVector{1}
end

function R2_state(x, u, p, t)
    (; kf, k, T0) = p
    (; i, T) = u
    kT = exp(k * (1 / T - 1 / T0))
    q = x[1]
    # q´ = ((soc - Δsoc) * q - zt.q.mean[1]) / zt.q.scale[1]
    ocv = uncertainty_gp(kf.p.ocv, q)
    r0 = uncertainty_gp(kf.p.r0, q) * kT
    ocv + i^2 * r0 + kf.p.vσ² |> SMatrix{1,1}
end

function build_kf_state(kf1; q0, Ts, θ)
    xid = kf1.p.xid
    T0 = kf1.p.arr.T0
    vσ² = kf1.p.vσ²
    xc = ComponentVector(kf1.x, xid)

    r1 = abs(xc.rc.r)
    τ1 = abs(xc.rc.τ)
    k = abs(xc.arr.k)

    p = (; kf=kf1, Ts, r1, τ1, k, T0, vσ²)

    x0 = SA[q0, 0.0]
    Σ = @SMatrix [θ.q.σ0^2 0; 0 θ.rc.σ0^2]
    d0 = LLPF.SimpleMvNormal(x0, Σ)
    R1 = @SMatrix [θ.q.σ1^2 0; 0 θ.rc.σ1^2]
    ExtendedKalmanFilter(dynamics_state, measurement_state, R1, R2_state, d0; nx=2, nu=1, ny=1, p)
end

