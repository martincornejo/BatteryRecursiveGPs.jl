
function dynamics_state(x, u, p, t)
    (; Ts, R1, τ1, q) = p
    soc = x[1]
    vrc = x[2]

    vrc⁺ = vrc * exp(-Ts / τ1) + u.i * R1 * (1 - exp(-Ts / τ1))
    soc⁺ = soc + u.î * Ts / (q * 3600)
    SA[soc⁺, vrc⁺]
end

function measurement_state(x, u, p, t)
    (; kf, Δsoc, q, zt) = p
    kfx = ComponentVector(kf.x, kf.p.xid)
    soc = x[1]
    vrc = x[2]
    q´ = ((soc - Δsoc) * q - zt.q.mean[1]) / zt.q.scale[1] # transform zt.q !!!

    ocv = measurement_gp(kf.p.ocv, kfx.ocv, q´)
    r0 = measurement_gp(kf.p.r0, kfx.r0, q´)
    ocv + u.i * r0 + vrc |> SVector{1}
end

function R2_state(x, u, p, t)
    (; kf, Δsoc, q, zt) = p
    soc = x[1]
    q´ = ((soc - Δsoc) * q - zt.q.mean[1]) / zt.q.scale[1]
    ocv = uncertainty_gp(kf.p.ocv, q´)
    r0 = uncertainty_gp(kf.p.r0, q´)
    ocv + u.i^2 * r0 + kf.p.vσ² |> SMatrix{1,1}
end

function build_kf_state(kf1, soc0, Δsoc, q, Ts, zt)
    xid = kf1.p.xid
    xc = ComponentVector(kf1.x, xid)

    R1 = xc.rc.r
    τ1 = xc.rc.τ

    θ = (; kf=kf1, Ts, R1, τ1, q, Δsoc, vσ²=1e-2, zt)

    x0 = SA[soc0, 0.0]
    d0 = MvNormal(x0, 1e-5I)
    R1 = @SMatrix [1e-4 0; 0 1e-5]
    ExtendedKalmanFilter(dynamics_state, measurement_state, R1, R2_state, d0; nx=2, nu=1, ny=1, p=θ)
end

