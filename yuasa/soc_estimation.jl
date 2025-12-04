
# TODO: 
# - add RCs
# - add vσ² IN R2 calculation

function dynamics_state(x, u, p, t)
    # (; Ts, R1, τ1, xid) = p
    (; Ts, q) = p
    soc = x[1]
    i = u.î # control

    # dx[1] = soc + i * Ts / (q * 3600)
    # dx[2] = v1 * exp(-Ts / τ1) + i * R1 * (1 - exp(-Ts / τ1))
    soc⁺ = soc + i * Ts / (q * 3600)
    SA[soc⁺]
end

function measurement_state(x, u, p, t)
    (; kf, Δsoc, q) = p
    (; xid, ocv, r0) = kf.p
    soc = x[1]
    soc´ = (soc + Δsoc) * q
    kfx = ComponentVector(kf.x, xid)
    ocv = measurement_gp(ocv, kfx.ocv, soc´)
    r0 = measurement_gp(r0, kfx.r0, soc´)
    v = ocv + u.i * r0 |> SVector{1}
    # StatsBase.reconstruct(zt.v, v) |> SVector{1}
end

function R2_state(x, u, p, t)
    (; kf) = p
    (; xid, ocv, r0) = kf.p
    soc = x[1]
    # kfx = ComponentVector(kf.x, xid)
    ocv = uncertainty_gp(ocv, soc)
    r0 = uncertainty_gp(r0, soc)
    ocv + u.i^2 * r0 |> SMatrix{1,1}
end

function build_kf_state(kf1, soc0, Δsoc, q, Ts)
    x0 = SA[soc0]
    θ = (; kf=kf1, Ts, q, Δsoc)

    d0 = MvNormal(x0, 1e-5I)
    R1 = @SMatrix [1e-12;;]
    ExtendedKalmanFilter(dynamics_state, measurement_state, R1, R2_state, d0; nx=1, nu=1, ny=1, p=θ)
end

