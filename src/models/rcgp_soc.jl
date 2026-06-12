"""
Reduced 2-state model for SOC estimation with frozen ECM parameters.
Built from a fitted RCGPModel: scalar R0, RC time constant τ1, and Arrhenius
parameters are fixed; R1 stays as a GP function of charge (frozen weights from
the parent KF). Only charge (q) and RC voltage (vrc) are estimated online.
"""
struct RCGPStateModel <: AbstractBatteryStateModel
    kf::ExtendedKalmanFilter
end

RCGPStateModel(model::RCGPModel; q0, Ts, θ) = RCGPStateModel(_build_rcgp_soc_kf(model.kf; q0, Ts, θ))


# === private dynamics / measurement / R2

function _rcgp_dynamics_soc(x, u, p, t)
    (; kf, Ts, τ1, T0, k) = p
    (; i, T) = u

    T_K = T + 273.15
    T0_K = T0 + 273.15
    kT = exp(k * (1 / T_K - 1 / T0_K))

    q = x[1]
    vrc = x[2]

    kfx = ComponentVector(kf.x, kf.p.xid)
    r1 = measurement_gp(kf.p.r1, kfx.r1, q) * kT

    vrc⁺ = exp(-Ts / τ1) * vrc + i * r1 * (1 - exp(-Ts / τ1))
    q⁺ = q + i * Ts / 3600
    return SA[q⁺, vrc⁺]
end

function _rcgp_measurement_soc(x, u, p, t)
    (; kf, T0, k, r0) = p
    kfx = ComponentVector(kf.x, kf.p.xid)
    q = x[1]
    vrc = x[2]
    (; i, T) = u
    T_K = T + 273.15
    T0_K = T0 + 273.15
    kT = exp(k * (1 / T_K - 1 / T0_K))

    ocv = measurement_gp(kf.p.ocv, kfx.ocv, q)
    return ocv + i * r0 * kT + vrc |> SVector{1}
end

function _rcgp_R2_soc(x, u, p, t)
    (; kf) = p
    q = x[1]
    ocv = uncertainty_gp(kf.p.ocv, q)
    return ocv + kf.p.vσ² |> SMatrix{1, 1}
end


# === builder

function _build_rcgp_soc_kf(kf1; q0, Ts, θ)
    xid = kf1.p.xid
    T0 = kf1.p.arr.T0
    vσ² = kf1.p.vσ²
    xc = ComponentVector(kf1.x, xid)

    r0 = abs(xc.r0.r)
    τ1 = abs(xc.rc.τ)
    k = abs(xc.arr.k)

    p = (; kf = kf1, Ts, r0, τ1, k, T0, vσ²)

    x0 = SA[q0, 0.0]
    Σ = @SMatrix [θ.q.σ0^2 0; 0 θ.rc.σ0^2]
    d0 = LLPF.SimpleMvNormal(x0, Σ)
    R1 = @SMatrix [θ.q.σ1^2 0; 0 θ.rc.σ1^2]
    return ExtendedKalmanFilter(_rcgp_dynamics_soc, _rcgp_measurement_soc, R1, _rcgp_R2_soc, d0; nx = 2, nu = 1, ny = 1, p)
end
