"""
Reduced 3-state model for SOC estimation with frozen 2RC ECM parameters.
Built from a fitted RCGP2RCModel: scalar R0, both RC time constants τ1/τ2, and
Arrhenius parameters are fixed; R1 and R2 stay as GP functions of charge with
weights frozen from the parent KF. Only charge (q) and the two RC voltages are
estimated online.
"""
struct RCGP2RCStateModel <: AbstractBatteryModel
    kf::ExtendedKalmanFilter
end

RCGP2RCStateModel(model::RCGP2RCModel; q0, Ts, θ) = RCGP2RCStateModel(_build_rcgp2rc_soc_kf(model.kf; q0, Ts, θ))


# === private dynamics / measurement / R2

function _rcgp2rc_dynamics_soc(x, u, p, t)
    (; kf, Ts, τ1, τ2, T0, k) = p
    (; i, T) = u

    T_K = T + 273.15
    T0_K = T0 + 273.15
    kT = exp(k * (1 / T_K - 1 / T0_K))

    q = x[1]
    vrc1 = x[2]
    vrc2 = x[3]

    kfx = ComponentVector(kf.x, kf.p.xid)
    r1 = measurement_gp(kf.p.r1, kfx.r1, q) * kT
    r2 = measurement_gp(kf.p.r2, kfx.r2, q) * kT

    vrc1⁺ = exp(-Ts / τ1) * vrc1 + i * r1 * (1 - exp(-Ts / τ1))
    vrc2⁺ = exp(-Ts / τ2) * vrc2 + i * r2 * (1 - exp(-Ts / τ2))
    q⁺ = q + i * Ts / 3600
    return SA[q⁺, vrc1⁺, vrc2⁺]
end

function _rcgp2rc_measurement_soc(x, u, p, t)
    (; kf, T0, k, r0) = p
    kfx = ComponentVector(kf.x, kf.p.xid)
    q = x[1]
    vrc1 = x[2]
    vrc2 = x[3]
    (; i, T) = u
    T_K = T + 273.15
    T0_K = T0 + 273.15
    kT = exp(k * (1 / T_K - 1 / T0_K))

    ocv = measurement_gp(kf.p.ocv, kfx.ocv, q)
    return ocv + i * r0 * kT + vrc1 + vrc2 |> SVector{1}
end

function _rcgp2rc_R2_soc(x, u, p, t)
    (; kf) = p
    q = x[1]
    ocv = uncertainty_gp(kf.p.ocv, q)
    return ocv + kf.p.vσ² |> SMatrix{1, 1}
end


# === builder

function _build_rcgp2rc_soc_kf(kf1; q0, Ts, θ)
    xid = kf1.p.xid
    T0 = kf1.p.arr.T0
    vσ² = kf1.p.vσ²
    xc = ComponentVector(kf1.x, xid)

    r0 = abs(xc.r0.r)
    τ1 = abs(xc.rc1.τ)
    τ2 = abs(xc.rc2.τ)
    k = abs(xc.arr.k)

    p = (; kf = kf1, Ts, r0, τ1, τ2, k, T0, vσ²)

    x0 = SA[q0, 0.0, 0.0]
    Σ = @SMatrix [θ.q.σ0^2 0 0; 0 θ.rc.σ0^2 0; 0 0 θ.rc.σ0^2]
    d0 = LLPF.SimpleMvNormal(x0, Σ)
    R1 = @SMatrix [θ.q.σ1^2 0 0; 0 θ.rc.σ1^2 0; 0 0 θ.rc.σ1^2]
    return ExtendedKalmanFilter(_rcgp2rc_dynamics_soc, _rcgp2rc_measurement_soc, R1, _rcgp2rc_R2_soc, d0; nx = 3, nu = 1, ny = 1, p)
end
