"""
Reduced 3-state model for SOC estimation with frozen 2RC ECM parameters.
Built from a fitted Fenecon2RCModel: scalar R0, both RC branches, and Arrhenius
parameters are fixed; only charge (q) and the two RC voltages are estimated online.
"""
struct Fenecon2RCStateModel <: AbstractBatteryStateModel
    kf::ExtendedKalmanFilter
end

Fenecon2RCStateModel(model::Fenecon2RCModel; q0, Ts, θ) = Fenecon2RCStateModel(_build_fenecon2rc_soc_kf(model.kf; q0, Ts, θ))


# === private dynamics / measurement / R2

function _fenecon2rc_dynamics_soc(x, u, p, t)
    (; Ts, r1, τ1, r2, τ2, T0, k) = p
    (; i, T) = u

    T_K = T + 273.15
    T0_K = T0 + 273.15
    kT = exp(k * (1 / T_K - 1 / T0_K))
    r1_T = r1 * kT
    r2_T = r2 * kT

    q = x[1]
    vrc1 = x[2]
    vrc2 = x[3]

    vrc1⁺ = exp(-Ts / τ1) * vrc1 + i * r1_T * (1 - exp(-Ts / τ1))
    vrc2⁺ = exp(-Ts / τ2) * vrc2 + i * r2_T * (1 - exp(-Ts / τ2))
    q⁺ = q + i * Ts / 3600
    return SA[q⁺, vrc1⁺, vrc2⁺]
end

function _fenecon2rc_measurement_soc(x, u, p, t)
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

function _fenecon2rc_R2_soc(x, u, p, t)
    (; kf) = p
    q = x[1]
    ocv = uncertainty_gp(kf.p.ocv, q)
    return ocv + kf.p.vσ² |> SMatrix{1, 1}
end


# === builder

function _build_fenecon2rc_soc_kf(kf1; q0, Ts, θ)
    xid = kf1.p.xid
    T0 = kf1.p.arr.T0
    vσ² = kf1.p.vσ²
    xc = ComponentVector(kf1.x, xid)

    r0 = abs(xc.r0.r)
    r1 = abs(xc.rc1.r)
    τ1 = abs(xc.rc1.τ)
    r2 = abs(xc.rc2.r)
    τ2 = abs(xc.rc2.τ)
    k = abs(xc.arr.k)

    p = (; kf = kf1, Ts, r0, r1, τ1, r2, τ2, k, T0, vσ²)

    x0 = SA[q0, 0.0, 0.0]
    Σ = @SMatrix [θ.q.σ0^2 0 0; 0 θ.rc.σ0^2 0; 0 0 θ.rc.σ0^2]
    d0 = LLPF.SimpleMvNormal(x0, Σ)
    R1 = @SMatrix [θ.q.σ1^2 0 0; 0 θ.rc.σ1^2 0; 0 0 θ.rc.σ1^2]
    return ExtendedKalmanFilter(_fenecon2rc_dynamics_soc, _fenecon2rc_measurement_soc, R1, _fenecon2rc_R2_soc, d0; nx = 3, nu = 1, ny = 1, p)
end
