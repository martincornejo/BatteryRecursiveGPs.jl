"""
Reduced 2-state model for SOC estimation with frozen ECM parameters.
Built from a fitted FeneconModel: scalar R0, RC, and Arrhenius parameters
are fixed; only charge (q) and RC voltage (vrc) are estimated online.
"""
struct FeneconStateModel <: AbstractBatteryStateModel
    kf::ExtendedKalmanFilter
end

FeneconStateModel(model::FeneconModel; q0, Ts, θ) = FeneconStateModel(_build_fenecon_soc_kf(model.kf; q0, Ts, θ))


# === private dynamics / measurement / R2

function _fenecon_dynamics_soc(x, u, p, t)
    (; Ts, r1, τ1, T0, k) = p
    (; i, T) = u

    T_K = T + 273.15
    T0_K = T0 + 273.15
    kT = exp(k * (1 / T_K - 1 / T0_K))
    r1 = r1 * kT

    q = x[1]
    vrc = x[2]

    vrc⁺ = exp(-Ts / τ1) * vrc + i * r1 * (1 - exp(-Ts / τ1))
    q⁺ = q + i * Ts / 3600
    return SA[q⁺, vrc⁺]
end

function _fenecon_measurement_soc(x, u, p, t)
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

function _fenecon_R2_soc(x, u, p, t)
    (; kf) = p
    q = x[1]
    ocv = uncertainty_gp(kf.p.ocv, q)
    return ocv + kf.p.vσ² |> SMatrix{1, 1}
end


# === builder

function _build_fenecon_soc_kf(kf1; q0, Ts, θ)
    xid = kf1.p.xid
    T0 = kf1.p.arr.T0
    vσ² = kf1.p.vσ²
    xc = ComponentVector(kf1.x, xid)

    r0 = abs(xc.r0.r)
    r1 = abs(xc.rc.r)
    τ1 = abs(xc.rc.τ)
    k = abs(xc.arr.k)

    p = (; kf = kf1, Ts, r0, r1, τ1, k, T0, vσ²)

    x0 = SA[q0, 0.0]
    Σ = @SMatrix [θ.q.σ0^2 0; 0 θ.rc.σ0^2]
    d0 = LLPF.SimpleMvNormal(x0, Σ)
    R1 = @SMatrix [θ.q.σ1^2 0; 0 θ.rc.σ1^2]
    return ExtendedKalmanFilter(_fenecon_dynamics_soc, _fenecon_measurement_soc, R1, _fenecon_R2_soc, d0; nx = 2, nu = 1, ny = 1, p)
end
