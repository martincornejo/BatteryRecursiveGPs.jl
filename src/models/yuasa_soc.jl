"""
Reduced 2-state model for SOC estimation with frozen ECM parameters.
Built from a fitted YuasaModel: RC and Arrhenius parameters are fixed,
only charge (q) and RC voltage (vrc) are estimated online.
"""
struct YuasaStateModel <: AbstractBatteryModel
    kf::ExtendedKalmanFilter
end

YuasaStateModel(model::YuasaModel; q0, Ts, θ) = YuasaStateModel(_build_soc_kf(model.kf; q0, Ts, θ))


# === private dynamics / measurement / R2

function _dynamics_soc(x, u, p, t)
    (; Ts, r1, τ1, T0, k) = p
    (; i, T) = u

    T_K = T + 273.15
    T0_K = T0 + 273.15
    kT = exp(k * (1 / T_K - 1 / T0_K))
    r1 = r1 * kT

    q = x[1]
    vrc = x[2]

    vrc⁺ = exp(-Ts / τ1) * vrc + i * r1 * (1 - exp(-Ts / τ1))
    q⁺ = q + i * Ts / (3600)
    return SA[q⁺, vrc⁺]
end

function _measurement_soc(x, u, p, t)
    (; kf, T0, k) = p
    kfx = ComponentVector(kf.x, kf.p.xid)
    q = x[1]
    vrc = x[2]
    (; i, T) = u
    T_K = T + 273.15
    T0_K = T0 + 273.15
    kT = exp(k * (1 / T_K - 1 / T0_K))

    ocv = measurement_gp(kf.p.ocv, kfx.ocv, q)
    r0 = measurement_gp(kf.p.r0, kfx.r0, q) * kT
    return ocv + i * r0 + vrc |> SVector{1}
end

function _R2_soc(x, u, p, t)
    (; kf, k, T0) = p
    (; i, T) = u
    T_K = T + 273.15
    T0_K = T0 + 273.15
    kT = exp(k * (1 / T_K - 1 / T0_K))
    q = x[1]
    ocv = uncertainty_gp(kf.p.ocv, q)
    r0 = uncertainty_gp(kf.p.r0, q) * kT
    return ocv + i^2 * r0 + kf.p.vσ² |> SMatrix{1, 1}
end


# === builder

function _build_soc_kf(kf1; q0, Ts, θ)
    xid = kf1.p.xid
    T0 = kf1.p.arr.T0
    vσ² = kf1.p.vσ²
    xc = ComponentVector(kf1.x, xid)

    r1 = abs(xc.rc.r)
    τ1 = abs(xc.rc.τ)
    k = abs(xc.arr.k)

    p = (; kf = kf1, Ts, r1, τ1, k, T0, vσ²)

    x0 = SA[q0, 0.0]
    Σ = @SMatrix [θ.q.σ0^2 0; 0 θ.rc.σ0^2]
    d0 = LLPF.SimpleMvNormal(x0, Σ)
    R1 = @SMatrix [θ.q.σ1^2 0; 0 θ.rc.σ1^2]
    return ExtendedKalmanFilter(_dynamics_soc, _measurement_soc, R1, _R2_soc, d0; nx = 2, nu = 1, ny = 1, p)
end
