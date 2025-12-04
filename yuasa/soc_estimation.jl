
# TODO: 
# - add RCs
# - add vσ² IN R2 calculation

function RC(; v0, r0, τ0, σ0_v, σ0_r, σ0_τ, σ1_v, σ1_r, σ1_τ)
    μ0 = ComponentVector(
        v=v0,
        r=r0, #
        τ=τ0,
    )
    Σ0 = false .* μ0 * μ0'
    Σ0[:v, :v] = σ0_v^2
    Σ0[:τ, :τ] = σ0_τ^2
    Σ0[:r, :r] = σ0_r^2

    R1 = diagm([σ1_v, σ1_r, σ1_τ]) .^ 2
    # R2 = σ2 .^ 2, p # let's put all R2 together in a single param

    return (; μ0, Σ0, R1) # R2
end

function dynamics_rc(x, i, p)
    (; Ts) = p
    (; v, r, τ) = x
    exp(-Ts / τ) * v + i * r * (1 - exp(-Ts / τ))
end


function dynamics_state(x, u, p, t)
    # (; Ts, R1, τ1, xid) = p
    (; kf, Ts, q) = p
    soc = x[1]
    i = u.î # control

    # dx[1] = soc + i * Ts / (q * 3600)
    # dx[2] = v1 * exp(-Ts / τ1) + i * R1 * (1 - exp(-Ts / τ1))
    soc⁺ = soc + i * Ts / (q * 3600)
    SA[soc⁺]

end

function measurement_state(x, u, p, t)
    (; kf, Δsoc, q) = p
    (; xid, ocv, r0, rc) = kf.p
    soc = x[1]
    soc´ = (soc + Δsoc) * q
    kfx = ComponentVector(kf.x, xid)
    ocv = measurement_gp(ocv, kfx.ocv, soc´)
    r0 = measurement_gp(r0, kfx.r0, soc´)
    vrc = measurement_rc(kfx.rc, u.î, rc.p)
    v = ocv + u.i * r0  + vrc|> SVector{1}ç
    
    # StatsBase.reconstruct(zt.v, v) |> SVector{1}
end

function R2_state(x, u, p, t)
    (; kf) = p
    (; xid, ocv, r0, rc) = kf.p
    soc = x[1]
    # kfx = ComponentVector(kf.x, xid)
    ocv = uncertainty_gp(ocv, soc)
    r0 = uncertainty_gp(r0, soc)
    rc = uncertainty_rc(rc, u.î)
    ocv + u.i^2 * r0 |> SMatrix{1,1}
end

function build_kf_state(kf1, soc0, Δsoc, q, Ts)
    x0 = SA[soc0]
    θ = (; kf=kf1, Ts, q, Δsoc)

    d0 = MvNormal(x0, 1e-5I)
    R1 = @SMatrix [1e-12;;]
    ExtendedKalmanFilter(dynamics_state, measurement_state, R1, R2_state, d0; nx=1, nu=1, ny=1, p=θ)
end

