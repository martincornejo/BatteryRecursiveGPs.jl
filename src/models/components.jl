"""
    CoulombCounting(; q0 = 0.0, σ0 = 0.0, σ1) -> (; μ0, Σ0, R1)

Charge state `q`, advanced by [`dynamics_cc`](@ref). `q0` is the initial charge, `σ0` its
initial standard deviation and `σ1` the per-step process-noise standard deviation; `σ1 = 0`
takes the integrated current as exact.
"""
function CoulombCounting(; q0 = 0.0, σ0 = 0.0, σ1)
    μ0 = ComponentVector(q = q0)
    Σ0 = false .* μ0 * μ0'
    Σ0[:q, :q] = σ0^2

    R1 = [σ1^2;;]

    return (; μ0, Σ0, R1)
end

"""
    dynamics_cc(cc, i, Ts) -> q

Integrate current `i` over one step of `Ts` seconds onto the charge state `cc.q`. Current in
A and `Ts` in s give charge in Ah.
"""
function dynamics_cc(cc, i, Ts)
    (; q) = cc
    return q + i * Ts / (3600)
end

"""
    Arrhenius(; T0, k0, σ0_k, σ1_k) -> (; μ0, Σ0, R1, p)

Activation-coefficient state `k` for the temperature correction applied by
[`arrhenius_factor`](@ref). `k0` is the initial coefficient, `σ0_k` its initial standard
deviation and `σ1_k` the per-step process noise. The reference temperature `T0`, in °C, is
returned in `p` for `arrhenius_factor` to read back.
"""
function Arrhenius(; T0, k0, σ0_k, σ1_k)
    μ0 = ComponentVector(;
        k = k0
    )

    Σ0 = false .* μ0 * μ0'
    Σ0[:k, :k] = σ0_k^2

    R1 = diagm([σ1_k]) .^ 2

    p = (; T0)

    return (; μ0, Σ0, R1, p)
end

"""
    arrhenius_factor(x, T, p) -> Float64

Multiplicative resistance correction `exp(k (1/T − 1/T₀))` at temperature `T` in °C, taking
`k = abs(x.k)` from the [`Arrhenius`](@ref) state and `T₀` from `p.T0`. Returns 1 at the
reference temperature and falls below it as `T` rises.
"""
function arrhenius_factor(x, T, p)
    (; T0) = p
    T0_K = T0 + 273.15 # convert to Kelvin
    T_K = T + 273.15
    k = abs(x.k)
    return exp(k * (1 / T_K - 1 / T0_K))
end

"""
    R0(; r0, σ0, σ1) -> (; μ0, Σ0, R1)

Series-resistance state `r` as a random walk. `r0` is the initial resistance, `σ0` its
initial standard deviation and `σ1` the per-step process noise; `σ1 = 0` holds it constant
over the run.
"""
function R0(; r0, σ0, σ1)
    μ0 = ComponentVector(;
        r = r0
    )

    Σ0 = false .* μ0 * μ0'
    Σ0[:r, :r] = σ0^2

    R1 = diagm([σ1]) .^ 2

    return (; μ0, Σ0, R1)
end

"""
    RC(; v0, r0, τ0, σ0_v, σ0_r, σ0_τ, σ1_v, σ1_r, σ1_τ) -> (; μ0, Σ0, R1)

RC-branch state `(v, r, τ)` — overvoltage, resistance and time constant — advanced by
[`dynamics_rc`](@ref). Each `σ0_*` is the initial standard deviation of the matching state
and each `σ1_*` its per-step process noise. Use [`RC_VTau`](@ref) when the branch resistance
comes from a GP instead of the state.
"""
function RC(; v0, r0, τ0, σ0_v, σ0_r, σ0_τ, σ1_v, σ1_r, σ1_τ)
    μ0 = ComponentVector(
        v = v0,
        r = r0,
        τ = τ0,
    )
    Σ0 = false .* μ0 * μ0'
    Σ0[:v, :v] = σ0_v^2
    Σ0[:τ, :τ] = σ0_τ^2
    Σ0[:r, :r] = σ0_r^2

    R1 = diagm([σ1_v, σ1_r, σ1_τ]) .^ 2

    return (; μ0, Σ0, R1) # R2
end


# Build an `RC` from a θ sub-tuple, in the filter's z-scored units.
function _build_rc(θrc, zt)
    return RC(;
        v0 = StatsBase.transform(zt.σ, [θrc.v0]) |> first,
        σ0_v = StatsBase.transform(zt.σ, [θrc.σ0_v]) |> first,
        σ1_v = StatsBase.transform(zt.σ, [θrc.σ1_v]) |> first,
        r0 = StatsBase.transform(zt.r, [θrc.r0]) |> first,
        σ0_r = StatsBase.transform(zt.r, [θrc.σ0_r]) |> first,
        σ1_r = StatsBase.transform(zt.r, [θrc.σ1_r]) |> first,
        τ0 = θrc.τ0,
        σ0_τ = θrc.σ0_τ,
        σ1_τ = θrc.σ1_τ,
    )
end


"""
    dynamics_rc(rc, i, Ts; kT = 1.0) -> v

Advance the RC-branch overvoltage `rc.v` one step of `Ts` seconds under current `i`, using
the branch resistance `rc.r` scaled by the temperature factor `kT`. `rc.r` and `rc.τ` are
taken as absolute values.
"""
function dynamics_rc(rc, i, Ts; kT = 1.0)
    (; v, r, τ) = rc

    r = abs(r) * kT # force positive values
    τ = abs(τ)
    return exp(-Ts / τ) * v + i * r * (1 - exp(-Ts / τ))
end


"""
    RC_VTau(; v0, τ0, σ0_v, σ0_τ, σ1_v, σ1_τ) -> (; μ0, Σ0, R1)

RC-branch state `(v, τ)` for models that carry the branch resistance in a GP rather than the
state, advanced by [`dynamics_rc_vτ`](@ref). Arguments as in [`RC`](@ref), without the
resistance.
"""
function RC_VTau(; v0, τ0, σ0_v, σ0_τ, σ1_v, σ1_τ)
    μ0 = ComponentVector(
        v = v0,
        τ = τ0,
    )
    Σ0 = false .* μ0 * μ0'
    Σ0[:v, :v] = σ0_v^2
    Σ0[:τ, :τ] = σ0_τ^2

    R1 = diagm([σ1_v, σ1_τ]) .^ 2

    return (; μ0, Σ0, R1)
end

"""
    dynamics_rc_vτ(rc, r, i, Ts; kT = 1.0) -> v

Advance the RC-branch overvoltage `rc.v` one step of `Ts` seconds under current `i`, with the
branch resistance `r` supplied per step and scaled by the temperature factor `kT`. Sibling of
[`dynamics_rc`](@ref) for [`RC_VTau`](@ref), where the resistance comes from a GP; `r` is
used with its sign.
"""
function dynamics_rc_vτ(rc, r, i, Ts; kT = 1.0)
    (; v, τ) = rc

    # no abs on r: the EKF residual gradient punishes wrong-sign R1 directly, while abs would
    # mask the sign and create a stable negative-basin attractor
    r = r * kT
    τ = abs(τ)
    return exp(-Ts / τ) * v + i * r * (1 - exp(-Ts / τ))
end
