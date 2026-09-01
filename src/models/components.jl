function CoulombCounting(; q0 = 0.0, σ0 = 0.0, σ1)
    μ0 = ComponentVector(q = q0)
    Σ0 = false .* μ0 * μ0'
    Σ0[:q, :q] = σ0^2

    R1 = [σ1^2;;]

    return (; μ0, Σ0, R1)
end

function dynamics_cc(cc, i, Ts)
    (; q) = cc
    return q + i * Ts / (3600)
end

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

function arrhenius_factor(x, T, p)
    (; T0) = p
    T0_K = T0 + 273.15 # convert to Kelvin
    T_K = T + 273.15
    k = abs(x.k)
    return exp(k * (1 / T_K - 1 / T0_K))
end

function R0(; r0, σ0, σ1)
    μ0 = ComponentVector(;
        r = r0
    )

    Σ0 = false .* μ0 * μ0'
    Σ0[:r, :r] = σ0^2

    R1 = diagm([σ1]) .^ 2

    return (; μ0, Σ0, R1)
end

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


function dynamics_rc(rc, i, Ts; kT = 1.0)
    (; v, r, τ) = rc

    r = abs(r) * kT # force positive values
    τ = abs(τ)
    return exp(-Ts / τ) * v + i * r * (1 - exp(-Ts / τ))
end


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

# Sibling of `dynamics_rc` for the case where R1 lives in a GP and is
# supplied externally per step. No abs on r: the EKF residual gradient
# punishes wrong-sign R1 directly, instead of the abs trick which masks
# sign and creates a stable negative-basin attractor.
function dynamics_rc_vτ(rc, r, i, Ts; kT = 1.0)
    (; v, τ) = rc

    r = r * kT
    τ = abs(τ)
    return exp(-Ts / τ) * v + i * r * (1 - exp(-Ts / τ))
end
