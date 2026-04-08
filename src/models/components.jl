function ColoumbCounting(; q0=0.0, σ0=0.0, σ1)
    μ0 = ComponentVector(q=q0)
    Σ0 = false .* μ0 * μ0'
    Σ0[:q, :q] = σ0

    R1 = [σ1^2;;]

    return (; μ0, Σ0, R1)
end

function dynamics_cc(x, u, p)
    (; Ts) = p
    (; q) = x.cc
    (; i) = u
    q + i * Ts / (3600)
end

function Arrhenius(; T0, k0, σ0_k, σ1_k)
    μ0 = ComponentVector(;
        k=k0
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
    exp(k * (1 / T_K - 1 / T0_K))
end

function R0(; r0, σ0, σ1)
    μ0 = ComponentVector(;
        r=r0
    )

    Σ0 = false .* μ0 * μ0'
    Σ0[:r, :r] = σ0^2

    R1 = diagm([σ1]) .^ 2

    return (; μ0, Σ0, R1)
end

function RC(; v0, r0, τ0, σ0_v, σ0_r, σ0_τ, σ1_v, σ1_r, σ1_τ)
    μ0 = ComponentVector(
        v=v0,
        r=r0,
        τ=τ0,
    )
    Σ0 = false .* μ0 * μ0'
    Σ0[:v, :v] = σ0_v^2
    Σ0[:τ, :τ] = σ0_τ^2
    Σ0[:r, :r] = σ0_r^2

    R1 = diagm([σ1_v, σ1_r, σ1_τ]) .^ 2

    return (; μ0, Σ0, R1) # R2
end


function dynamics_rc(x, u, p)
    (; Ts) = p
    (; i, T) = u
    (; v, r, τ) = x.rc
    kT = arrhenius_factor(x.arr, T, p.arr)

    r = abs(r) * kT # force positive values
    τ = abs(τ)
    exp(-Ts / τ) * v + i * r * (1 - exp(-Ts / τ))
end
