# Extends AbstractGPs to evaluate `mean` and `cov` of a GP to single values (instead of `Vector`s only)
mean_value(m::ZeroMean, x::Real) = zero(x)
mean_value(m::ConstMean, x::Real) = m.c
mean_value(m::CustomMean, x::Real) = m.f(x)

Statistics.mean(gp::GP, x::Real) = mean_value(gp.mean, x)

Statistics.cov(gp::GP, x::AbstractVector, y::Real) = gp.kernel.(x, y)
Statistics.cov(gp::GP, x::Real, y::AbstractVector) = gp.kernel.(x, y)'

function cov!(c::AbstractVector, gp::GP, x::AbstractVector, y::Real)
    @. c = gp.kernel(x, y)
end

struct RGP{bT,BT,RT,cT}
    gp::GP
    b0::bT
    μ0::bT
    Σ0::BT
    Σ0⁻¹::BT
    R1::RT
    cache::cT
end


function RGP(gp::GP, b0::T) where T<:AbstractArray
    nb = length(b0) # 1-dim basis vector (for now)

    μ0 = mean(gp, b0) |> T
    # μ0 = SVector{nb}(mean(gp, b0))
    Σ0 = cov(gp, b0) + 1e-6I
    Σ0⁻¹ = inv(Σ0)

    R1 = zeros(nb, nb)

    cache = (;
        k=DiffCache(similar(b0)),
        k⁻=DiffCache(similar(b0)),
        H=DiffCache(similar(b0')),
        # Δg=similar(b0), # <- use DiffCache
    )

    RGP(gp, b0, μ0, Σ0, Σ0⁻¹, R1, cache)
end


function RGP(kernel::Kernel, b0::AbstractArray)
    gp = GP(kernel)
    RGP(gp, b0)
end

function RGP(mean, kernel::Kernel, b0::AbstractArray)
    gp = GP(mean, kernel)
    RGP(gp, b0)
end


# dynamics(x, u, p, t) = x
function measurement_gp(rgp::RGP, g::AbstractArray, b::Real)
    (; gp, b0, μ0, Σ0⁻¹, cache) = rgp
    # (; k, H) = cache
    # Δg = get_tmp(cache.Δg, g)
    k = get_tmp(cache.k, b)
    H = get_tmp(cache.H, b)

    # (cov(gp, b, b0) * Σ0⁻¹) * (g - μ0) + mean(gp, b)
    #        k                    Δg
    #                H
    cov!(k, gp, b0, b) # k = cov(gp, b, b0)
    mul!(H, k', Σ0⁻¹) # H = k' * Σ0⁻¹
    Δg = g - μ0
    # Δg .= g .- μ0
    muladd(H, Δg, mean(gp, b)) # H * (g - μ0) + m
end

function uncertainty_gp(rgp::RGP, b::Real)
    (; gp, b0, Σ0⁻¹, cache) = rgp
    k = get_tmp(cache.k, b)
    H = get_tmp(cache.H, b)
    k⁻ = get_tmp(cache.k⁻, b)
    cov!(k, gp, b0, b)
    mul!(H, k', Σ0⁻¹) # H
    @. k⁻ = -k
    muladd(H, k⁻, gp.kernel(b, b))
end

