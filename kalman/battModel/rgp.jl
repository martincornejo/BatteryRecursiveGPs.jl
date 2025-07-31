function RGP(gp, b0; σ2=1e-5, tr=ZScoreTransform(1, 1, [0.0], [1.0]), tr_b=ZScoreTransform(1, 1, [0.0], [1.0]))
    """
    Main function and only functoins user need to know
    """
    R1 = Diagonal(zero(b0))
    nx = length(b0)
    ny = 1

    x0 = mean(gp, b0)
    Σ0 = cov(gp, b0) + 1e-6I

    d0 = MvNormal(x0, Σ0)
    p = generate_p_gp(gp, d0, b0, σ2, tr, tr_b)

    rgp = (;
        dynamics=dynamics_gp,
        measurement=measurement_gp,
        R1=R1,
        R2=R2fun_gp,
        d0=d0,
        nx=nx,
        ny=ny,
        p=p
    )

    return rgp
end

function generate_p_gp(gp, d0, b0, σ2, tr, tr_b)
    μ0 = d0.μ
    Σ0 = d0.Σ
    Σ0⁻¹ = inv(Σ0)
    nb = length(μ0)

    v = similar(μ0)
    cache = (;
        k=DiffCache(v, nb),
        H=DiffCache(similar(μ0'), nb),
        Δx=DiffCache(v, nb),
    )

    p = (;
        gp=gp,
        b0=b0,
        μ0=μ0,
        Σ0⁻¹=Σ0⁻¹,
        tr=tr,
        tr_b=tr_b,
        σ2=σ2,
        cache=cache
    )
    return p
end

function R2fun_gp(x, u, p, t)
    (; gp, b0, Σ0⁻¹, σ2, tr, tr_b, cache) = p
    (; k, H, Δx) = cache

    k = get_tmp(k, x)
    H = get_tmp(H, x')
    Δx = get_tmp(Δx, x)

    b = StatsBase.transform(tr_b, u.b)  ## Each submodule is the one of retrieving its control parameter
    k .= cov(gp, b0, b)
    mul!(H, k', Σ0⁻¹)
    @. k = -k
    return tr.scale .^ 2 * (muladd(H, k, gp.kernel(b, b) .+ σ2 .^ 2))
end

function dynamics_gp(x, u, p, t)
    return x # identity
end


function measurement_gp(x, u, p, t)
    (; gp, b0, μ0, Σ0⁻¹, tr, tr_b, cache) = p
    (; k, H, Δx) = cache

    k = get_tmp(k, x)
    H = get_tmp(H, x')
    Δx = get_tmp(Δx, x)

    b = StatsBase.transform(tr_b, u.b)
    #cov!(k, gp, b0, b)
    k .= cov(gp, b0, b)
    mul!(H, k', Σ0⁻¹)
    Δx .= x - μ0
    return StatsBase.reconstruct(tr, muladd(H, Δx, mean(gp, b)))
end


function cov!(c::AbstractVector, gp::GP, x::AbstractVector, y::Real)
    @. c = gp.kernel(x, y)
end

mean_value(m::ZeroMean, x::Real) = zero(x)
mean_value(m::ConstMean, x::Real) = m.c
mean_value(m::CustomMean, x::Real) = m.f(x)

Statistics.mean(gp::GP, x::Real) = mean_value(gp.mean, x)

