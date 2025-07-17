function RGP(gp, b0; σ2=1e-5, tr=ZScoreTransform(1, 1, [0.0], [1.0]), tr_b=ZScoreTransform(1, 1, [0.0], [1.0]))
    """
    Main function and only functoins user need to know
    """
    R1 = Diagonal(zero(b0))
    nx = length(b0)
    ny = 1


    p = generate_p_gp(gp, b0, σ2, tr, tr_b)

    x0 = mean(gp, b0)
    Σ0 = cov(gp, b0) + 1e-6I

    d0 = MvNormal(x0, Σ0)

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

function generate_p_gp(gp, b0, σ2, tr, tr_b)
    μ = mean(gp, b0)
    Σ = cov(gp, b0) + 1e-6I
    nb = length(b0)
    μ0 = μ
    Σ0⁻¹ = inv(Σ)
    cache = (
        k=similar(b0),
        k´=similar(b0),
        H=similar(b0'),
        Δg=similar(b0),
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
    (; k, H, Δg) = cache
    b = StatsBase.transform(tr_b, u.b)  ## Each submodule is the one of retrieving its control parameter
    H = cov(gp, b, b0) * Σ0⁻¹
    return tr.scale .^ 2 * (cov(gp, b) - H * cov(gp, b0, b) + I * σ2 .^ 2)
end

function dynamics_gp(x, u, p, t)
    return x # identity
end

function measurement_gp(x, u, p, t)
    (; gp, b0, μ0, Σ0⁻¹, tr, tr_b, cache) = p
    (; k, H, Δg) = cache
    b = StatsBase.transform(tr_b, u.b)[1]
    @. k = gp.kernel(b, b0)
    mul!(H, k', Σ0⁻¹)
    Δx = x - μ0
    return StatsBase.reconstruct(tr, muladd(H, Δx, mean(gp, b)))
end

