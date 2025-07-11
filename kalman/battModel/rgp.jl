
function RGP(gp, b0; σ=1e-5, tr=1)
    """
    Main function and only functoins user need to know
    """
    R1 = Diagonal(zero(b0))
    nx = length(b0)
    ny = 1


    p = generate_p_gp(gp, b0, σ, tr)

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

function generate_p_gp(gp, b0, σ, tr)
    μ = mean(gp, b0)
    Σ = cov(gp, b0) + 1e-6I
    μ0 = μ
    Σ0⁻¹ = inv(Σ)

    p = (;
        gp=gp,     # gp (mean + kernel functions)
        b0=b0,      # basis vector
        μ0=μ0,     # ptir of basis vector
        Σ0⁻¹=Σ0⁻¹,
        tr=tr,
        σ=σ      # inv convariance basis vector
    )
    return p
end


function R2fun_gp(x, u, p, t)
    (; gp, b0, Σ0⁻¹, σ, tr) = p
    b = u.b  ## Each submodule is the one of retrieving its control parameter
    H = cov(gp, b, b0) * Σ0⁻¹
    return tr.scale .^ 2 * (cov(gp, b) - H * cov(gp, b0, b) + I * σ .^ 2)
end

function dynamics_gp(x, u, p, t)
    return x # identity
end

function measurement_gp(x, u, p, t)
    (; gp, b0, μ0, Σ0⁻¹, tr) = p

    b = u.b ## Each submodule is the one of retrieving its control parameter

    H = cov(gp, b, b0) * Σ0⁻¹
    ## Denormalizing the measurement))
    return StatsBase.reconstruct(tr, mean(gp, b) + H * (x - μ0))
end

