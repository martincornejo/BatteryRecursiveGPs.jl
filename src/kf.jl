
struct RGPModel{K}
    rgp::RGP
    kf::K
end

dynamics(x, u, p, t) = x
measurement(x, u, p, t) = SA[measurement_gp(p.rgp, x, u[1])]
R2(x, u, p, t) = SMatrix{1,1}(uncertainty_gp(p.rgp, u[1]))

function fAjac(x, u, p, t)
    (; Ajac) = p.cache
    fAjac_gp!(Ajac, x, u, p, t)
    return Ajac
end

function fCjac(x, u, p, t)
    (; cache) = p
    (; Cjac) = cache
    ForwardDiff.jacobian!(Cjac, x -> measurement(x, u, p, t), x)
    # return Cjac
end

function make_kf(rgp::RGP)
    (; μ0, Σ0) = rgp
    nb = length(μ0)
    p = (;
        rgp,
        cache=(;
            Ajac=I(nb),
            Cjac=zeros(1, nb)
        )
    )
    d0 = LLPF.SimpleMvNormal(μ0, Σ0)
    nx = length(μ0)
    nu = 1
    ny = 1
    R1 = Diagonal(zero(μ0))
    ExtendedKalmanFilter(dynamics, measurement, R1, R2, d0; nx, nu, ny, p, Ajac=fAjac, Cjac=fCjac)
end


function predict_gp(kf, rgp, b)
    (; gp, b0, μ0, Σ0⁻¹) = kf.p.rgp
    H = cov(gp, b, b0) * Σ0⁻¹
    m = mean(gp, b)
    μ = H * (kf.x - μ0) + m

    R = cov(gp, bgp) - H * cov(gp, b0, bgp) #eq.7 
    Σgp = R + H * kf.R * H' #eq.9
    σ = sqrt.(diag(Σgp))
    (; μ, σ)
end

