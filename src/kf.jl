
dynamics(x, u, p, t) = x
measurement(x, u, p, t) = SA[measurement_gp(p, x, u)]
R2(x, u, p, t) = SA[uncertainty_gp(p, u)]

function make_kf(rgp::RGP)
    (; μ0, Σ0) = rgp
    d0 = MvNormal(μ0, Σ0)
    nx = length(μ0)
    nu = 1
    ny = 1
    R1 = Diagonal(zero(μ0))
    ExtendedKalmanFilter(dynamics, measurement, R1, R2, d0; nx, nu, ny, p=rgp)
end








function make_kf_opt()
    m1(x) = 0.1 + 0.5 .* x
    kernel1 = 0.02 * with_lengthscale(SEKernel(), 0.1)
    gp1 = GP(m1, kernel1)

    kernel2 = LinearKernel() + 0.02 * with_lengthscale(SEKernel(), 0.1)
    gp2 = GP(kernel2)

    b0 = collect(0:0.05:1)
    nb = length(b0)

    # initial guess
    μ1 = mean(gp1, b0)
    # μ1 = SVector{nb}(mean(gp1, b0))
    Σ1 = cov(gp1, b0) + 1e-6I
    Σ1⁻¹ = inv(Σ1)


    μ2 = mean(gp2, b0)
    # μ2 = SVector{nb}(mean(gp2, b0))
    Σ2 = cov(gp2, b0) + 1e-6I
    Σ2⁻¹ = inv(Σ2)


    x0 = ComponentVector(; x1=μ1, x2=μ1)
    Σ0 = false .* x0 * x0'
    Σ0[:x1, :x1] = Σ1
    Σ0[:x2, :x2] = Σ2

    xid = getaxes(x0)
    Σid = getaxes(Σ0)

    d0 = MvNormal(x0, Σ0)


    p = (;
        xid,
        Σid,
        x1=(; f=f1, # only for validation purposes
            b0,      # basis vector
            gp=gp1,     # gp (mean + kernel functions)
            μ0=SVector{nb}(μ1),     # mean basis vector
            # Σ0,
            Σ0⁻¹=Σ1⁻¹,   # inv convariance basis vector,
            cache=(
                k=similar(b0),
                k´=similar(b0),
                H=similar(b0'),
                Δg=similar(b0),
            ),
        ),
        x2=(;
            f=f2, # only for validation purposes
            b0,      # basis vector
            # gp=gp2,     # gp (mean + kernel functions)
            gp=gp1,     # gp (mean + kernel functions)
            # μ0=μ2,     # mean basis vector
            μ0=SVector{nb}(μ1),     # mean basis vector
            # Σ0,
            # Σ0⁻¹=Σ2⁻¹,   # inv convariance basis vector
            Σ0⁻¹=Σ1⁻¹,   # inv convariance basis vector
            cache=(
                k=similar(b0),
                k´=similar(b0),
                H=similar(b0'),
                Δg=similar(b0),
            ),
        ),
        Ajac=I(2nb),
        cache=(
            C=zeros(1, 2nb),
        )
    )

    # R1 = SMatrix{2nb,2nb}(Diagonal(zero(x0)))
    R1 = Diagonal(zero(x0))
    fAjac(x, u, p, t) = p.Ajac
    function fCjac(x, u, p, t)
        (; cache) = p
        (; C) = cache
        ForwardDiff.jacobian!(C, x -> measurement_combined_noallocs(x, u, p, t), x)
        # return C
    end
    kf = ExtendedKalmanFilter(dynamics, measurement_combined_noallocs, R1, R2combined_noallocs, d0; Ajac=fAjac, Cjac=fCjac, nx=length(x0), ny=1, nu=1, p)
    # kf = UnscentedKalmanFilter(dynamics!, measurement_combined_noallocs, R1, R2combined_noallocs, d0; nx=length(x0), ny=1, nu=1, p)
    # return d0
end