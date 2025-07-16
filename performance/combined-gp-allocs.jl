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


##
function dynamics!(dx, x, u, p, t)
    dx .= x # identity
end

dynamics(x, u, p, t) = x

function measurement_gp_noallocs(g, b, p)
    (; gp, b0, μ0, Σ0⁻¹, cache) = p
    (; k, H, Δg) = cache

    # (cov(gp, b, b0) * Σ0⁻¹) * (g - μ0) + mean(gp, b)
    #        c1                    c3
    #                c2
    cov!(k, gp, b0, b)
    mul!(H, k', Σ0⁻¹)
    Δg = g - μ0
    # Δg .= g .- μ0
    muladd(H, Δg, mean(gp, b))
end

function R2_noallocs(b, p)
    (; gp, b0, Σ0⁻¹, cache) = p
    (; k, H, k´) = cache
    cov!(k, gp, b0, b)
    mul!(H, k', Σ0⁻¹) # H
    @. k´ = -k
    muladd(H, k´, gp.kernel(b, b))
end

function measurement_combined_noallocs(x, u, p, t)
    (; xid) = p
    xc = ComponentVector(x, xid)
    @static_unpack x1, x2 = xc
    # (; x1, x2) = xc
    μ1 = measurement_gp_noallocs(x1, u[1], p.x1)
    μ2 = measurement_gp_noallocs(x2, u[1], p.x2)
    SA[μ2+u[2]*μ1]
end

function R2combined_noallocs(x, u, p, t)
    # (; xid) = p
    # c = ComponentVector(x, xid)
    R1 = R2_noallocs(u[1], p.x1)
    R2 = R2_noallocs(u[1], p.x2)
    SMatrix{1,1}(R1 + u[2]^2 * R2)
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
