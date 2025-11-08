dynamics_params(x, u, p, t) = x

function measurement_params(x, u, p, t)
    (; xid) = p
    xc = ComponentVector(x, xid)
    ocv = measurement_gp(p.ocv, xc.ocv, u.s)
    r0 = measurement_gp(p.r0, xc.r0, u.s)
    ocv + u.i * r0 |> SVector{1}
end

function R2_params(x, u, p, t)
    ocv = uncertainty_gp(p.ocv, u.s)
    r0 = uncertainty_gp(p.r0, u.s)
    ocv + u.i^2 * r0 |> SMatrix{1,1}
end

function Cjac_params(x, u, p, t)
    (; C) = p.cache
    ForwardDiff.jacobian!(C, x -> measurement_params(x, u, p, t), x)
    # return Cjac
end

function Ajac_params(x, u, p, t)
    (; A) = p.cache
    return A
end

function predict(kf, df)
    dfn = normalize_data(df)
    ocv = predict_gp(kf, dfn.s, :ocv)
    r0 = predict_gp(kf, dfn.s, :r0)
    μ = @. ocv.μ + u.i * r0.μ
    σ = @. ocv.σ + u.i^2 * r0.σ
    (; μ, σ)
end

function build_kf_params(focv_prior, n=21)
    b0 = collect(range(0, 1, n))
    # b0n = StatsBase.transform(zt.s, b0)
    r0 = StatsBase.transform(zt.r, [15e-3]) |> first

    # kernel1 = 0.1 * with_lengthscale(SEKernel(), 0.1)
    kernel1 = 0.1 * with_lengthscale(SEKernel(), 0.05)
    gp1 = GP(focv_prior, kernel1)
    rgp1 = RGP(gp1, b0)

    # kernel2 = 0.001 * with_lengthscale(SEKernel(), 0.5)
    kernel2 = 0.001 * with_lengthscale(SEKernel(), 0.5)
    gp2 = GP(r0, kernel2)
    rgp2 = RGP(gp2, b0)

    nx = (length(rgp1.μ0) + (length(rgp2.μ0)))
    p = (;
        cache=(;
            A=I(nx),
            C=zeros(1, nx),
        ),
    )
    rgps = (; ocv=rgp1, r0=rgp2)

    make_ekf(rgps, dynamics_params, measurement_params, R2_params; Ajac=Ajac_params, Cjac=Cjac_params, p)
end