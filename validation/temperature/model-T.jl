
function fit_zscore(df)
    v = StatsBase.fit(ZScoreTransform, df.v)
    σ = StatsBase.fit(ZScoreTransform, df.v, center=false)
    i = StatsBase.fit(ZScoreTransform, df.i, center=false)
    q = StatsBase.fit(ZScoreTransform, df.q)
    r = ZScoreTransform(1, 1, [0.0], [σ.scale[1] / i.scale[1]])
    return (; v, σ, i, q, r)
end

function normalize_data(df, zt)
    v = StatsBase.transform(zt.v, df.v)
    i = StatsBase.transform(zt.i, df.i)
    q = StatsBase.transform(zt.q, df.q)
    return DataFrame(; df.t, v, i, q, T=df.T)
end

# === model
dynamics(x, u, p, t) = x

function measurement_combined(x, u, p, t)
    (; xid) = p
    xc = ComponentVector(x, xid)
    ocv = measurement_gp(p.ocv, xc.ocv, u.q)
    r0 = measurement_gp(p.r0, xc.r0, u.q)
    ocv + u.i * r0 |> SVector{1}
end

function R2combined(x, u, p, t)
    ocv = uncertainty_gp(p.ocv, u.q)
    r0 = uncertainty_gp(p.r0, u.q)
    ocv + u.i^2 * r0 |> SMatrix{1,1}
end

function Cjac(x, u, p, t)
    (; C) = p.cache
    ForwardDiff.jacobian!(C, x -> measurement_combined(x, u, p, t), x)
    # return Cjac
end

function Ajac(x, u, p, t)
    (; A) = p.cache
    return A
end

function predict(kf, df, zt)
    dfn = normalize_data(df, zt)
    ocv = predict_gp(kf, dfn.q, :ocv)
    r0 = predict_gp(kf, dfn.q, :r0)
    μ̂ = @. ocv.μ + dfn.i * r0.μ
    σ̂ = @. sqrt(ocv.σ + dfn.i^2 * r0.σ)
    μ = StatsBase.reconstruct(zt.v, μ̂)
    σ = StatsBase.reconstruct(zt.σ, σ̂)
    (; μ, σ)
end

function build_kf(dfn, n=21)
    qmin, qmax = extrema(dfn.q)
    q0n = range(qmin, qmax, n) |> collect
    # q0n = StatsBase.transform(zt.q, q0)
    r0 = StatsBase.transform(zt.r, [1e-3]) |> first

    # kernel1 = LinearKernel() + 0.02 * with_lengthscale(SEKernel(), 0.33)
    kernel1 = 0.2 * with_lengthscale(SEKernel(), 0.33)
    rgp1 = RGP(kernel1, q0n)

    kernel2 = 0.01 * with_lengthscale(SEKernel(), 0.5)
    rgp2 = RGP(r0, kernel2, q0n)

    nx = (length(rgp1.μ0) + (length(rgp2.μ0)))
    p = (; cache=(;
        A=I(nx),
        C=zeros(1, nx),
    ))
    rgps = (; ocv=rgp1, r0=rgp2)

    make_ekf(rgps, dynamics, measurement_combined, R2combined; Ajac, Cjac, p)
end

