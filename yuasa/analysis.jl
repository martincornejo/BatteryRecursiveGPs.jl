
function calc_deltaq(kf, df, zt; v=(3.8, 4.0))
    v1, v2 = v

    qmin, qmax = extrema(df.q)
    q = qmin:0.01:qmax
    q̂ = StatsBase.transform(zt.q, q)

    # OCV 
    ocv = predict_gp(kf, q̂, :ocv)
    μ = StatsBase.reconstruct(zt.v, ocv.μ)
    σ = StatsBase.reconstruct(zt.σ, ocv.σ)

    q1μ = q[findfirst(>=(v1), μ)]
    q1σ = q1μ - q[findfirst(>=(v1), μ + σ)]
    q1 = q1μ ± q1σ

    q2μ = q[findfirst(>=(v2), μ)]
    q2σ = q[findfirst(>=(v2), μ - σ)] - q2μ
    q2 = q2μ ± q2σ

    #
    # fμ = LinearInterpolation(q, μ)
    # fσ⁺ = LinearInterpolation(q, μ + σ)
    # fσ⁻ = LinearInterpolation(q, μ - σ)
    # q1μ = fμ(v1)
    # q1σ = q1μ - fσ⁺(v1)
    # q1 = q1μ ± q1σ

    # q2μ = fμ(v2)
    # q2σ = fσ⁻(v2) - q2μ
    # q2 = q2μ ± q2σ

    q2 - q1
end

function calc_Q(kf, df, zt, fsoc; v=(3.8, 4.0))
    v1, v2 = v
    Δsoc = fsoc(v2) - fsoc(v1)
    Δq = calc_deltaq(kf, df, zt; v)
    Δq / (Δsoc)
end

function calc_soh(kf, df, zt, fsoc, Q; v=(3.8, 4.0))
    Q´ = calc_Q(kf, df, zt, fsoc; v)
    return Q´ / Q
end

function calc_soc0(kf, df, zt, fsoc; v=(3.8, 4.0))
    v1, v2 = v

    qmin, qmax = extrema(df.q)
    q = qmin:0.01:qmax
    q̂ = StatsBase.transform(zt.q, q)

    # OCV 
    ocv = predict_gp(kf, q̂, :ocv)
    μ = StatsBase.reconstruct(zt.v, ocv.μ)
    σ = StatsBase.reconstruct(zt.σ, ocv.σ)

    q1μ = q[findfirst(>=(v1), μ)]
    q1σ = q1μ - q[findfirst(>=(v1), μ + σ)]
    q1 = q1μ ± q1σ

    # fμ = LinearInterpolation(q, μ)
    # fσ⁺ = LinearInterpolation(q, μ + σ)
    # # fσ⁻ = LinearInterpolation(q, μ - σ)
    # q1μ = fμ(v1)
    # q1σ = q1μ - fσ⁺(v1)
    # q1 = q1μ ± q1σ

    Q´ = calc_Q(kf, df, zt, fsoc; v)
    Δs = q1 / Q´

    s0 = fsoc(v1)
    s0 - Δs
end

