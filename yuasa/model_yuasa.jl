

function RC(; v0, r0, τ0, σ0_v, σ0_r, σ0_τ, σ1_v, σ1_r, σ1_τ)
    μ0 = ComponentVector(
        v=v0,
        r=r0, #
        τ=τ0,
    )
    Σ0 = false .* μ0 * μ0'
    Σ0[:v, :v] = σ0_v^2
    Σ0[:τ, :τ] = σ0_τ^2
    Σ0[:r, :r] = σ0_r^2

    R1 = diagm([σ1_v, σ1_r, σ1_τ]) .^ 2
    # R2 = σ2 .^ 2, p # let's put all R2 together in a single param

    return (; μ0, Σ0, R1) # R2
end

function dynamics_rc(x, i, p)
    (; Ts) = p
    (; v, r, τ) = x
    exp(-Ts / τ) * v + i * r * (1 - exp(-Ts / τ))

end

# === model
function dynamics!(x⁺, x⁻, u, p, t)
    (; xid) = p
    xc⁻ = ComponentVector(x⁻, xid)
    xc⁺ = ComponentVector(x⁺, xid)
    xc⁺ .= xc⁻ # previous values

    xc⁺.rc.v = dynamics_rc(xc⁻.rc, u.i, p)
    nothing # IPD
end

function measurement(x, u, p, t)
    (; xid) = p
    xc = ComponentVector(x, xid)

    ocv = measurement_gp(p.ocv, xc.ocv, u.q)
    r0 = measurement_gp(p.r0, xc.r0, u.q)
    vrc = xc.rc.v # measurement rc
    ocv + u.i * r0 + vrc |> SVector{1}
end

function R2(x, u, p, t)
    (; vσ²) = p
    ocv = uncertainty_gp(p.ocv, u.q)
    r0 = uncertainty_gp(p.r0, u.q)
    ocv + u.i^2 * r0 + vσ² |> SMatrix{1,1}
end


##
function build_kf(θ, ϑ, df, zt; n=21)
    # basis vectors
    dfn = normalize_data(zt, df)
    qmin, qmax = extrema(dfn.q)
    b0 = range(qmin, qmax, n) |> collect

    # OCV GP
    kernel1 = θ.ocv.σ * with_lengthscale(SEKernel(), θ.ocv.ℓ)
    rgp1 = RGP(kernel1, b0)

    # R0 GP
    r0 = StatsBase.transform(zt.r, [ϑ.r0.r0]) |> first
    kernel2 = θ.r0.σ * with_lengthscale(SEKernel(), θ.r0.ℓ)
    rgp2 = RGP(r0, kernel2, b0)

    # RC
    r1 = StatsBase.transform(zt.r, [ϑ.rc.r0]) |> first
    rc = RC(; r0=r1, τ0=ϑ.rc.τ0, v0=ϑ.rc.v0, θ.rc...)

    # measurement / model noise
    vσ² = StatsBase.transform(zt.σ, [θ.vσ^2]) |> first

    # model
    nx = (length(rc.μ0) + length(rgp1.μ0) + (length(rgp2.μ0)))
    p = (;
        # cache=(;
        #     A=I(nx),
        #     C=zeros(1, nx),
        # ),
        Ts=ϑ.Ts,
        vσ²,
    )
    rgps = (; ocv=rgp1, r0=rgp2, rc=rc)

    make_ekf(rgps, dynamics!, measurement, R2; p)
end


function save_train_kf(kf, us, ys; step_size=1000)
    μs = []
    Σs = []
    n = 1
    (; xid, Σid) = kf.p
    for (u, y) in zip(us, ys)
        LLPF.update!(kf, u, y)
        if n % step_size == 0
            push!(μs, copy(ComponentVector(kf.x, xid)))
            push!(Σs, copy(ComponentMatrix(kf.R, Σid)))
        end
        n += 1
    end
    evo = (; μs, Σs)
    return evo
end


function model_predict(kf, u)
    (; xid, vσ²) = kf.p
    xc = ComponentVector(kf.x, xid)

    vrc = xc.rc.v
    ocv = predict_gp(kf, [u.q], :ocv)
    r0 = predict_gp(kf, [u.q], :r0)
    μ = ocv.μ[1] + u.i * r0.μ[1] + vrc
    σ = sqrt(ocv.σ[1]^2 + u.i^2 * r0.σ[1]^2 + vσ²) # TODO: vσ
    # μ = StatsBase.reconstruct(zt.v, [μ̂]) |> first
    # σ = StatsBase.reconstruct(zt.σ, [σ̂]) |> first
    (; μ, σ)
end

function run_sim!(kf, us, ys, ut)
    vμ = Float64[]
    vσ = Float64[]
    # sμ = Float64[]
    # sσ = Float64[]
    for (u, y) in zip(us, ys)
        (vμᵢ, vσᵢ) = model_predict(kf, u)
        push!(vμ, vμᵢ)
        push!(vσ, vσᵢ)
        # push!(sμ, kf.x[1])
        # push!(sσ, sqrt(kf.R[1, 1]))
        LLPF.update!(kf, u, y)
    end

    for u in ut
        vμᵢ, vσᵢ = model_predict(kf, u)
        push!(vμ, vμᵢ)
        push!(vσ, vσᵢ)
        # push!(sμ, kf.x[1])
        # push!(sσ, sqrt(kf.R[1, 1]))
        LLPF.predict!(kf, u)
    end

    # vμ = StatsBase.reconstruct(zt.v, vμ)
    # vσ = StatsBase.reconstruct(zt.σ, vσ)

    # return (; vμ, vσ, sμ, sσ)
    return (; vμ, vσ)
end
