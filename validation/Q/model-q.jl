function SOC(soc0, σ0_soc, σ1_soc, q0, σ0_q, σ1_q)
    x0 = ComponentVector(
        soc=soc0,
        q=q0,
    )
    Σ0 = false .* x0 * x0'
    Σ0[:soc, :soc] = σ0_soc
    Σ0[:q, :q] = σ0_q

    # R1 = [σ1;;]
    R1 = diagm([σ1_soc, σ1_q])

    return (; μ0=x0, Σ0, R1)
end

function dynamics!(x⁺, x, u, p, t)
    (; Ts, xid) = p # params
    (; i) = u # control
    x = ComponentVector(x, xid)
    x⁺ = ComponentVector(x⁺, xid)
    x⁺ .= x # previous values

    x⁺.soc.soc = x.soc.soc + i * Ts / (x.soc.q * 3600)
    # x⁺.soc.soc = x.soc.soc + i * Ts / (q * 3600)
    # dx[2] = v1 * exp(-Ts / τ1) + i * R1 * (1 - exp(-Ts / τ1))
end

function dynamics(x, u, p, t)
    x⁺ = similar(x)
    dynamics!(x⁺, x, u, p, t)
    return x⁺
end

function measurement(x, u, p, t)
    (; xid) = p
    xc = ComponentVector(x, xid)
    ocv = measurement_gp(p.ocv, xc.ocv, xc.soc.soc)
    r0 = measurement_gp(p.r0, xc.r0, xc.soc.soc)
    ocv + u.î * r0 |> SVector{1}
end

function R2(x, u, p, t)
    (; xid, vσ) = p
    xc = ComponentVector(x, xid)
    ocv = uncertainty_gp(p.ocv, xc.soc.soc)
    r0 = uncertainty_gp(p.r0, xc.soc.soc)
    ocv + u.î^2 * r0 + vσ |> SMatrix{1,1}
end

function Cjac(x, u, p, t)
    (; C) = p.cache
    ForwardDiff.jacobian!(C, x -> measurement(x, u, p, t), x)
    # return Cjac
end

function Ajac(x, u, p, t)
    # (; A) = p.cache
    return ForwardDiff.jacobian(x -> dynamics(x, u, p, t), x) # TODO: do more performant
end

# function predict(kf, df)
#     dfn = normalize_data(df)
#     ocv = predict_gp(kf, dfn.s, :ocv)
#     r0 = predict_gp(kf, dfn.s, :r0)
#     μ = @. ocv.μ + u.i * r0.μ
#     σ = @. ocv.σ + u.i^2 * r0.σ
#     (; μ, σ)
# end

function build_kf(θ, ϑ, focv_prior, zt; n=21)
    b0 = collect(range(0, 1, n)) # basis vector
    # priors
    r0 = StatsBase.transform(zt.r, [15e-3]) |> first

    # OCV GP
    kernel1 = θ.ocv.σ * with_lengthscale(SEKernel(), θ.ocv.ℓ)
    rgp1 = RGP(focv_prior, kernel1, b0)

    # R0 GP
    kernel2 = θ.r0.σ * with_lengthscale(SEKernel(), θ.r0.ℓ)
    rgp2 = RGP(r0, kernel2, b0)

    # SOC / SOH estimation
    soc = SOC(θ.soc.soc0, θ.soc.σ1, θ.soc.σ2, θ.q.q0, θ.q.σ1, θ.q.σ2)

    # model
    nx = (length(soc.μ0) + length(rgp1.μ0) + (length(rgp2.μ0)))
    p = (;
        cache=(;
            A=I(nx),
            C=zeros(1, nx),
        ),
        Ts=ϑ.Ts,
        # q=ϑ.q,
        vσ=StatsBase.transform(zt.σ, [0.005^2]) |> first,
    )
    rgps = (; soc, ocv=rgp1, r0=rgp2)

    make_ekf(rgps, dynamics!, measurement, R2; Ajac, Cjac, p)
end

function model_predict(kf, u)
    s = kf.x[1]
    ocv = predict_gp(kf, [s], :ocv)
    r0 = predict_gp(kf, [s], :r0)
    μ = ocv.μ[1] + u.î * r0.μ[1]
    σ = sqrt(ocv.σ[1]^2 + u.î^2 * r0.σ[1]^2) # TODO: vσ
    # μ = StatsBase.reconstruct(zt.v, [μ̂]) |> first
    # σ = StatsBase.reconstruct(zt.σ, [σ̂]) |> first
    (; μ, σ)
end

function run_sim!(kf, us, ys, ut)
    vμ = Float64[]
    vσ = Float64[]
    sμ = Float64[]
    sσ = Float64[]
    for (u, y) in zip(us, ys)
        (vμᵢ, vσᵢ) = model_predict(kf, u)
        push!(vμ, vμᵢ)
        push!(vσ, vσᵢ)
        push!(sμ, kf.x[1])
        push!(sσ, sqrt(kf.R[1, 1]))
        LLPF.update!(kf, u, y)
    end

    for u in ut
        vμᵢ, vσᵢ = model_predict(kf, u)
        push!(vμ, vμᵢ)
        push!(vσ, vσᵢ)
        push!(sμ, kf.x[1])
        push!(sσ, sqrt(kf.R[1, 1]))
        LLPF.predict!(kf, u)
    end

    vμ = StatsBase.reconstruct(zt.v, vμ)
    vσ = StatsBase.reconstruct(zt.σ, vσ)

    return (; vμ, vσ, sμ, sσ)
end

function plot_soc_estimation(sol, df)
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]
    s´ = sol.xt .|> first
    sσ = [sqrt(R[1, 1]) for R in sol.Rt]
    lines!(ax[1], df.t / 3600, s´)
    band!(ax[1], df.t / 3600, s´ - 2sσ, s´ + 2sσ, alpha=0.5)
    lines!(ax[1], df.t / 3600, df.s)

    Δ = s´ - df.s
    lines!(ax[2], df.t / 3600, Δ)
    band!(ax[2], df.t / 3600, Δ - 2sσ, Δ + 2sσ, alpha=0.5)

    xlims!(ax[1], df[begin, :t] / 3600, df[end, :t] / 3600)
    xlims!(ax[2], df[begin, :t] / 3600, df[end, :t] / 3600)
    linkxaxes!(ax...)
    fig
end

function plot_q_trajectory(sol, q´; Ts=1.0)
    fig = Figure()
    ax = Axis(fig[1, 1])
    q = getindex.(sol.xt, 2)
    # R = getindex.(sol.Rt, 2, 2)
    σ = [sqrt(R[2, 2]) for R in sol.Rt]
    tt = (1:Ts:(Ts*length(q))) ./ 3600
    hlines!(ax, q´, color=:black, linestyle=:dash)
    lines!(ax, tt, q)
    band!(ax, tt, q - 2σ, q + 2σ, alpha=0.5)
    ax.ylabel = "Cell capacity / Ah"
    fig
end

function plot_ecm(kf, df, zt, focv, fR0; focv_prior=nothing, closeup=false, soc_shift=0.0)
    fig = Figure(size=(600, 600))
    colors = Makie.wong_colors()
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / Ω"
    ax[2].xlabel = "SOC / p.u."
    hidexdecorations!(ax[1], ticks=false, grid=false)

    soc = 0:0.01:1

    # OCV 
    ocv = predict_gp(kf, soc, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
    ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)

    lines!(ax[1], soc .+ soc_shift, ocvμ)
    band!(ax[1], soc .+ soc_shift, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha=0.8)
    lines!(ax[1], soc, focv(soc), color=:black, linestyle=:dot)
    if focv_prior !== nothing
        ocv_prior = StatsBase.reconstruct(zt.v, focv_prior(soc))
        lines!(ax[1], soc, ocv_prior, color=:green, linestyle=:dot)
    end

    if closeup
        ylims!(ax[1], 3.45, 4.2)
    end

    # R0
    r0 = predict_gp(kf, soc, :r0)
    rμ = StatsBase.reconstruct(zt.r, r0.μ)
    rσ = StatsBase.reconstruct(zt.r, r0.σ)

    lines!(ax[2], soc .+ soc_shift, rμ)
    band!(ax[2], soc .+ soc_shift, rμ + 2rσ, rμ - 2rσ, alpha=0.8)
    lines!(ax[2], soc, fR0.(soc), color=:black, linestyle=:dot)

    # data - SOC window
    smin, smax = df.s |> extrema
    vlines!(ax[1], [smin, smax], color=:red)
    vlines!(ax[2], [smin, smax], color=:red)

    # xlims!(ax[1], 0, 1)
    # xlims!(ax[2], 0, 1)
    linkxaxes!(ax...)
    fig
end


function plot_sim(sol, df, ttrain; soc_shift=0.0)
    (; vμ, vσ, sμ, sσ) = sol
    fig = Figure()
    ax = [Axis(fig[i, j]) for i in 1:2, j in 1:2]

    # terminal voltage
    lines!(ax[1, 1], df.t / 3600, vμ)
    band!(ax[1, 1], df.t / 3600, vμ - 2vσ, vμ + 2vσ; alpha=0.5)
    lines!(ax[1, 1], df.t / 3600, df.v, color=(:black, 0.5), linestyle=:dash)

    ve = vμ - df.v
    lines!(ax[2, 1], df.t / 3600, ve)
    band!(ax[2, 1], df.t / 3600, ve - 2vσ, ve + 2vσ; alpha=0.5)

    # soc
    sμ´ = sμ .+ soc_shift
    lines!(ax[1, 2], df.t / 3600, sμ´)
    band!(ax[1, 2], df.t / 3600, sμ´ - 2sσ, sμ´ + 2sσ; alpha=0.5)
    lines!(ax[1, 2], df.t / 3600, df.s, color=(:black, 0.5), linestyle=:dash)

    se = sμ´ - df.s
    lines!(ax[2, 2], df.t / 3600, se)
    band!(ax[2, 2], df.t / 3600, se - 2sσ, se + 2sσ; alpha=0.5)

    for a in ax
        vlines!(a, [ttrain / 3600], color=:red)
    end

    ylims!(ax[2, 1], -0.05, 0.05) # TODO

    linkxaxes!(ax...)
    fig
end


function calc_ocv_mae(kf, df, zt, focv; soc_shift=0.0)
    smin, smax = extrema(df.s)
    s = smin:0.01:smax # TODO or range wit constant points?


    ocv = predict_gp(kf, s .- soc_shift, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
    ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)

    e = ocvμ - focv(s)

    mae_ocv = mean(abs, e) * 1e3 # mV
    max_ocv = maximum(abs, e) * 1e3 # mV

    mae_σ = mean(ocvσ) * 1e3
    max_σ = maximum(ocvσ) * 1e3


    (; mae_ocv, max_ocv, mae_σ, max_σ)
end

# for (u, y) in zip(us, ys)
#     LLPF.update!(kf, u, y)
# end

# sol = forward_trajectory(kf, us, ys)

# sol = run_sim!(kf, us, ys, ut)

# function predict(sol, kf, zt)
#     s´ = sol.xt .|> first
#     î = [u.î for u in us]
#     map(zip(s´, î)) do (s, i)
#         ocv = predict_gp(kf, [s], :ocv)
#         r0 = predict_gp(kf, [s], :r0)
#         μ̂ = ocv.μ[1] + i * r0.μ[1]
#         σ̂ = ocv.σ[1] + i^2 * r0.σ[1]
#         μ = StatsBase.reconstruct(zt.v, [μ̂]) |> first
#         σ = StatsBase.reconstruct(zt.σ, [σ̂]) |> first
#         (; μ, σ)
#     end |> DataFrame
# end

# function plot_prediction(sol, kf, zt, df)
#     fig = Figure()
#     ax = [Axis(fig[i, 1]) for i in 1:2]
#     pred = predict(sol, kf, zt)
#     # μ = [x.μ for x in pred]
#     # σ = [x.σ for x in pred]
#     (; μ, σ) = pred
#     lines!(ax[1], df.t / 3600, μ)
#     band!(ax[1], df.t / 3600, μ - 2σ, μ + 2σ; alpha=0.5)

#     e = μ - df.v
#     lines!(ax[2], df.t / 3600, e)
#     band!(ax[2], df.t / 3600, e - 2σ, e + 2σ; alpha=0.5)
#     fig
# end