function plot_dataset(df)
    fig = Figure()
    colors = Makie.wong_colors()
    ax = [Axis(fig[i, 1]) for i in 1:4]
    for i in 1:12
        lines!(ax[1], df.t / 3600, df[:, "v_cell_$i"])
    end
    lines!(ax[2], df.t / 3600, df.v, color=colors[1])
    lines!(ax[3], df.t / 3600, df.i, color=colors[2])
    lines!(ax[4], df.t / 3600, df.q, color=colors[3])
    fig
end


function plot_ecm(kf, df, zt; focv=nothing, fR0=nothing, Q)
    fig = Figure(size=(600, 600))
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / mΩ"
    # ax[2].xlabel = "SOC / p.u."
    ax[2].xlabel = "ΔQ / Ah"
    hidexdecorations!(ax[1], ticks=false, grid=false)

    # soc = 0:0.01:1
    qmin, qmax = extrema(df.q)
    q = qmin:0.01:qmax
    q̂ = StatsBase.transform(zt.q, q)

    # OCV 
    ocv = predict_gp(kf, q̂, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
    ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)

    lines!(ax[1], q, ocvμ)
    band!(ax[1], q, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha=0.8)
    if focv !== nothing
        soc0 = df[begin, :s]
        soc = q / Q .+ soc0
        lines!(ax[1], q, focv(soc), color=:black, linestyle=:dot)
    end

    # R0
    r0 = predict_gp(kf, q̂, :r0)
    rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1e3
    rσ = StatsBase.reconstruct(zt.r, r0.σ) * 1e3

    lines!(ax[2], q, rμ)
    band!(ax[2], q, rμ + 2rσ, rμ - 2rσ, alpha=0.8)
    if fR0 !== nothing
        soc0 = df[begin, :s]
        soc = q / Q .+ soc0
        lines!(ax[2], soc, fR0.(soc), color=:black, linestyle=:dot)
    end

    # data - SOC window
    # smin, smax = df.q |> extrema
    # vlines!(ax[1], [smin, smax], color=:red)
    # vlines!(ax[2], [smin, smax], color=:red)

    # xlims!(ax[1], 0, 1)
    # xlims!(ax[2], 0, 1)
    linkxaxes!(ax...)
    fig
end

function plot_simulation(vμ, vσ, df)
    colors = Makie.wong_colors()

    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]

    lines!(ax[1], df.t / 3600, df.v, color=colors[1])
    lines!(ax[1], df.t / 3600, vμ, color=colors[2])
    band!(ax[1], df.t / 3600, vμ - 2vσ, vμ + 2vσ, color=(colors[2], 0.5))

    Δv = vμ - df.v
    lines!(ax[2], df.t / 3600, Δv * 1e3, color=colors[3])
    band!(ax[2], df.t / 3600, (Δv - 2vσ) * 1e3, (Δv + 2vσ) * 1e3, color=(colors[3], 0.5))

    ylims!(ax[2], -30, 30)

    ax[1].ylabel = "Voltage / V"
    ax[2].ylabel = "Error / mV"

    fig
end

function plot_soc_estimation(sol, time, s)
    μ = sol.xt .|> first
    σ = sol.Rt .|> first .|> sqrt

    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]
    lines!(ax[1], time / 3600, μ)
    band!(ax[1], time / 3600, μ - 2σ, μ + 2σ, alpha=0.5)
    lines!(ax[1], time / 3600, s)

    Δs = μ - s
    lines!(ax[2], time / 3600, Δs)
    band!(ax[2], time / 3600, Δs - 2σ, Δs + 2σ, alpha=0.5)

    fig
end