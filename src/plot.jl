function plot_ecm(model::AbstractBatteryModel, sol = nothing; n = 1)
    fig = Figure(size = (600, 600))
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / mΩ"
    ax[2].xlabel = "ΔQ / Ah"
    hidexdecorations!(ax[1], ticks = false, grid = false)

    plot_ecm!(ax, model, sol)

    ylims!(ax[1], 3.2 * n, 4.2 * n)
    ylims!(ax[2], 0.0 * n, 3.0 * n)
    linkxaxes!(ax...)
    return fig
end


function plot_ecms(models::AbstractDict, sols; n = 1)
    fig = Figure(size = (450, 400))
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV (V)"
    ax[2].ylabel = "R₀ (mΩ)"
    ax[2].xlabel = "Charge (Ah)"
    hidexdecorations!(ax[1], ticks = false, grid = false)

    for (id, model) in models
        plot_ecm!(ax, model, sols[id])
    end

    ylims!(ax[1], n * 3.2, n * 4.2)
    ylims!(ax[2], n * 0.0, n * 15)
    ax[1].yticks = (n * 3.2):(n * 0.3):(n * 4.2)
    ax[2].yticks = (0):(n * 5):(n * 15)
    linkxaxes!(ax...)
    return fig
end


function plot_sim(model::AbstractBatteryModel, sol; Ts = 1.0, plot_Δv = true)
    kf = model.kf
    zt = kf.p.zt
    (; idx, u, yt, yμ, yΣ) = sol

    μ = StatsBase.reconstruct(zt.v, first.(yμ))
    σ = StatsBase.reconstruct(zt.σ, sqrt.(first.(yΣ)))
    t = (0:(length(u) - 1)) * Ts / 3600 |> collect

    # plot
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]
    colors = Makie.wong_colors()

    # terminal voltage
    v = StatsBase.reconstruct(zt.v, first.(yt))
    lines!(ax[1], t[idx], v)
    lines!(ax[1], t[idx], μ, color = colors[2])
    band!(ax[1], t[idx], μ - 2σ, μ + 2σ, color = (colors[2], 0.5))

    # voltage error
    e = v - μ
    lines!(ax[2], t[idx], e * 1.0e3, color = colors[2])
    band!(ax[2], t[idx], (e - 2σ) * 1.0e3, (e + 2σ) * 1.0e3, color = (colors[2], 0.5))

    if 0 < sol.tt < length(sol.u)
        vlines!(ax[1], t[sol.tt]; color = :red)
        vlines!(ax[2], t[sol.tt]; color = :red)
    end

    xlims!.(ax, t[begin], t[end])

    ax[1].ylabel = "Voltage / V"
    ax[2].ylabel = "Voltage error / mV"
    ax[2].xlabel = "Time / h"

    linkxaxes!(ax...)

    return fig
end


function plot_q_estimation(q_ref, sol, model::AbstractBatteryModel)
    kf = model.kf
    (; zt) = kf.p

    t = sol.idx  # seconds since start of ti, at observation times only
    q = StatsBase.reconstruct(zt.q, [u.q for u in sol.ut])
    qμ = StatsBase.reconstruct(zt.q, sol.qμ)
    qσ = StatsBase.reconstruct(zt.q, sqrt.(sol.qσ))
    q_ref_t = q_ref[t]  # align reference to observation times

    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]

    lines!(ax[1], t / 3600, q_ref_t; color = :black, label = "Reference")
    lines!(ax[1], t / 3600, q; color = :red, linestyle = :dash, label = "Coulomb counting")
    lines!(ax[1], t / 3600, qμ; color = Cycled(2), label = "Estimated Q")
    band!(ax[1], t / 3600, qμ - 2qσ, qμ + 2qσ; color = Cycled(2), alpha = 0.5, label = "Estimated Q")

    lines!(ax[2], t / 3600, qμ - q_ref_t; color = Cycled(2), label = "Estimated Q")
    band!(ax[2], t / 3600, (qμ - q_ref_t) - 2qσ, (qμ - q_ref_t) + 2qσ; color = Cycled(2), alpha = 0.5, label = "Estimated Q")
    lines!(ax[2], t / 3600, q - q_ref_t; color = :red, linestyle = :dash, label = "Coulomb counting")

    xlims!(ax[1], t[begin] / 3600, t[end] / 3600)

    ax[1].ylabel = "Charge / Ah"
    ax[2].ylabel = "Error / Ah"
    ax[2].xlabel = "Time / h"

    Legend(fig[3, 1], ax[1]; merge = true, orientation = :horizontal)
    return fig
end
