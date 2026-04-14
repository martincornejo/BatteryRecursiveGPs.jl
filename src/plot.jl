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
    ylims!(ax[2], n * 0.0, n * 3.0)
    ax[1].yticks = (n * 3.2):(n * 0.3):(n * 4.2)
    ax[2].yticks = (0):(n):(n * 3)
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

    if sol.tt != length(sol.u)
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


function plot_module_soh(param_cells, param_modules)
    # cell soh
    cell_soh = [param[:soh] for (id, param) in param_cells] .|> Measurements.value

    # module soh
    module_ids = [(; p, m) for p in 1:3, m in 1:9] |> vec |> sort
    df_soh = map(module_ids) do id
        (; p, m) = id
        # module state from cell state
        ids_ = [(; p, m, c) for c in 1:12]
        param_cells_module = Dict(id => param_cells[id] for id in ids_)
        soh = calc_soh_pack(param_cells_module, 100)
        u = calc_Q_utilization(param_cells_module)

        # module state from module state
        soh_module = param_modules[id][:soh]

        (; p, m, soh, u, soh_module)
    end |> DataFrame


    # figure
    fig = Figure(size = (950, 360))
    gl1 = GridLayout(fig[1, 1])
    ax1 = Axis(gl1[1, 1])

    bins = 40:1:90
    hist!(ax1, cell_soh * 100; bins, color = Cycled(5), strokecolor = :gray, strokewidth = 1)

    ax1.xlabel = "Cell SOH (%)"
    ax1.ylabel = "Cell count"
    ylims!(ax1, 0, nothing)
    xlims!(ax1, first(bins), last(bins))

    ylims!(ax1, 0, 60)

    ax1.xgridvisible = false
    ax1.ygridvisible = false
    ax1.rightspinevisible = false
    ax1.topspinevisible = false
    ax1.ytrimspine = (false, true)

    gl2 = GridLayout(fig[1, 2])
    ax2 = [Axis(gl2[j, i]) for i in 1:3, j in 1:2]

    for p in 1:3
        df_ = subset(df_soh, :p => ByRow(==(p)))
        soh_μ = Measurements.value.(df_.soh) * 100
        soh_σ = Measurements.uncertainty.(df_.soh) * 100

        m_soh_μ = Measurements.value.(df_.soh_module) * 100

        u_μ = Measurements.value.(df_.u) * 100
        u_σ = Measurements.uncertainty.(df_.u) * 100

        ids = 1:9

        barplot!(ax2[p, 1], ids, m_soh_μ; color = Cycled(3))
        barplot!(ax2[p, 2], ids, soh_μ; color = Cycled(1))
    end

    for p in 1:3, j in 1:2
        if j == 1
            ax2[p, j].title = "Phase $p"
        end
        ylims!(ax2[p, 1], 40, 90)
        ylims!(ax2[p, 2], 40, 90)

        ax2[p, j].xticks = 1:9

        ax2[p, j].xgridvisible = false
        ax2[p, j].ygridvisible = false
        ax2[p, j].rightspinevisible = false
        ax2[p, j].topspinevisible = false
    end

    for j in 1:2
        hideydecorations!(ax2[2, j]; ticks = false)
        hideydecorations!(ax2[3, j]; ticks = false)
    end
    ax2[1, 1].ylabel = "Module-based\nmodule SOH (%)"
    ax2[1, 2].ylabel = "Cell-based\nmodule SOH (%)"

    for p in 1:3
        ax2[p, end].xlabel = "Module"
    end

    Label(
        fig[0, 1], "Cell SOH estimation",
        fontsize = 16, font = :bold, tellwidth = false
    )
    Label(
        fig[0, 2], "Module SOH estimation",
        fontsize = 16, font = :bold, tellwidth = false
    )
    colsize!(fig.layout, 1, Relative(0.3))
    return fig
end


function plot_cell_soh_hist(params)
    cell_soh = [param[:soh] for (id, param) in params] .|> Measurements.value

    fig = Figure(size = (450, 400))
    gl1 = GridLayout(fig[1, 1])
    ax1 = Axis(gl1[1, 1])

    bins = 40:1:90
    hist!(ax1, cell_soh * 100; bins, color = Cycled(5), strokecolor = :gray, strokewidth = 1)

    ax1.xlabel = "Cell SOH (%)"
    ax1.ylabel = "Cell count"
    ylims!(ax1, 0, nothing)
    xlims!(ax1, first(bins), last(bins))

    ylims!(ax1, 0, 60)

    ax1.xgridvisible = false
    ax1.ygridvisible = false
    ax1.rightspinevisible = false
    ax1.topspinevisible = false
    ax1.ytrimspine = (false, true)

    return fig
end


function plot_module_inhomogenity(params)
    fig = Figure(size = (800, 400))
    ax2 = [Axis(fig[j, i]) for i in 1:3, j in 1:3]

    module_ids = [(; p, m) for p in 1:3, m in 1:9] |> vec |> sort
    df_soh = map(module_ids) do id
        (; p, m) = id
        ids_ = [(; p, m, c) for c in 1:12]
        params_module = Dict(id => params[id] for id in ids_)
        soh = calc_soh_pack(params_module, 100)
        u1 = calc_Q_utilization(params_module)
        u2 = calc_Q_utilization(params_module; delta_soc = false)

        socs = [val[:soc] for val in values(params_module)]
        sohs = [val[:soh] for val in values(params_module)]

        Δsoc_max = maximum(socs) - minimum(socs)
        Δsoh_max = maximum(sohs) - minimum(sohs)

        soh_module = params2[id][:soh]

        (; p, m, soh, u1, u2, soh_module, Δsoc_max, Δsoh_max)
    end |> DataFrame

    for p in 1:3
        df_ = subset(df_soh, :p => ByRow(==(p)))
        soh_μ = Measurements.value.(df_.soh) * 100
        soh_σ = Measurements.uncertainty.(df_.soh) * 100

        m_soh_μ = Measurements.value.(df_.soh_module) * 100

        u1_μ = Measurements.value.(df_.u1) * 100
        u_σ = Measurements.uncertainty.(df_.u1) * 100

        u2_μ = Measurements.value.(df_.u2) * 100

        Δsoc = Measurements.value.(df_.Δsoc_max) * 100
        Δsoh = Measurements.value.(df_.Δsoh_max) * 100

        ids = 1:9

        barplot!(ax2[p, 1], ids, Δsoc; color = Cycled(8))
        barplot!(ax2[p, 2], ids, Δsoh; color = Cycled(3))
        barplot!(ax2[p, 3], ids, u2_μ; color = Cycled(2), label = "ΔSOH only")
        barplot!(ax2[p, 3], ids, u1_μ; color = Cycled(6), label = "ΔSOH and ΔSOC")
    end

    for p in 1:3, j in 1:3
        if j == 1
            ax2[p, j].title = "Phase $p"
        end
        ylims!(ax2[p, 1], 0, 10)
        ylims!(ax2[p, 2], 0, 40)
        ylims!(ax2[p, 3], 50, 100)

        ax2[p, j].xticks = 1:9

        ax2[p, j].xgridvisible = false
        ax2[p, j].ygridvisible = false
        ax2[p, j].rightspinevisible = false
        ax2[p, j].topspinevisible = false
    end

    for j in 1:2
        hideydecorations!(ax2[2, j]; ticks = false)
        hideydecorations!(ax2[3, j]; ticks = false)
    end
    ax2[1, 1].ylabel = "Max.\nΔSOC (%)"
    ax2[1, 2].ylabel = "Max.\nΔSOH (%)"
    ax2[1, 3].ylabel = "Module Q\nutilization (%)"

    for p in 1:3
        ax2[p, end].xlabel = "Module"
    end

    Legend(fig[4, :], ax2[1, 3]; merge = true, orientation = :horizontal, framevisible = false)

    colsize!(fig.layout, 1, Relative(0.3))
    return fig
end


function plot_q_trajectory(model::AbstractBatteryModel, sol)
    kf = model.kf
    (; zt) = kf.p

    q = StatsBase.reconstruct(zt.q, [u.q for u in sol.ut])
    t = 1:length(q)

    qμ = StatsBase.reconstruct(zt.q, sol.qμ)
    qσ = StatsBase.reconstruct(zt.q, sqrt.(sol.qσ))

    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]

    lines!(ax[1], t / 3600, q; color = Cycled(1), label = "Coulomb counting")
    lines!(ax[1], t / 3600, qμ; color = Cycled(2), label = "Estimated Q")
    band!(ax[1], t / 3600, qμ - 2qσ, qμ + 2qσ; color = Cycled(2), alpha = 0.5, label = "Estimated Q")

    lines!(ax[2], t / 3600, q - qμ; color = Cycled(2), label = "Estimated Q")
    band!(ax[2], t / 3600, (q - qμ) - 2qσ, (q - qμ) + 2qσ; color = Cycled(2), alpha = 0.5, label = "Estimated Q")

    xlims!(ax[1], t[begin] / 3600, t[end] / 3600)

    ax[1].ylabel = "Charge / Ah"
    ax[2].ylabel = "Error / Ah"
    ax[2].xlabel = "Time / h"

    Legend(fig[3, 1], ax[1]; merge = true, orientation = :horizontal)
    return fig
end


function plot_q_estimation(data, sol, model::AbstractBatteryModel)
    kf = model.kf
    (; zt) = kf.p

    q = StatsBase.reconstruct(zt.q, [u.q for u in sol.ut])
    t = 1:length(q)

    qμ = StatsBase.reconstruct(zt.q, sol.qμ)
    qσ = StatsBase.reconstruct(zt.q, sqrt.(sol.qσ))

    q_real = data.q
    t_real = data.t

    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]

    lines!(ax[1], t_real / 3600, q_real; color = :black, label = "Real Q")
    lines!(ax[1], t / 3600, q; color = :red, linestyle = :dash, label = "Coulomb counting")
    lines!(ax[1], t / 3600, qμ; color = Cycled(2), label = "Estimated Q")
    band!(ax[1], t / 3600, qμ - 2qσ, qμ + 2qσ; color = Cycled(2), alpha = 0.5, label = "Estimated Q")

    lines!(ax[2], t / 3600, qμ - q_real; color = Cycled(2), label = "Estimated Q")
    band!(ax[2], t / 3600, (qμ - q_real) - 2qσ, (qμ - q_real) + 2qσ; color = Cycled(2), alpha = 0.5, label = "Estimated Q")
    lines!(ax[2], t_real / 3600, q - q_real; color = :red, linestyle = :dash, label = "Coulomb counting")

    xlims!(ax[1], t[begin] / 3600, t[end] / 3600)

    ax[1].ylabel = "Charge / Ah"
    ax[2].ylabel = "Error / Ah"
    ax[2].xlabel = "Time / h"

    Legend(fig[3, 1], ax[1]; merge = true, orientation = :horizontal)
    return fig
end


function plot_soc_trajectories(models, sols, models2, sols2, id_module)
    fig = Figure(size = (540, 400))
    ax = Axis(fig[1, 1])

    colors = Makie.wong_colors()

    (; p, m) = id_module
    Ts = 1.0
    t = sols[(; p, m, c = 1)].idx * Ts
    df_soc = DataFrame(; t)

    for c in 1:12
        cell_id = (; p, m, c)
        sol = sols[cell_id]
        kf = models[cell_id].kf
        (; zt) = kf.p
        qμ = StatsBase.reconstruct(zt.q, sol.qμ)
        qσ = StatsBase.reconstruct(zt.q, sqrt.(sol.qσ))

        Q = params[cell_id][:Q] |> Measurements.value
        soc0 = params[cell_id][:soc] |> Measurements.value

        socμ = qμ / Q .+ soc0
        socσ = qσ / Q

        df_soc[!, "soc_cell_$c"] = socμ

        lines!(ax, t / 3600, socμ; color = (:gray, 0.2), label = "Cell SOC")
        band!(ax, t / 3600, socμ - socσ, socμ + socσ; color = (:gray, 0.1), label = "Cell SOC")
    end

    # soc
    soc_pack = calc_module_soc(df_soc, params, id_module)
    soc_pack_μ = soc_pack .|> Measurements.value
    soc_pack_σ = soc_pack .|> Measurements.uncertainty
    lines!(ax, t / 3600, soc_pack_μ; color = colors[1], label = "Module SOC (cell)")
    band!(ax, t / 3600, soc_pack_μ - 2soc_pack_σ, soc_pack_μ + 2soc_pack_σ; color = (colors[1], 0.5), label = "Module SOC (cell)")

    # module-level model
    sol2 = sols2[id_module]
    kf2 = models2[id_module].kf
    t2 = sol2.idx * Ts
    (; zt) = kf2.p
    qμ = StatsBase.reconstruct(zt.q, sol2.qμ)
    qσ = StatsBase.reconstruct(zt.q, sqrt.(sol2.qσ))

    Q = params2[id_module][:Q] |> Measurements.value
    soc0 = params2[id_module][:soc] |> Measurements.value

    soc_moudle_μ = qμ / Q .+ soc0
    soc_moudle_σ = qσ / Q

    lines!(ax, t2 / 3600, soc_moudle_μ; color = colors[3], label = "Module SOC (module)")
    band!(ax, t2 / 3600, soc_moudle_μ - 2soc_moudle_σ, soc_moudle_μ + 2soc_moudle_σ; color = (colors[3], 0.5), label = "Module SOC (module)")

    ylims!(ax, 0, 1)
    xlims!(ax, first(t) / 3600, last(t) / 3600)

    ax.ylabel = "SOC (p.u.)"
    ax.xlabel = "Time (h)"

    ax.title = "Phase $p, Module $m"

    Legend(fig[2, 1], ax; merge = true, orientation = :horizontal)

    return fig
end
