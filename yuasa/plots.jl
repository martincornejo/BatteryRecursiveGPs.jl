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

    for i in 1:4
        xlims!(ax[i], df[begin, :t] / 3600, df[end, :t] / 3600)
    end
    for i in 1:3
        hidexdecorations!(ax[i], ticks=false, grid=false)
    end

    ax[1].ylabel = "Cell / V"
    ax[2].ylabel = "Module / V"
    ax[3].ylabel = "Current / A"
    ax[4].ylabel = "Coloumb / Ah"
    ax[4].xlabel = "Time / h"

    fig
end

function plot_cell_states(params)
    fig = Figure(size=(500, 500))
    ax = [Axis(fig[i, 1]) for i in 1:2]
    colors = Makie.wong_colors()

    Q = [params[Symbol("cell_$i")][:Q] for i in 1:12]
    barplot!(ax[1], 1:12, Q)
    ylims!(ax[1], 55, 70)
    ax[1].xticks = 1:12
    ax[1].ylabel = "Capacity / Ah"
    ax[1].xlabel = "Cell ID"

    s = [params[Symbol("cell_$i")][:soc] for i in 1:12]
    barplot!(ax[2], 1:12, s; color=colors[3])
    ylims!(ax[2], 0.45, 0.60)
    ax[2].xticks = 1:12
    ax[2].ylabel = "Initial SOC / p.u."
    ax[2].xlabel = "Cell ID"

    fig
end

function plot_ecm(kf, df, zt; focv=nothing, fR0=nothing, Q, external_cc=true)
    fig = Figure(size=(600, 600))
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / mΩ"
    # ax[2].xlabel = "SOC / p.u."
    ax[2].xlabel = "ΔQ / Ah"
    hidexdecorations!(ax[1], ticks=false, grid=false)

    # soc = 0:0.01:1
    if external_cc
        qmin, qmax = extrema(df.q)
        q = qmin:0.01:qmax
        q̂ = StatsBase.transform(zt.q, q)
    else
        qmin, qmax = extrema(cumsum(df.i) * kf.p.Ts / 3600)
        q = qmin:0.01:qmax
        q̂ = StatsBase.transform(zt.i, q)
    end

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

    lines!(ax[1], df.t / 3600, vμ, color=colors[1], label="Model")
    band!(ax[1], df.t / 3600, vμ - 2vσ, vμ + 2vσ, color=(colors[1], 0.5), label="Model")
    lines!(ax[1], df.t / 3600, df.v, color=colors[2], label="Real")

    Δv = vμ - df.v
    lines!(ax[2], df.t / 3600, Δv * 1e3, color=colors[1])
    band!(ax[2], df.t / 3600, (Δv - 2vσ) * 1e3, (Δv + 2vσ) * 1e3, color=(colors[1], 0.5))

    hidexdecorations!(ax[1], ticks=false, grid=false)
    xlims!(ax[1], df[begin, :t] / 3600, df[end, :t] / 3600)
    xlims!(ax[2], df[begin, :t] / 3600, df[end, :t] / 3600)
    # ylims!(ax[2], -30, 30)
    ylims!(ax[2], -20, 20)

    ax[1].ylabel = "Voltage / V"
    ax[2].ylabel = "Error / mV"

    Legend(fig[3, 1], ax[1], merge=true, orientation=:horizontal)

    fig
end

function plot_soc_estimation(time, μ, σ, s, s´=nothing)
    colors = Makie.wong_colors()

    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]
    lines!(ax[1], time / 3600, μ; label="Model")
    band!(ax[1], time / 3600, μ - 2σ, μ + 2σ, alpha=0.5; label="Model")
    lines!(ax[1], time / 3600, s; label="Real")

    if s´ !== nothing
        lines!(ax[1], time / 3600, s´; color=:gray, linestyle=:dash, label="Coloumb")
    end

    Δs = μ - s
    lines!(ax[2], time / 3600, Δs; color=colors[1])
    band!(ax[2], time / 3600, Δs - 2σ, Δs + 2σ; color=(colors[1], 0.5))

    hidexdecorations!(ax[1], ticks=false, grid=false)
    ax[1].ylabel = "SOC / p.u."
    ax[2].ylabel = "Error / p.u."
    ax[2].xlabel = "Time / h"

    xlims!(ax[1], time[begin] / 3600, time[end] / 3600)
    xlims!(ax[2], time[begin] / 3600, time[end] / 3600)
    ylims!(ax[2], -0.025, 0.025)

    # axislegend(ax[1], position=:lb, merge=true)
    Legend(fig[3, 1], ax[1], merge=true, orientation=:horizontal)

    fig
end


function plot_q_estimation(evo, df_cell, zt)
    qμ = StatsBase.reconstruct(zt.i, [μ.cc.q for μ in evo.μs])
    qσ = StatsBase.reconstruct(zt.i, [sqrt.(Σ[:cc, :cc][:q, :q]) for Σ in evo.Σs])

    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]
    lines!(ax[1], df_cell.t / 3600, qμ, color=:blue)
    band!(ax[1], df_cell.t / 3600, qμ - 2qσ, qμ + 2qσ, color=(:blue, 0.5))
    lines!(ax[1], df_cell.t / 3600, df_cell.q, color=:orange)

    ax[1].ylabel = "Evolution"

    error = qμ - df_cell.q
    lines!(ax[2], df_cell.t / 3600, error, color=:red)
    ax[1].ylabel = "Difference with original"
    fig
end



function plot_simulation_cell(df, res)
    v = zero.(res[:cell_1][:v])
    for i in 1:12
        v .+= res[Symbol("cell_$i")][:v]
    end

    vμ = Measurements.value.(v)
    vσ = Measurements.uncertainty.(v)

    plot_simulation(vμ, vσ, df)
end

function plot_soc_estimation_cell(df, res, params)
    dfs = DataFrame(
        :t => df.t,
        [Symbol("soc_$i") => res[i][:s] for i in collect(keys(res))]...
    )
    s´ = calc_module_soc(dfs, res)

    sμ = Measurements.value.(s´)
    sσ = Measurements.uncertainty.(s´)

    Q´ = calc_Q_pack(res)

    ŝ = sμ[begin] .+ cumsum(df.i .+ 0.5) * Ts / (Q´ * 3600)

    soc_module = calc_module_soc(df, params)
    plot_soc_estimation(df.t, sμ, sσ, soc_module, ŝ)
end
