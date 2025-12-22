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

function plot_ecm(kf, df, zt; focv=nothing, fR0=nothing, Q, external_cc=true, title = nothing)
    fig = Figure(size=(600, 600))
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / mΩ"
    # ax[2].xlabel = "SOC / p.u."
    ax[2].xlabel = "ΔQ / Ah"
    hidexdecorations!(ax[1], ticks=false, grid=false)

    # soc = 0:0.01:1
    if external_cc
        #qmin, qmax = extrema(df.q)
        soc0 = df[begin, :s]
        qmin = (0 .- soc0) .* Q
        qmax = (1 .- soc0) .* Q
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
        rmse_ocv = sqrt(mean((ocvμ .- focv.(soc)) .^ 2))
        text!(
            ax[1],
            median(q), 4.1,                # position in data coordinates (x=0.5, y=0.1)
            text="RMSE iod = $(round(rmse_ocv * 1e3, digits = 3)) mV",
            align=(:center, :center),
            color=:red
        )
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
        lines!(ax[2], q, fR0.(soc)* 1e3, color=:black, linestyle=:dot)
        rmse_r0 = sqrt(mean((rμ .- fR0.(soc)* 1e3) .^ 2))
        text!(
            ax[2],
            median(q), 1.6,                # position in data coordinates (x=0.5, y=0.1)
            text="RMSE iod = $(round(rmse_r0, digits = 3)) mΩ ",
            align=(:center, :center),
            color=:red
        )
    end

    if title !== nothing
        ax[1].title = title
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


function plot_rc_evo(kf, evoμ, evoΣ, zt; name=:rc, τ0=[60], R0=[0.8], title = nothing)
    fig = Figure()
    ax = [CairoMakie.Axis(fig[i, 1]) for i in 1:2]
    (; p) = kf.kf

    r = 1e3 * zt.σ.scale[1] / zt.i.scale[1]
    τ = Float64[]
    τσ = Float64[]

    R = Float64[]
    Rσ = Float64[]

    for i in 1:1:length(evoμ)
        x_kf = ComponentVector(evoμ[i], p.xid)
        Σ_kf = ComponentMatrix(evoΣ[i], p.Σid)

        x_rc = x_kf[name]
        Σ_rc = Σ_kf[name]
        if i%50 == 0
            push!(τ, exp(x_rc[:τ]))
            push!(τσ, sqrt.(Σ_rc[:τ, :τ]))

            push!(R, exp(x_rc.r) * r)
            push!(Rσ, sqrt.(Σ_rc[:r, :r]) * r)
        end
    end

    x_axis = collect(1:1:length(τ))
    scatter!(ax[1], x_axis, τ, color=:blue, label="Predicted")
    hlines!(ax[1], τ0, color=:green, label="Ground truth")
    errorbars!(ax[1], x_axis, τ, τσ, color=:black, label="Deviation")

    scatter!(ax[2], x_axis, R, color=:blue, label="Predicted")
    hlines!(ax[2], R0, color=:green, label="Ground truth")
    errorbars!(ax[2], x_axis, R, Rσ, color=:black, label="Deviation")

    Legend(fig[1, 2], ax[1];
        labelsize=8,        # shrink text
        markersize=8,       # shrink markers
        padding=2,          # tighter box
        framevisible=false  # optional: remove box
    )

    Legend(fig[2, 2], ax[2];
        labelsize=8,        # shrink text
        markersize=8,       # shrink markers
        padding=2,          # tighter box
        framevisible=false  # optional: remove box
    )
    ylims!(ax[1], 30,100)
    ylims!(ax[2], 0.5,1.2)

    x_rc = ComponentVector(kf.x, kf.p.xid).rc
    text!(
        ax[1],
        median(x_axis), 80,                # position in data coordinates (x=0.5, y=0.1)
        text="Abs error = $(abs.(round(exp(x_rc.τ) .- τ0[1], digits = 3)))",
        align=(:center, :center),
        color=:red
    )
    r = StatsBase.reconstruct(zt.r,[exp(x_rc.r)])[1] * 1e3
    text!(
        ax[2],
        median(x_axis), 1.0,                # position in data coordinates (x=0.5, y=0.1)
        text="Abs error = $(abs.(round(r .- R0[1], digits = 3))) mΩ ",
        align=(:center, :center),
        color=:red
    )
    ax[1].xlabel = "Time in h"
    ax[1].ylabel = "τ"
    ax[2].xlabel = "Time in h"
    ax[2].ylabel = "R"

    if title !== nothing
        ax[1].title = title
    end
    fig
end


function plot_simulation(vμ, vσ, df; title = nothing)
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
    ylims!(ax[1], 3.6,4.1)

    rmse = sqrt(mean((Δv* 1e3).^2)) 
    text!(
        ax[2],
        median(df.t / 3600), 10.0,                # position in data coordinates (x=0.5, y=0.1)
        text="RMSE error = $(round(rmse, digits=2)) mV ",
        align=(:center, :center),
        color=:red
    )

    if title !== nothing
        ax[1].title = title
    end
    fig
end

function plot_soc_estimation(time, μ, σ, s, s´=nothing; title = nothing)
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
    
    if title !== nothing
        ax[1].title = title
    end
    fig
end


function plot_q_estimation(evoμ, evoΣ, df_cell, zt; title = nothing)
    qμ = StatsBase.reconstruct(zt.i, [μ.cc.q for μ in evoμ])
    qσ = StatsBase.reconstruct(zt.i, [sqrt.(Σ[:cc, :cc][:q, :q]) for Σ in evoΣ])

    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]
    lines!(ax[1], df_cell.t / 3600, qμ, color=:blue, label = "aprox")
    band!(ax[1], df_cell.t / 3600, qμ - 2qσ, qμ + 2qσ, color=(:blue, 0.5))
    lines!(ax[1], df_cell.t / 3600, df_cell.q, color=:orange, label = "true")

    ax[1].ylabel = "Evolution"

    error = qμ - df_cell.q
    lines!(ax[2], df_cell.t / 3600, error, color=:red)
    ax[1].ylabel = "Error"
    axislegend(ax[1]; position = :rt)
    if title !== nothing
        ax[1].title = title
    end
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
