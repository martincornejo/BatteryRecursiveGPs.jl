# === dataset figures ===

function plot_cell_voltage_system(
        data; panel_tags = true,
        # highlight the outlier modules discussed in the text
        highlights = Dict((3, 5) => :firebrick, (1, 6) => :firebrick),
    )
    df = copy(data[:cell_voltage])
    fig = Figure(size = (700, 600))
    ax = [Axis(fig[i, j]) for i in 1:9, j in 1:3]

    t0 = first(df._time)
    df[!, :t] = Dates.value.(df._time .- t0) * 1.0e-3 # seconds
    phases = 1:3
    modules = 1:9

    for p in phases, m in modules
        for i in 1:12
            lines!(ax[m, p], df.t / 3600, df[:, "cell_voltage_$(p)_$(m)_1_$(i)"])
        end
    end

    for i in modules, j in phases[2:end]
        hideydecorations!(ax[i, j], ticks = false)
    end
    for i in modules[1:(end - 1)], j in phases
        hidexdecorations!(ax[i, j], ticks = false)
    end

    for ((p, m), color) in highlights
        for spine in (:topspinecolor, :bottomspinecolor, :leftspinecolor, :rightspinecolor)
            setproperty!(ax[m, p], spine, color)
        end
        ax[m, p].spinewidth = 2
    end

    for p in phases, m in modules
        hlines!(ax[m, p], [3.4, 4.07]; color = (:black, 0.3), linestyle = :dash, linewidth = 0.8)
    end

    for i in modules, j in phases
        ylims!(ax[i, j], 3.35, 4.15)
        xlims!(ax[i, j], first(df.t) / 3600, last(df.t) / 3600)
        ax[i, j].xticks = 0:4:12
        ax[i, j].yticks = [3.5, 4.0]
        ax[i, j].xminorticks = IntervalsBetween(5)
        ax[i, j].xminorticksvisible = true
        ax[i, j].yminorticks = IntervalsBetween(2)
        ax[i, j].yminorticksvisible = true
        ax[i, j].xgridvisible = false
        ax[i, j].ygridvisible = false
    end

    if panel_tags
        for p in phases, m in modules
            text!(
                ax[m, p], 0.97, 0.06; text = "P$(p)M$(m)",
                space = :relative, align = (:right, :bottom), font = :bold, fontsize = 12,
                color = get(highlights, (p, m), :gray40),  # tag matches the frame color
            )
        end
    else
        for i in 1:9
            Label(fig[i, 4], "Module $i", font = :bold, fontsize = 11, rotation = pi / 2, tellheight = false)
        end
        for j in 1:3
            Label(fig[0, j], "Phase $j", font = :bold, fontsize = 11, tellwidth = false)
        end
    end
    for j in 1:3
        ax[end, j].xlabel = "Time / h"
    end

    ax[5, 1].ylabel = "Cell voltages / V"

    rowgap!(fig.layout, 2.5)
    colgap!(fig.layout, 2.5)
    return fig
end

# Per-signal sampling-interval histograms. All tables share a single dense `_time`
# column, so the sampling intervals are a property of each signal table.
function plot_data_resolution(data; yscale = log10)
    colors = Makie.wong_colors()
    # signal → color mapping matches plot_dataset_overview
    signals = [
        (:module_current, "Module current", colors[2]),
        (:module_voltage, "Module voltage", colors[3]),
        (:battery_temperature, "Module temperature", colors[4]),
        (:cell_voltage, "Cell voltage", colors[1]),
    ]
    logscale = yscale === log10

    fig = Figure(size = (550, 500))
    ax = [
        Axis(fig[i, 1]; yscale, ylabel = "Count", titlealign = :left, titlesize = 12,
            xticks = 0:10:80, xminorticks = IntervalsBetween(10), xminorticksvisible = true)
            for i in 1:4
    ]
    bins = 0.5:1:80.5  # integer-second timestamps, center bars on integers

    nmax = 0
    for (i, (key, name, color)) in enumerate(signals)
        Δt = Dates.value.(diff(data[key][!, "_time"])) * 1.0e-3
        hist!(ax[i], Δt; strokewidth = 1, strokecolor = :black, color, bins,
            fillto = logscale ? 0.5 : 0.0)
        nmax = max(nmax, maximum(StatsBase.fit(Histogram, Δt, bins).weights))
        ax[i].title = "$name  (median $(round(Int, median(Δt))) s)"
    end

    linkaxes!(ax...)
    xlims!(ax[1], 0, 81)
    logscale && ylims!(ax[1], 0.5, 2nmax)
    foreach(a -> hidexdecorations!(a; grid = false, ticks = false, minorticks = false), ax[1:3])
    ax[4].xlabel = "Sampling interval / s"

    return fig
end

function plot_dataset_overview(data; id_norm = (3, 7), id_out = (3, 5))
    fig = Figure(size = (700, 450))
    colors = Makie.wong_colors()

    df_v = copy(data[:cell_voltage])
    t0 = first(df_v._time)
    df_v[!, :t] = Dates.value.(df_v._time .- t0) * 1.0e-3 / 3600 # time in hours

    df_dr = coalesce.(data[:derating_current], NaN)
    t_dr = Dates.value.(df_dr._time .- t0) * 1.0e-3 / 3600

    axs = [Axis(fig[i, j]) for i in 1:4, j in 1:2]
    for (j, (p, m)) in enumerate((id_norm, id_out))
        for i in 1:12
            lines!(axs[1, j], df_v.t, df_v[:, "cell_voltage_$(p)_$(m)_1_$(i)"])
        end

        df_V = select(data[:module_voltage], "_time" => "time", "module_voltage_$(p)_$(m)" => "value")
        df_i = select(data[:module_current], "_time" => "time", "module_average_current_$(p)_$(m)" => ByRow(x -> -x) => "value")
        df_T = select(data[:battery_temperature], "_time" => "time", "battery_sensor_temperature_$(p)_$(m)_1" => "value")
        N = 5
        for (k, df) in enumerate((df_V, df_i, df_T))
            df[!, :t] = Dates.value.(df.time .- t0) * 1.0e-3 / 3600
            lines!(axs[1 + k, j], df.t[1:N:end], df.value[1:N:end]; color = colors[[3, 2, 4][k]])
        end
        # BMS limits: cell/module voltage window and derating current envelope
        hlines!(axs[1, j], [3.4, 4.07]; color = (:black, 0.6), linestyle = :dash)
        hlines!(axs[2, j], 12 .* [3.4, 4.07]; color = (:black, 0.6), linestyle = :dash)
        lines!(axs[3, j], t_dr, df_dr[:, "dr_ch_p$(p)_m$(m)"]; color = (:black, 0.6), linestyle = :dash)
        lines!(axs[3, j], t_dr, -df_dr[:, "dr_dch_p$(p)_m$(m)"]; color = (:black, 0.6), linestyle = :dash)

        axs[1, j].title = "Phase $p, Module $m" * (j == 2 ? " (outlier)" : "")
        axs[4, j].xlabel = "Time / h"
        for k in 1:4
            xlims!(axs[k, j], first(df_v.t), last(df_v.t))
            axs[k, j].xticks = 0:4:12
        end
    end

    # same voltage range and 0.25 V/cell tick lattice as the ECM comparison figure
    ylims!.(axs[1, :], 3.35, 4.15)
    ylims!.(axs[2, :], 12 * 3.35, 12 * 4.15)
    ylims!.(axs[3, :], -110, 110)  # full derating range: no-limit level is ±100 A
    ylims!.(axs[4, :], 18, 33)
    for j in 1:2
        axs[1, j].yticks = 3.5:0.25:4.0
        axs[2, j].yticks = 42:3:48
        axs[3, j].yticks = -100:50:100
        axs[4, j].yticks = 20:5:30
    end
    axs[1, 1].ylabel = "Cell\nvoltages / V"
    axs[2, 1].ylabel = "Module\nvoltage / V"
    axs[3, 1].ylabel = "Module\ncurrent / A"
    axs[4, 1].ylabel = "Module\ntemp. / °C"

    for i in 1:4
        hideydecorations!(axs[i, 2], ticks = false, grid = false)
        i < 4 && hidexdecorations!.(axs[i, :], ticks = false, grid = false)
        for j in 1:2
            axs[i, j].xgridvisible = false
            axs[i, j].ygridvisible = false
            axs[i, j].xminorticks = IntervalsBetween(5)
            axs[i, j].xminorticksvisible = true
            axs[i, j].yminorticks = IntervalsBetween(2)
            axs[i, j].yminorticksvisible = true
        end
    end
    rowgap!(fig.layout, 8)
    colgap!(fig.layout, 10)
    return fig
end


# === plotters migrated out of the package (BatteryRecursiveGPs no longer ships plotting) ===

# terminal-voltage fit + innovation over time
function plot_sim(model::AbstractBatteryModel, sol; Ts = 1.0)
    kf = model.kf
    zt = kf.p.zt
    (; idx, u, yt, yμ, yΣ) = sol

    μ = StatsBase.reconstruct(zt.v, first.(yμ))
    σ = StatsBase.reconstruct(zt.σ, sqrt.(first.(yΣ)))
    t = (0:(length(u) - 1)) * Ts / 3600 |> collect

    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]
    colors = Makie.wong_colors()

    v = StatsBase.reconstruct(zt.v, first.(yt))
    lines!(ax[1], t[idx], v)
    lines!(ax[1], t[idx], μ, color = colors[2])
    band!(ax[1], t[idx], μ - 2σ, μ + 2σ, color = (colors[2], 0.5))

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

# charge estimate vs reference (Coulomb counting + filtered Q) with error panel
function plot_q_estimation(q_ref, sol, model::AbstractBatteryModel)
    kf = model.kf
    (; zt) = kf.p

    t = sol.idx  # observation times only
    q = StatsBase.reconstruct(zt.q, [u.q for u in sol.ut])
    qμ = StatsBase.reconstruct(zt.q, sol.qμ)
    qσ = StatsBase.reconstruct(zt.q, sqrt.(sol.qσ))
    q_ref_t = q_ref[t]

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

# Animate the RCGP ECM (OCV + R0/R1) as the filter learns over the dataset, next to the measured
# voltage with a moving cursor. Adapted from the package's YuasaModel animate_model.
# NOTE: pass the FULL sol from run_kf! BEFORE reduce_sol (needs per-timestep xt/Rt), e.g.
#   sol = run_kf!(cell_models[id], cell_sols[id].u, cell_sols[id].y)
function animate_model(file, model::RCGPModel, sol; step = 60)
    kf = model.kf
    zt = kf.p.zt
    (; xt, Rt, yt) = sol

    q̂ = collect(range(extrema(kf.p.r1.b0)...; step = 0.01))
    q = StatsBase.reconstruct(zt.q, q̂)

    # ECM curves at a given filter state (x, R)
    ecm(x, R) = begin
        ocv = predict_gp(kf, q̂, x, R, :ocv)
        ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
        ocvσ = StatsBase.reconstruct(zt.σ, sqrt.(diag(ocv.Σ)))
        xc = ComponentVector(x, kf.p.xid)
        r0μ = StatsBase.reconstruct(zt.r, [abs(xc.r0.r)]) |> first
        r1 = predict_gp(kf, q̂, x, R, :r1)
        r1μ = StatsBase.reconstruct(zt.r, r1.μ) * 1.0e3
        r1σ = StatsBase.reconstruct(zt.r, sqrt.(diag(r1.Σ))) * 1.0e3
        (; ocvμ, ocvσ, rμ = r0μ .+ r1μ, rσ = r1σ)
    end

    fig = Figure(size = (900, 400))
    axo = Axis(fig[1, 2]; ylabel = "OCV / V")
    axr = Axis(fig[2, 2]; ylabel = "R / mΩ", xlabel = "Charge / Ah")
    axv = Axis(fig[1:2, 1]; ylabel = "Voltage / V", xlabel = "Step")
    linkxaxes!(axo, axr)
    hidexdecorations!(axo; ticks = false, grid = false)

    (; ocvμ, ocvσ, rμ, rσ) = ecm(xt[1], Rt[1])
    lo = lines!(axo, q, ocvμ)
    bo = band!(axo, q, ocvμ + 2ocvσ, ocvμ - 2ocvσ; alpha = 0.6)
    lr = lines!(axr, q, rμ)
    br = band!(axr, q, rμ + 2rσ, rμ - 2rσ; alpha = 0.6)

    v = StatsBase.reconstruct(zt.v, first.(yt))
    lines!(axv, v; color = :gray)
    cursor = vlines!(axv, 1; color = :red)

    return record(fig, file, 1:step:length(xt)) do i
        (; ocvμ, ocvσ, rμ, rσ) = ecm(xt[i], Rt[i])
        Makie.update!(lo, arg2 = ocvμ)
        Makie.update!(bo, arg2 = ocvμ + 2ocvσ, arg3 = ocvμ - 2ocvσ)
        Makie.update!(lr, arg2 = rμ)
        Makie.update!(br, arg2 = rμ + 2rσ, arg3 = rμ - 2rσ)
        Makie.update!(cursor, arg1 = i)
    end
end


# ECM curves for one RCGP cell/module onto a 2-row axis (OCV top, R0+R1(q) bottom).
# Helper for plot_ecms_comparison below.
function plot_ecm!(ax, model::RCGPModel, sol; color = nothing)
    kw = isnothing(color) ? (;) : (; color)
    kf = model.kf
    zt = kf.p.zt

    q̂ = collect(range(extrema(sol.qμ)...; step = 0.01))
    q = StatsBase.reconstruct(zt.q, q̂)

    # OCV
    ocv = predict_gp(kf, q̂, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
    ocvσ = StatsBase.reconstruct(zt.σ, sqrt.(diag(ocv.Σ)))
    lines!(ax[1], q, ocvμ; kw...)
    band!(ax[1], q, ocvμ + 2ocvσ, ocvμ - 2ocvσ; alpha = 0.8, kw...)

    # R0 scalar from the final state estimate
    xc = ComponentVector(sol.x_end, kf.p.xid)
    Σ = ComponentMatrix(sol.R_end, kf.p.Σid)
    r0μ = StatsBase.reconstruct(zt.r, [abs(xc.r0.r)]) |> first
    r0σ = StatsBase.reconstruct(zt.r, [sqrt(Σ[:r0, :r0][:r, :r])]) |> first

    # R1 GP curve overlaid on the R0 panel
    r1 = predict_gp(kf, q̂, :r1)
    r1μ = StatsBase.reconstruct(zt.r, r1.μ) * 1.0e3
    r1σ = StatsBase.reconstruct(zt.r, sqrt.(diag(r1.Σ))) * 1.0e3
    rμ = r0μ .+ r1μ
    rσ = r0σ .+ r1σ
    lines!(ax[2], q, rμ; kw...)
    band!(ax[2], q, rμ + 2rσ, rμ - 2rσ; alpha = 0.8, kw...)
    return nothing
end

function plot_ecms_comparison(
        cell_models, cell_sols, module_models, module_sols;
        n_cell = 1, n_mod = 12, tags = true,
        # color = module ID M1–M9 (same across phases);
        # Wong palette extended to 9 with black + gray
        colors = vcat(Makie.wong_colors(), [RGBAf(0, 0, 0, 1), RGBAf(0.6, 0.6, 0.6, 1)]),
    )
    fig = Figure(size = (700, 500))
    gl1 = GridLayout(fig[1, 1])
    gl2 = GridLayout(fig[1, 2])

    scenarios = [
        (; title = "Cell level", gl = gl1, models = cell_models, sols = cell_sols, n = n_cell, tags = ("A", "C")),
        (; title = "Module level", gl = gl2, models = module_models, sols = module_sols, n = n_mod, tags = ("B", "D")),
    ]

    for args in scenarios
        (; title, gl, models, sols, n) = args
        ax = [Axis(gl[i, 1]) for i in 1:2]
        if tags
            for i in 1:2  # tags inside the axes: no layout space needed
                text!(ax[i], 0.02, 0.98; text = args.tags[i], space = :relative, align = (:left, :top), font = :bold, fontsize = 20)
            end
        end
        ax[1].title = title
        ax[1].ylabel = "OCV / V"
        ax[2].ylabel = rich("R", subscript("DC"), " / mΩ")
        ax[2].xlabel = "Charge / Ah"
        ax[1].xgridvisible = false
        ax[1].ygridvisible = false
        ax[2].xgridvisible = false
        ax[2].ygridvisible = false
        hidexdecorations!(ax[1], ticks = false)

        for (id, model) in models
            plot_ecm!(ax, model, sols[id]; color = colors[id.m])
        end

        ylims!(ax[1], n * 3.35, n * 4.15)
        ylims!(ax[2], n * 0.0, n * 15)
        # 0.25 V/cell steps so module ticks (×12) land on integers (42, 45, 48)
        ax[1].yticks = (n * 3.5):(n * 0.25):(n * 4.0)
        ax[2].yticks = (0):(n * 5):(n * 15)
        linkxaxes!(ax...)
    end

    mod_elems = [LineElement(color = colors[m], linewidth = 3) for m in 1:9]
    Legend(
        fig[2, 1:2], mod_elems, ["M$m" for m in 1:9], "Module ID";
        orientation = :horizontal, titleposition = :left, framevisible = false
    )

    return fig
end


function plot_soh_heatmap(comp_fit)
    fig = Figure(size = (300, 400))
    gl = GridLayout(fig[1, 1])
    plot_soh_heatmap!(gl, comp_fit)
    return fig
end

function plot_soh_heatmap!(layout, comp_fit)
    ax = Axis(layout[1, 1], xlabel = "Cell ID", ylabel = "Module ID")
    ax.xticks = [3, 6, 9, 12]
    ax.xminorticks = 1:12
    ax.xminorticksvisible = true

    ax.ytickformat = values -> begin
        map(values) do value
            p, m = divrem(value, 9)
            "P$(Int(p) + 1)M$(Int(m))"
        end
    end
    ax.yticks = [1, 6, 10, 15, 19, 24] #27
    ax.yminorticks = 1:27
    ax.yminorticksvisible = true

    colormap = :lipari
    colorrange = (65, 87)
    lowclip = :black
    soh = reshape(Measurements.value.(comp_fit.Q_cell), 12, 27)
    heatmap!(ax, soh; colorrange, colormap, lowclip)
    Colorbar(layout[1, 2]; limits = colorrange, colormap, lowclip, label = "Cell Capacity / Ah")
    return nothing
end

function module_id_xticks!(ax)
    ax.xtickformat = values -> begin
        map(values) do value
            p, m = divrem(Int(value), 9)
            "P$(p + 1)M$m"
        end
    end
    ax.xticks = [1, 6, 10, 15, 19, 24]
    ax.xminorticks = 1:27
    ax.xminorticksvisible = true
    return
end

function plot_composite_ocv(comp_fit, cells; xaxis = :soc)
    fig = Figure(size = (400, 400))
    gl = GridLayout(fig[1, 1])
    plot_composite_ocv!(gl, comp_fit, cells; xaxis)
    return fig
end


function plot_composite_ocv!(layout, comp_fit, cells; xaxis = :soc, vertical = true)
    (; soc_grid, v_grid, Q_cell, s0, Q_full) = comp_fit
    xscale = xaxis == :ah ? Q_full : 1.0
    xlabel = xaxis == :ah ? "Capacity / Ah" : "SOC"

    if vertical
        ax1 = Axis(layout[1, 1]; ylabel = "Voltage / V")
        ax2 = Axis(layout[2, 1]; ylabel = "dV/d$(xlabel)", xlabel)
        hidexdecorations!(ax1; grid = false, ticks = false)
        linkxaxes!(ax1, ax2)
        rowgap!(layout, 1, 5)
    else
        ax1 = Axis(layout[1, 1]; ylabel = "Voltage / V", xlabel)
        ax2 = Axis(layout[1, 2]; ylabel = "dV/d$(xlabel)", xlabel)
    end

    Q_cell_μ = Measurements.value.(Q_cell)
    s0_μ = Measurements.value.(s0)
    colormap = :lipari
    colorrange = (65, 87)
    lowclip = :black
    for i in eachindex(cells)
        q = cells[i].q
        v = cells[i].μ
        x = (q ./ Q_cell_μ[i] .+ s0_μ[i]) .* xscale
        lines!(ax1, x, v; color = Q_cell_μ[i], colorrange, colormap, alpha = 0.6, label = "Cell OCV")
        lines!(ax2, x[2:end], diff(v) ./ diff(x); color = Q_cell_μ[i], colorrange, colormap, alpha = 0.4)
    end

    order = sortperm(soc_grid)
    itp = PCHIPInterpolation(v_grid[order], soc_grid[order])
    soc_dense = range(soc_grid[order[1]], soc_grid[order[end]]; length = 500)
    dv_dx_raw = [DataInterpolations.derivative(itp, s) / xscale for s in soc_dense]
    hw = 15
    n = length(dv_dx_raw)
    dv_dx = [mean(dv_dx_raw[max(1, i - hw):min(n, i + hw)]) for i in 1:n]

    x_comp = soc_grid .* xscale
    lines!(ax1, x_comp, v_grid; color = :black, linewidth = 2, label = "Composite OCV")
    lines!(ax2, collect(soc_dense) .* xscale, dv_dx; color = :black, linewidth = 2)
    ylims!(ax2, 0, nothing)
    ax1.xgridvisible = false
    ax1.ygridvisible = false
    ax2.xgridvisible = false
    ax2.ygridvisible = false

    axislegend(ax1; position = :rb, merge = true, framevisible = false)
    return
end

# distribution of cell SOH (capacity relative to nominal) across the fleet
function plot_cell_soh_hist(comp_fit; Q_nom = 100)
    soh = Measurements.value.(comp_fit.Q_cell) ./ Q_nom .* 100

    fig = Figure(size = (450, 400))
    ax = Axis(fig[1, 1], xlabel = "Cell SOH / %", ylabel = "Cell count")

    bins = 40:1:90
    hist!(ax, soh; bins, color = :gray, strokewidth = 1)
    xlims!(ax, first(bins), last(bins))
    ylims!(ax, 0, nothing)

    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.rightspinevisible = false
    ax.topspinevisible = false
    ax.ytrimspine = (false, true)
    return fig
end

function plot_cell_soh(comp_fit, cells)
    fig = Figure(size = (700, 400))
    gl1 = GridLayout(fig[1, 1])
    gl2 = GridLayout(fig[1, 2])
    plot_composite_ocv!(gl1, comp_fit, cells; vertical = true)
    plot_soh_heatmap!(gl2, comp_fit)
    colsize!(fig.layout, 2, Relative(0.4))
    Label(fig[1, 1, TopLeft()], "A"; fontsize = 20, font = :bold, padding = (0, 0, 5, 0))
    Label(fig[1, 2, TopLeft()], "B"; fontsize = 20, font = :bold, padding = (0, 0, 5, 0))
    return fig
end

function plot_module_soh(df_soh; whiskers = true)
    fig = Figure(size = (700, 360))
    gl = GridLayout(fig[1, 1])
    plot_module_soh!(gl, df_soh; whiskers)
    return fig
end

function plot_module_soh!(
        layout, df_soh;
        whiskers = true, legend = true,
        # (module-based, cell-based)
        colors = Makie.wong_colors()[[2, 1]],
    )
    ax = Axis(layout[1, 1])

    soh_μ = Measurements.value.(df_soh.soh) * 100
    soh_σ = Measurements.uncertainty.(df_soh.soh) * 100

    m_soh_μ = Measurements.value.(df_soh.soh_module) * 100
    m_soh_σ = Measurements.uncertainty.(df_soh.soh_module) * 100

    ids = 1:27

    # dumbbell: gray connector emphasizes the module-vs-cell gap
    linesegments!(ax, repeat(ids; inner = 2), collect(Iterators.flatten(zip(m_soh_μ, soh_μ))); color = (:gray, 0.6), linewidth = 2)
    if whiskers
        errorbars!(ax, ids, m_soh_μ, 2 .* m_soh_σ; color = (colors[1], 0.6), whiskerwidth = 6)
        errorbars!(ax, ids, soh_μ, 2 .* soh_σ; color = (colors[2], 0.6), whiskerwidth = 6)
    end
    scatter!(ax, ids, m_soh_μ; color = colors[1], markersize = 10, label = "Module-based")
    scatter!(ax, ids, soh_μ; color = colors[2], markersize = 10, label = "Cell-based")

    vlines!(ax, [9.5, 18.5]; color = (:black, 0.3), linewidth = 1)

    ylims!(ax, 38, 100)
    xlims!(ax, 0.3, 27.7)
    ax.yticks = 40:10:100
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.rightspinevisible = false
    ax.topspinevisible = false
    ax.ylabel = "Module SOH / %"
    ax.xlabel = "Module ID"
    module_id_xticks!(ax)

    if legend
        Legend(layout[2, 1], ax; orientation = :horizontal, framevisible = false)
    end
    return ax
end

function plot_module_inhomogeneity(df_soh; whiskers = true)
    fig = Figure(size = (700, 360))
    gl = GridLayout(fig[1, 1])
    plot_module_inhomogeneity!(gl, df_soh; whiskers)
    return fig
end

function plot_module_inhomogeneity!(
        layout, df_soh;
        whiskers = true, legend = true,
        # (irreversible, reversible)
        bar_colors = Makie.wong_colors()[[3, 4]],
    )
    ax = Axis(layout[1, 1])

    total = (df_soh.loss_soh .+ df_soh.loss_soc) * 100
    irrev_μ = Measurements.value.(df_soh.loss_soh) * 100
    rev_μ = Measurements.value.(df_soh.loss_soc) * 100
    total_μ = Measurements.value.(total)
    total_σ = Measurements.uncertainty.(total)

    ids = 1:27

    barplot!(ax, ids, irrev_μ; width = 0.6, color = bar_colors[1], label = "Irreversible")
    barplot!(ax, ids, rev_μ; offset = irrev_μ, width = 0.6, color = bar_colors[2], label = "Reversible")
    if whiskers
        errorbars!(ax, ids, total_μ, 2 .* total_σ; color = (:black, 0.5), linewidth = 1.2, whiskerwidth = 5)
    end

    vlines!(ax, [9.5, 18.5]; color = (:black, 0.3), linewidth = 1)

    ylims!(ax, 0, 35)
    xlims!(ax, 0.3, 27.7)
    ax.yticks = 0:10:30
    ax.yminorticks = 0:5:35
    ax.yminorticksvisible = true
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.rightspinevisible = false
    ax.topspinevisible = false
    ax.ylabel = "SOH loss / %"
    ax.xlabel = "Module ID"
    module_id_xticks!(ax)

    if legend
        Legend(layout[2, 1], ax; orientation = :horizontal, framevisible = false)
    end
    return ax
end


function plot_module_summary(
        df_soh; whiskers = true,
        soh_colors = Makie.wong_colors()[[1, 2]],
        bar_colors = Makie.wong_colors()[[4, 3]],
    )
    fig = Figure(size = (700, 400))
    gl = GridLayout(fig[1, 1])
    gl1 = GridLayout(gl[1, 1])
    gl2 = GridLayout(gl[2, 1])
    ax1 = plot_module_soh!(gl1, df_soh; whiskers, legend = false, colors = soh_colors)
    ax2 = plot_module_inhomogeneity!(gl2, df_soh; whiskers, legend = false, bar_colors)

    hidexdecorations!(ax1, ticks = false, minorticks = false)
    linkxaxes!(ax1, ax2)
    rowsize!(gl, 1, Auto(1.4))
    rowgap!(gl, 1, 8)

    axislegend(ax1; position = :cb, framevisible = false)
    axislegend(ax2; position = :ct, framevisible = false, patchsize = (12, 12), padding = (0, 0, 0, 0))

    for (ax, tag) in zip((ax1, ax2), ("A", "B"))
        text!(ax, 0.01, 0.98; text = tag, space = :relative, align = (:left, :top), font = :bold, fontsize = 20)
    end
    return fig
end

function plot_soc_trajectories!(
        layout, df;
        title = "",
        # (module-based, cell-based)
        colors = Makie.wong_colors()[[1, 2]],
    )
    ax = Axis(layout[1, 1]; title)
    th = df.t ./ 3600

    for c in 1:12
        lines!(ax, th, df[:, "soc_cell_$c"] .* 100; color = (:gray, 0.6), linewidth = 1)
    end
    lines!(ax, th, df.soc_module .* 100; color = colors[1], linewidth = 2)
    lines!(ax, th, df.soc_pack .* 100; color = colors[2], linewidth = 2)

    xlims!(ax, first(th), last(th))
    ax.xlabel = "Time / h"
    ax.ylabel = "SOC / %"
    ax.xgridvisible = false
    ax.ygridvisible = false
    return ax
end

function plot_soc_comparison(df_norm, df_out; titles = ("", ""), colors = Makie.wong_colors()[[1, 2]])
    fig = Figure(size = (700, 320))
    gl = GridLayout(fig[1, 1])
    ax1 = plot_soc_trajectories!(GridLayout(gl[1, 1]), df_norm; colors, title = titles[1])
    ax2 = plot_soc_trajectories!(GridLayout(gl[1, 2]), df_out; colors, title = titles[2])
    linkyaxes!(ax1, ax2)

    elements = [
        LineElement(color = (:gray, 0.6), linewidth = 1),
        LineElement(color = colors[1], linewidth = 2),
        LineElement(color = colors[2], linewidth = 2),
    ]
    Legend(
        fig[2, 1], elements, ["Cells", "Module-based", "Cell-based"];
        orientation = :horizontal, framevisible = false,
    )
    return fig
end

# per-module distribution of the SOC error trajectories
function plot_soc_discrepancy(soc_err)
    fig = Figure(size = (700, 300))
    ax = Axis(fig[1, 1])
    n_t, n_mod = size(soc_err)
    xs = repeat(1:n_mod; inner = n_t)

    boxplot!(ax, xs, vec(soc_err) .* 100; width = 0.6, color = (:gray, 0.6), whiskerwidth = 0.4, markersize = 3)
    hlines!(ax, [0]; color = (:black, 0.4), linewidth = 1)
    vlines!(ax, [9.5, 18.5]; color = (:black, 0.3), linewidth = 1)

    xlims!(ax, 0.3, n_mod + 0.7)
    ax.ylabel = "Module − cell SOC / %"
    ax.xlabel = "Module ID"
    module_id_xticks!(ax)
    ax.xgridvisible = false
    ax.ygridvisible = false
    return fig
end

function plot_cell_v_rmse(df_v_cell)
    fig = Figure(size = (700, 300))
    ax = Axis(fig[1, 1], xlabel = "Cell voltage RMSE / mV", ylabel = "Cell count")
    hist!(ax, df_v_cell.rmse; bins = 0:0.5:25, color = :gray, strokewidth = 1)
    ylims!(ax, 0, nothing)
    return fig
end

function plot_module_v_rmse(df_v_module; colors = Makie.wong_colors()[[1, 2]])
    fig = Figure(size = (700, 300))
    ax = Axis(fig[1, 1], xlabel = "Module ID", ylabel = "Module voltage RMSE / mV")
    ids = 1:nrow(df_v_module)

    linesegments!(
        ax, repeat(ids; inner = 2),
        collect(Iterators.flatten(zip(df_v_module.rmse_module, df_v_module.rmse_cells)));
        color = (:gray, 0.6), linewidth = 2,
    )
    scatter!(ax, ids, df_v_module.rmse_module; color = colors[1], markersize = 10, label = "Module-based")
    scatter!(ax, ids, df_v_module.rmse_cells; color = colors[2], markersize = 10, label = "Cell-based (Σ cells)")
    vlines!(ax, [9.5, 18.5]; color = (:black, 0.3), linewidth = 1)

    xlims!(ax, 0.3, last(ids) + 0.7)
    ylims!(ax, 0, nothing)
    module_id_xticks!(ax)
    ax.xgridvisible = false
    ax.ygridvisible = false
    axislegend(ax; framevisible = false)
    return fig
end

# charge-estimation accuracy on the reference module: time-resolved SOC error vs the oscilloscope
# ground truth, one line per cell. Shows the error magnitude, its temporal structure, and the
# cell-to-cell consistency in one view.
function plot_charge_error(df; color = Makie.wong_colors()[1])
    fig = Figure(size = (700, 320))
    ax = Axis(fig[1, 1], xlabel = "Time / h", ylabel = "SOC error / %")
    cols = filter(!=("t"), names(df))

    hlines!(ax, [0]; color = :gray, linewidth = 0.5)
    for c in cols
        lines!(ax, df.t, df[!, c]; color = (color, 0.7), linewidth = 1.5)
    end

    xlims!(ax, extrema(df.t)...)
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.yminorticksvisible = true
    return fig
end

# fault-rejection diagnostic: per scenario (column), charge trajectory (top) and error (bottom) for
# the oscilloscope reference, module-current Coulomb counting, and the EKF estimate (±2σ band).
function plot_soc_diagnostic(diag; color = :dodgerblue)
    fig = Figure(size = (700, 400))
    ax = [Axis(fig[i, j]) for i in 1:2, j in 1:2]
    for (col, d) in enumerate(diag)
        e_cc, e = d.q_cc .- d.q_ref, d.q .- d.q_ref

        xlims!(ax[1, col], d.t[begin], d.t[end])
        lines!(ax[1, col], d.t, d.q_ref; color = :black, label = "Reference (oscilloscope CC)")
        lines!(ax[1, col], d.t, d.q_cc; color = :red, linestyle = :dash, label = "Coulomb counting (module + faults)")
        band!(ax[1, col], d.t, d.q .- 2d.qσ, d.q .+ 2d.qσ; color = (color, 0.5), label = "EKF estimate")
        lines!(ax[1, col], d.t, d.q; color, label = "EKF estimate")
        hidexdecorations!(ax[1, col]; ticks = false)

        xlims!(ax[2, col], d.t[begin], d.t[end])
        ax[2, col].yautolimitmargin = (0.1, 0.1)
        hlines!(ax[2, col], 0; color = :black, linestyle = :dot)
        lines!(ax[2, col], d.t, e_cc; color = :red, linestyle = :dash)
        lines!(ax[2, col], d.t, e; color)
        band!(ax[2, col], d.t, e .- 2d.qσ, e .+ 2d.qσ; color = (color, 0.5))
        linkxaxes!(ax[1, col], ax[2, col])

        ax[1, col].title = d.offset == 0 && d.bias == 0 ? "No faults" : "$(d.offset) Ah offset + $(d.bias) A current bias"
        ax[2, col].xlabel = "Time / h"
    end
    for col in 1:2
        ax[1, col].ylabel = "Charge / Ah"
        ax[2, col].ylabel = "Error / Ah"
    end
    foreach(a -> (a.xgridvisible = false; a.ygridvisible = false), ax)
    linkyaxes!(ax[1, 1], ax[1, 2])
    linkyaxes!(ax[2, 1], ax[2, 2])
    rowsize!(fig.layout, 2, Relative(0.3))
    Legend(fig[3, 1:2], fig.content[1]; orientation = :horizontal, merge = true, framevisible = false)
    return fig
end

# supplementary: time-resolved SOC error, module × time
function plot_soc_discrepancy_heatmap(tg, soc_err)
    fig = Figure(size = (700, 340))
    ax = Axis(fig[1, 1], xlabel = "Time / h", ylabel = "Module ID")
    n_mod = size(soc_err, 2)

    cr = maximum(abs, soc_err) * 100
    hm = heatmap!(ax, tg ./ 3600, 1:n_mod, soc_err .* 100; colormap = :vik, colorrange = (-cr, cr))
    hlines!(ax, [9.5, 18.5]; color = :black, linewidth = 1)
    Colorbar(fig[1, 2], hm; label = "Module − cell SOC / %")

    ax.yticks = ([1, 6, 10, 15, 19, 24], ["P1M1", "P1M6", "P2M1", "P2M6", "P3M1", "P3M6"])
    return fig
end

