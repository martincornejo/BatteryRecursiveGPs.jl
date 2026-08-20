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
function plot_data_resolution(data; completeness = nothing, yscale = log10)
    avail = isnothing(completeness) ? nothing : Dict(r.signal => r.completeness for r in eachrow(completeness))
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
        Axis(
                fig[i, 1]; yscale, ylabel = "Count", titlealign = :left, titlesize = 12,
                xticks = 0:10:80, xminorticks = IntervalsBetween(10), xminorticksvisible = true
            )
            for i in 1:4
    ]
    bins = 0.5:1:80.5  # integer-second timestamps, center bars on integers

    nmax = 0
    for (i, (key, name, color)) in enumerate(signals)
        Δt = Dates.value.(diff(data[key][!, "_time"])) * 1.0e-3
        hist!(
            ax[i], Δt; strokewidth = 1, strokecolor = :black, color, bins,
            fillto = logscale ? 0.5 : 0.0
        )
        nmax = max(nmax, maximum(StatsBase.fit(Histogram, Δt, bins).weights))
        ax[i].title = if isnothing(avail)
            "$name  (median $(round(Int, median(Δt))) s)"
        else
            "$name  (median $(round(Int, median(Δt))) s, $(round(Int, 100avail[key]))% available)"
        end
    end

    linkaxes!(ax...)
    xlims!(ax[1], 0, 81)
    logscale && ylims!(ax[1], 0.5, 2nmax)
    foreach(a -> hidexdecorations!(a; grid = false, ticks = false, minorticks = false), ax[1:3])
    ax[4].xlabel = "Sampling interval / s"

    return fig
end

function plot_module_data(data; N = 5)
    fig = Figure(size=(700, 420))
    ax = [Axis(fig[i,1]) for i in 1:3]
        
    # color = module ID M1–M9 (same across phases)
    colors = vcat(Makie.wong_colors(), [RGBAf(0, 0, 0, 1), RGBAf(0.6, 0.6, 0.6, 1)])

    t0 = data[:module_voltage]._time[begin]
    t_end = Dates.value(data[:module_voltage]._time[end] - t0) * 1.0e-3 / 3600

    ids = [(; p, m) for p in 1:3, m in 1:9] |> vec |> sort
    for id in ids
        (;p, m) = id
        df_V = select(data[:module_voltage], "_time" => "time", "module_voltage_$(p)_$(m)" => "value")
        df_i = select(data[:module_current], "_time" => "time", "module_average_current_$(p)_$(m)" => ByRow(x -> -x) => "value")
        df_T = select(data[:battery_temperature], "_time" => "time", "battery_sensor_temperature_$(p)_$(m)_1" => "value")
        for (k, df) in enumerate((df_V, df_i, df_T))
            df[!, :t] = Dates.value.(df.time .- t0) * 1.0e-3 / 3600
            lines!(ax[k, 1], df.t[1:N:end], df.value[1:N:end]; color = (colors[m], 0.5))
        end
    end

    for i in eachindex(ax)
        xlims!(ax[i], 0, t_end)
        ax[i].xgridvisible = false
        ax[i].ygridvisible = false
        ax[i].xminorticksvisible = true
        ax[i].xminorticks = IntervalsBetween(5)

        if i < 3
            hidexdecorations!(ax[i], ticks=false, minorticks=false)
        end
    end
    ax[1].ylabel = "Voltage / V"
    ax[2].ylabel = "Current / A"
    ax[3].ylabel = "Temperature / °C"
    ax[3].xlabel = "Time / h"

    mod_elems = [LineElement(color = colors[m], linewidth = 3) for m in 1:9]
    Legend(
        fig[1:3, 2], mod_elems, ["M$m" for m in 1:9], "Module ID";
        orientation = :vertical, titleposition = :top, framevisible = false
    )

    fig
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

        axs[1, j].title = "Module P$(p)M$(m)"
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
# draws the example open-loop fit into `gl` as two stacked panels (voltage + error,
# error ≈ 1/3 height); returns (ax_v, ax_e). Measured = neutral gray, cell model = orange.
function plot_sim!(gl, model::AbstractBatteryModel, sol; Ts = 1.0)
    kf = model.kf
    zt = kf.p.zt
    (; idx, u, yt, yμ, yΣ) = sol

    μ = StatsBase.reconstruct(zt.v, first.(yμ))
    σ = StatsBase.reconstruct(zt.σ, sqrt.(first.(yΣ)))
    t = (0:(length(u) - 1)) * Ts / 3600 |> collect

    c_meas = :gray30        # measured reference (neutral, dark enough to balance the model)
    c_model = Makie.wong_colors()[2]   # cell-model prediction / error (orange), matching "cells" in the accuracy figure

    ax_v = Axis(gl[1, 1]; ylabel = "Voltage / V")
    ax_e = Axis(gl[2, 1]; ylabel = "Error / mV", xlabel = "Time / h")

    v = StatsBase.reconstruct(zt.v, first.(yt))
    band!(ax_v, t[idx], μ - 2σ, μ + 2σ; color = (c_model, 0.3), label = "Model")
    lines!(ax_v, t[idx], μ; color = c_model, label = "Model")
    lines!(ax_v, t[idx], v; color = c_meas, label = "Measured")

    e = v - μ
    rmse = sqrt(sum(abs2, e) / length(e)) * 1.0e3
    elim = maximum(abs, vcat(e - 2σ, e + 2σ)) * 1.0e3
    hlines!(ax_e, [0]; color = (:black, 0.4), linewidth = 1.2)
    band!(ax_e, t[idx], (e - 2σ) * 1.0e3, (e + 2σ) * 1.0e3; color = (c_model, 0.3))
    lines!(ax_e, t[idx], e * 1.0e3; color = c_model)
    text!(
        ax_e, 0.99, 0.96; text = "RMSE = $(round(rmse; digits = 1)) mV",
        space = :relative, align = (:right, :top), fontsize = 11
    )

    if 0 < sol.tt < length(sol.u)
        vlines!(ax_v, t[sol.tt]; color = :red)
        vlines!(ax_e, t[sol.tt]; color = :red)
    end

    xlims!(ax_v, t[begin], t[end])
    xlims!(ax_e, t[begin], t[end])
    ylims!(ax_e, -1.1elim, 1.1elim)
    for a in (ax_v, ax_e)
        a.xgridvisible = false
        a.ygridvisible = false
        a.yminorticks = IntervalsBetween(2)
        a.yminorticksvisible = true
        a.xminorticks = IntervalsBetween(2)
        a.xminorticksvisible = true
    end
    linkxaxes!(ax_v, ax_e)
    hidexdecorations!(ax_v; grid = false, ticks = false)
    axislegend(ax_v; merge = true, framevisible = false, position = :rb, padding = (4, 4, 2, 2))
    rowsize!(gl, 1, Auto(2))   # error panel ≈ 1/3 of the example height
    rowsize!(gl, 2, Auto(1))
    rowgap!(gl, 6)
    return (ax_v, ax_e)
end

function plot_sim(model::AbstractBatteryModel, sol; Ts = 1.0)
    fig = Figure(size = (700, 300))
    plot_sim!(GridLayout(fig[1, 1]), model, sol; Ts)
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

# Figure showing the RCGP ECM (OCV + R0/R1) at one filter step, next to the measured voltage and
# the voltage the ECM reproduces on its own, with a cursor marking how much data the filter has
# seen. Returns the figure plus a `set_frame!(i)` closure that moves it to frame `i` — shared by
# `animate_model` and `plot_animation_snapshot`.
#
# NOTE: pass the FULL sol from run_kf! BEFORE reduce_sol (needs per-timestep xt/Rt), and pass a
# FRESHLY BUILT model — `run_kf!` mutates the KF in place, so re-running a fitted model starts
# from its converged posterior and the ECM is already learned at step 1.
#
# `prior = (; x, R)` prepends the pre-observation state as frame 1: `run_kf!` records the state
# AFTER each correction, so `sol.xt[1]` has already absorbed one measurement. Capture it off the
# freshly built model (`copy(model.kf.x)`, `copy(model.kf.R)`) before running the filter.
#
# Mutates `model.kf` (see `predict_v` below) — the ECM curves read the passed-in x/R rather than
# the filter state, so the sol stays valid, but do not rely on `model` afterwards.
function _model_frame(model::RCGPModel, sol; prior = nothing)
    kf = model.kf
    zt = kf.p.zt
    (; yt) = sol

    xt = isnothing(prior) ? sol.xt : vcat([prior.x], sol.xt)
    Rt = isnothing(prior) ? sol.Rt : vcat([prior.R], sol.Rt)
    cursor_at = i -> clamp(isnothing(prior) ? i : i - 1, 1, length(yt))

    q̂ = collect(range(extrema(kf.p.r1.b0)...; step = 0.01))
    q = StatsBase.reconstruct(zt.q, q̂)

    # ECM curves at a given filter state (x, R)
    ecm(x, R) = begin
        ocv = predict_gp(kf, q̂, x, R, :ocv)
        ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
        ocvσ = StatsBase.reconstruct(zt.σ, sqrt.(diag(ocv.Σ)))
        xc = ComponentVector(x, kf.p.xid)
        r0μ = (StatsBase.reconstruct(zt.r, [abs(xc.r0.r)]) |> first) * 1.0e3
        r1 = predict_gp(kf, q̂, x, R, :r1)
        r1μ = StatsBase.reconstruct(zt.r, r1.μ) * 1.0e3
        r1σ = StatsBase.reconstruct(zt.r, sqrt.(diag(r1.Σ))) * 1.0e3
        (; ocvμ, ocvσ, rμ = r0μ .+ r1μ, rσ = r1σ)
    end

    # Voltage the ECM frozen at (x, R) reproduces on its own: `reinit_kf!` resets charge and RC
    # voltage to the start of the window while keeping the GP/R0 posteriors, and `tt = 0` skips
    # every correction — a pure simulation over the whole dataset, not the one-step-ahead
    # prediction in `sol.yμ` (which is corrected each step and therefore always tracks).
    # The simulation is uncorrected across the whole window, but the parameters come from data up
    # to the cursor only: left of it the curve reconstructs data the filter has seen, right of it
    # it extrapolates. Hence the plain "Model" label rather than "prediction".
    predict_v = (x, R) -> begin
        reinit_kf!(model; x, R)
        ol = run_kf!(model, sol.u, sol.y; tt = 0)
        μ = StatsBase.reconstruct(zt.v, first.(ol.yμ))
        σ = StatsBase.reconstruct(zt.σ, sqrt.(first.(ol.yΣ)))
        (; μ, σ)
    end

    fig = Figure(size = (900, 400))
    axo = Axis(fig[1, 2]; ylabel = "OCV / V")
    axr = Axis(fig[2, 2]; ylabel = "R / mΩ", xlabel = "Charge / Ah")
    axv = Axis(fig[1:2, 1]; ylabel = "Voltage / V", xlabel = "Step")
    linkxaxes!(axo, axr)
    hidexdecorations!(axo; ticks = false, grid = false)

    (; ocvμ, ocvσ, rμ, rσ) = ecm(xt[1], Rt[1])
    lo = lines!(axo, q, ocvμ; color = Cycled(1))
    bo = band!(axo, q, ocvμ + 2ocvσ, ocvμ - 2ocvσ; color = Cycled(1), alpha = 0.3)
    lr = lines!(axr, q, rμ; color = Cycled(1))
    br = band!(axr, q, rμ + 2rσ, rμ - 2rσ; color = Cycled(1), alpha = 0.3)

    v = StatsBase.reconstruct(zt.v, first.(yt))
    k = eachindex(v)
    v̂ = predict_v(xt[1], Rt[1])
    lines!(axv, k, v; color = :gray, label = "Measured")
    lv = lines!(axv, k, v̂.μ; color = Cycled(2), label = "Model")
    bv = band!(axv, k, v̂.μ - 2v̂.σ, v̂.μ + 2v̂.σ; color = Cycled(2), alpha = 0.3)
    cursor = vlines!(axv, cursor_at(1); color = :red)
    axislegend(axv; position = :rb)

    # Fixed limits: the prior bands are far wider than the converged ones, so autoscaling would
    # make the early frames unreadable and the late ones flat. Anchor each ECM panel on its own
    # converged (final-state) curve — the measured voltage range does not cover the OCV, which
    # extends past it at the edges of the charge window. The early model traces and their bands
    # run well outside these ranges and are deliberately left to clip.
    fin = ecm(xt[end], Rt[end])
    olo, ohi = extrema(vcat(fin.ocvμ - 2fin.ocvσ, fin.ocvμ + 2fin.ocvσ))
    ylims!(axo, olo - 0.1(ohi - olo), ohi + 0.1(ohi - olo))
    rlo, rhi = extrema(vcat(fin.rμ - 2fin.rσ, fin.rμ + 2fin.rσ))
    ylims!(axr, rlo - 0.5(rhi - rlo), rhi + 0.5(rhi - rlo))
    xlims!(axo, extrema(q)...)  # linked to axr

    vlo, vhi = extrema(v)
    ylims!(axv, vlo - 0.15(vhi - vlo), vhi + 0.15(vhi - vlo))
    xlims!(axv, first(k), last(k))

    set_frame! = i -> begin
        (; ocvμ, ocvσ, rμ, rσ) = ecm(xt[i], Rt[i])
        Makie.update!(lo, arg2 = ocvμ)
        Makie.update!(bo, arg2 = ocvμ + 2ocvσ, arg3 = ocvμ - 2ocvσ)
        Makie.update!(lr, arg2 = rμ)
        Makie.update!(br, arg2 = rμ + 2rσ, arg3 = rμ - 2rσ)
        v̂ = predict_v(xt[i], Rt[i])
        Makie.update!(lv, arg2 = v̂.μ)
        Makie.update!(bv, arg2 = v̂.μ - 2v̂.σ, arg3 = v̂.μ + 2v̂.σ)
        Makie.update!(cursor, arg1 = cursor_at(i))
    end

    return (; fig, set_frame!, n_frames = length(xt))
end

# Animate the RCGP ECM as the filter learns over the dataset. `step` is the stride in filter steps
# between rendered frames; duration is `length(1:step:n_frames) / framerate` seconds. See
# `_model_frame` for what `model`/`sol`/`prior` must be.
# NOTE: each frame re-simulates the whole dataset open-loop (~3 s), so a full run costs roughly
# `n_frames / step × 3 s` — check the frame count before starting. Stretching the duration via
# `framerate` is free; doing it via `step` is not.
function animate_model(file, model::RCGPModel, sol; step = 60, framerate = 24, prior = nothing)
    (; fig, set_frame!, n_frames) = _model_frame(model, sol; prior)
    return record(set_frame!, fig, file, 1:step:n_frames; framerate)
end

# Single frame of the learning animation, at frame `i` — debugging aid for checking the figure
# without paying for the full video. Unexported: call it as `YuasaAnalysis.plot_animation_snapshot`.
# With `prior` given, frame 1 is the pre-observation state and filter step `k` is frame `k + 1`.
function plot_animation_snapshot(model::RCGPModel, sol, i; prior = nothing)
    (; fig, set_frame!) = _model_frame(model, sol; prior)
    set_frame!(i)
    return fig
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
    colorrange = (60, 82)
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
    xscale = xaxis == :ah ? Q_full : 100.0   # SOC in %, as elsewhere
    xlabel = xaxis == :ah ? "Capacity / Ah" : "SOC / %"
    dlabel = xaxis == :ah ? "dV/dQ / (mV/Ah)" : "dV/dSOC / (mV/%)"

    if vertical
        ax1 = Axis(layout[1, 1]; ylabel = "Voltage / V")
        ax2 = Axis(layout[2, 1]; ylabel = dlabel, xlabel)
        hidexdecorations!(ax1; grid = false, ticks = false)
        linkxaxes!(ax1, ax2)
        rowgap!(layout, 1, 5)
    else
        ax1 = Axis(layout[1, 1]; ylabel = "Voltage / V", xlabel)
        ax2 = Axis(layout[1, 2]; ylabel = dlabel, xlabel)
    end

    for ax in (ax1, ax2)
        xaxis == :soc && (ax.xticks = 0:25:100)
        ax.xminorticks = IntervalsBetween(5)   # every 5 % SOC
        ax.xminorticksvisible = true
        ax.yminorticks = IntervalsBetween(2)
        ax.yminorticksvisible = true
    end
    # sparse majors, the intermediate values demoted to minor ticks
    ax1.yticks = 3.5:0.2:4.1
    ax2.yticks = 0:20:60

    Q_cell_μ = Measurements.value.(Q_cell)
    s0_μ = Measurements.value.(s0)
    colormap = :lipari
    colorrange = (60, 82)
    lowclip = :black
    for i in eachindex(cells)
        q = cells[i].q
        v = cells[i].μ
        x = (q ./ Q_cell_μ[i] .+ s0_μ[i]) .* xscale
        lines!(ax1, x, v; color = Q_cell_μ[i], colorrange, colormap, alpha = 0.6, label = "Cell OCV")
        lines!(ax2, x[2:end], diff(v) ./ diff(x) .* 1000; color = Q_cell_μ[i], colorrange, colormap, alpha = 0.4)
    end

    order = sortperm(soc_grid)
    itp = PCHIPInterpolation(v_grid[order], soc_grid[order])
    soc_dense = range(soc_grid[order[1]], soc_grid[order[end]]; length = 500)
    dv_dx_raw = [DataInterpolations.derivative(itp, s) / xscale * 1000 for s in soc_dense]
    hw = 15
    n = length(dv_dx_raw)
    dv_dx = [mean(dv_dx_raw[max(1, i - hw):min(n, i + hw)]) for i in 1:n]

    x_comp = soc_grid .* xscale
    lines!(ax1, x_comp, v_grid; color = :black, linewidth = 2, label = "Composite OCV")
    lines!(ax2, collect(soc_dense) .* xscale, dv_dx; color = :black, linewidth = 2)
    ylims!(ax1, 3.4, 4.1)
    ylims!(ax2, 0, 60)
    xaxis == :soc && xlims!(ax2, 0, 100)   # linked, so ax1 follows
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

    bins = 35:1:85
    hist!(ax, soh; bins, color = :gray, strokewidth = 1)
    xlims!(ax, first(bins), last(bins))
    ylims!(ax, 0, 100)
    ax.yticks = 0:20:100

    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.rightspinevisible = false
    ax.topspinevisible = false
    ax.ytrimspine = (false, true)
    return fig
end

function plot_cell_soh(comp_fit, cells; xaxis = :soc)
    fig = Figure(size = (700, 400))
    gl1 = GridLayout(fig[1, 1])
    gl2 = GridLayout(fig[1, 2])
    plot_composite_ocv!(gl1, comp_fit, cells; xaxis, vertical = true)
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

    ylims!(ax, 35, 90)
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
        text!(ax, 0.01, 1.02; text = tag, space = :relative, align = (:left, :top), font = :bold, fontsize = 20)
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
        μ = Measurements.value.(df[:, "soc_cell_$c"]) .* 100
        σ = Measurements.uncertainty.(df[:, "soc_cell_$c"]) .* 100
        band!(ax, th, μ .- 2σ, μ .+ 2σ; color = (:gray, 0.15))
        lines!(ax, th, μ; color = (:gray, 0.6), linewidth = 1)
    end

    for (col, color) in ((:soc_module, colors[1]), (:soc_pack, colors[2]))
        μ = Measurements.value.(df[!, col]) .* 100
        σ = Measurements.uncertainty.(df[!, col]) .* 100
        band!(ax, th, μ .- 2σ, μ .+ 2σ; color = (color, 0.3))
        lines!(ax, th, μ; color, linewidth = 2)
    end

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
        [PolyElement(color = (colors[1], 0.3)), LineElement(color = colors[1], linewidth = 2)],
        [PolyElement(color = (colors[2], 0.3)), LineElement(color = colors[2], linewidth = 2)],
        [PolyElement(color = (:gray, 0.15)), LineElement(color = (:gray, 0.6), linewidth = 1)],
    ]
    Legend(
        fig[2, 1], elements, ["Module-based", "Cell-based", "Cells"];
        orientation = :horizontal, framevisible = false,
    )
    return fig
end

# per-module distribution of the SOC error trajectories
function plot_soc_discrepancy!(layout, soc_err)
    ax = Axis(layout[1, 1])
    n_t, n_mod = size(soc_err)
    xs = repeat(1:n_mod; inner = n_t)

    boxplot!(ax, xs, vec(Measurements.value.(soc_err)) .* 100; width = 0.6, color = (:gray, 0.6), whiskerwidth = 0.4, markersize = 3)
    hlines!(ax, [0]; color = (:black, 0.4), linewidth = 1)
    vlines!(ax, [9.5, 18.5]; color = (:black, 0.3), linewidth = 1)

    xlims!(ax, 0.3, n_mod + 0.7)
    ax.ylabel = "Module − cell SOC / %"
    ax.xlabel = "Module ID"
    module_id_xticks!(ax)
    ax.xgridvisible = false
    ax.ygridvisible = false
    return ax
end

function plot_soc_discrepancy(soc_err)
    fig = Figure(size = (700, 300))
    plot_soc_discrepancy!(GridLayout(fig[1, 1]), soc_err)
    return fig
end

function plot_soc_overview(
        df_norm, df_out, soc_err;
        titles = ("", ""), colors = Makie.wong_colors()[[1, 2]],
    )
    fig = Figure(size = (700, 530))

    ax_disc = plot_soc_discrepancy!(GridLayout(fig[1, 1:2]), soc_err)

    ax1 = plot_soc_trajectories!(GridLayout(fig[2, 1]), df_norm; colors, title = titles[1])
    ax2 = plot_soc_trajectories!(GridLayout(fig[2, 2]), df_out; colors, title = titles[2])
    linkyaxes!(ax1, ax2)
    hideydecorations!(ax2; ticks = false, grid = false)

    elements = [
        [PolyElement(color = (colors[1], 0.3)), LineElement(color = colors[1], linewidth = 2)],
        [PolyElement(color = (colors[2], 0.3)), LineElement(color = colors[2], linewidth = 2)],
        [PolyElement(color = (:gray, 0.15)), LineElement(color = (:gray, 0.6), linewidth = 1)],
    ]
    Legend(
        fig[3, 1:2], elements, ["Module-based", "Cell-based", "Cells"];
        orientation = :horizontal, framevisible = false,
    )

    rowsize!(fig.layout, 1, Auto(1.0))
    rowsize!(fig.layout, 2, Auto(1.2))

    for (ax, tag) in zip((ax_disc, ax1, ax2), ("A", "B", "C"))
        text!(ax, 0.01, 1.0; text = tag, space = :relative, align = (:left, :top), font = :bold, fontsize = 20)
    end

    return fig
end

# Fleet-wide open-loop voltage accuracy: every cell RMSE as a point, grouped by module,
# with the combined cell-model tick (the 12 cells as a virtual module, Σ predicted vs Σ
# measured cell voltage, ÷12), vs the dedicated module model on the per-cell scale (÷12).
# The combined tick — not the per-cell mean/median — is the like-for-like counterpart to the
# module model: both score a *sum* of cell voltages, so both keep the temporal cancellation
# between cells (mean(RMSE) discards it and overstates the error on heterogeneous modules).
# The strip
# (rather than a boxplot) shows the actual within-module distribution — one stray cell vs a
# broad spread — fits the cell-to-cell heterogeneity story, and avoids the unreliable Tukey
# 1.5·IQR fence at n = 12 plus any glyph clash with the Tukey boxplots used elsewhere (e.g.
# the SOC-error figure). The single off-scale cell (a current/voltage desync artifact) is
# named in place; fleet stats belong in the caption.
function plot_v_accuracy!(layout, df_v_cell, df_v_module; legend = true, colors = Makie.wong_colors()[[1, 2, 6]])
    ax = Axis(layout[1, 1])
    q99 = quantile(df_v_cell.rmse, 0.99)

    # focus the axis on the bulk + module comparison; the lone extreme cell (a desync
    # artifact) is named off-scale rather than allowed to stretch the whole axis.
    ytop = ceil(max(maximum(df_v_module.rmse_module) / 12, q99) * 1.05)
    # floor the axis just below the lowest point (not at 0) so the 4–14 mV band fills the panel
    ybot = floor(min(minimum(df_v_cell.rmse), minimum(df_v_module.rmse_module ./ 12))) - 0.5

    ax.xlabel = "Module ID"
    ax.ylabel = "Voltage RMSE / mV"
    mod_idx = Dict((; p, m) => (p - 1) * 9 + m for p in 1:3, m in 1:9)
    xs = [mod_idx[(; p = r.p, m = r.m)] for r in eachrow(df_v_cell)]

    # every on-scale cell as a point (the off-scale cell is named below) + a per-module
    # combined tick (cell models as a virtual module, ÷12); the point cloud shows whether a
    # module's spread is one stray cell or broad, the tick the like-for-like module comparison
    ks = 1:nrow(df_v_module)
    keep = df_v_cell.rmse .<= ytop
    cell_combined = df_v_module.rmse_cells ./ 12

    scatter!(ax, xs[keep], df_v_cell.rmse[keep]; color = (colors[2], 0.55), markersize = 7)
    scatter!(ax, ks, cell_combined; marker = :hline, markersize = 15, color = colors[2])
    scatter!(ax, ks, df_v_module.rmse_module ./ 12; color = colors[1], markersize = 9)

    # name the single worst cell in place — it is the off-scale point readers ask about
    worst = argmax(df_v_cell.rmse)
    if df_v_cell.rmse[worst] > ytop
        r = df_v_cell[worst, :]
        y_mark = ytop - 0.035 * (ytop - ybot)  # just below the top edge so the ▲ shows fully
        scatter!(ax, [xs[worst]], [y_mark]; color = colors[3], markersize = 11, marker = :utriangle)
        text!(
            ax, xs[worst] + 0.5, y_mark; text = "P$(r.p)M$(r.m)C$(r.c): $(round(Int, r.rmse)) mV",
            align = (:left, :center), fontsize = 12, color = colors[3]
        )
    end

    vlines!(ax, [9.5, 18.5]; color = (:black, 0.3), linewidth = 1)
    xlims!(ax, 0.3, 27.7)
    module_id_xticks!(ax)
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.yminorticks = IntervalsBetween(5)  # 1 mV steps between the 5 mV majors
    ax.yminorticksvisible = true
    ylims!(ax, ybot, ytop)

    # three entries below the axis (outside the data) — individual cells (cloud), the 12 cells
    # combined as a virtual module (tick), and the module model. orange = cell-level approach,
    # blue = module-level; per-cell scaling (÷12) is a derivation detail, kept to the caption.
    # placed below rather than inside because the tall points span the full width of the axis.
    if legend
        cell_pt_el = MarkerElement(; marker = :circle, color = (colors[2], 0.55), markersize = 7)
        cell_comb_el = MarkerElement(; marker = :hline, color = colors[2], markersize = 15)
        mod_el = MarkerElement(; marker = :circle, color = colors[1], markersize = 9)
        Legend(layout[2, 1], [cell_pt_el, cell_comb_el, mod_el], ["Cell-level (individual)", "Cell-level (virtual module)", "Module-level"]; orientation = :horizontal, framevisible = false)
    end
    return ax
end

function plot_v_accuracy(df_v_cell, df_v_module; colors = Makie.wong_colors()[[1, 2, 6]])
    fig = Figure(size = (700, 360))
    plot_v_accuracy!(GridLayout(fig[1, 1]), df_v_cell, df_v_module; colors)
    return fig
end

# Combined voltage-accuracy figure: (A) one example open-loop cell fit + residual — the
# qualitative story — over (B) the fleet-wide per-module accuracy — the quantitative one.
function plot_v_accuracy_overview(model::AbstractBatteryModel, sol, df_v_cell, df_v_module; Ts = 1.0, title = "")
    fig = Figure(size = (700, 530))

    gl_sim = GridLayout(fig[1, 1])
    ax_v, _ = plot_sim!(gl_sim, model, sol; Ts)
    ax_v.title = title

    plot_v_accuracy!(GridLayout(fig[2, 1]), df_v_cell, df_v_module)

    Label(gl_sim[1, 1, TopLeft()], "A"; font = :bold, fontsize = 18, padding = (0, 0, 5, 0))
    Label(fig[2, 1, TopLeft()], "B"; font = :bold, fontsize = 18, padding = (0, 0, 5, 0))
    rowgap!(fig.layout, -10)
    # panel A stacks two subplots (voltage + error) and panel B's row also carries the legend,
    # so give row B more weight to keep the two plotting areas similar in size
    rowsize!(fig.layout, 1, Auto(1))
    rowsize!(fig.layout, 2, Auto(1.5))
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

    soc_err_v = Measurements.value.(soc_err) * 100
    cr = maximum(abs, soc_err_v)
    hm = heatmap!(ax, tg ./ 3600, 1:n_mod, soc_err_v; colormap = :vik, colorrange = (-cr, cr))
    hlines!(ax, [9.5, 18.5]; color = :black, linewidth = 1)
    Colorbar(fig[1, 2], hm; label = "Module − cell SOC / %")

    ax.yticks = ([1, 6, 10, 15, 19, 24], ["P1M1", "P1M6", "P2M1", "P2M6", "P3M1", "P3M6"])
    return fig
end


# === hyperparameter selection figures ===

# Paper figure (supplementary): two-stage hyperparameter selection. Columns = cells |
# modules (module OCV ÷ n → per-cell scale); each column reads top-to-bottom as one story.
# A/B: dV/dSOC fans at the shared initial ℓ, flagged units (> thresh at init) highlighted;
# C/D: fans after per-unit adaptation; E/F: composite-OCV RMSE distribution, initial vs
# adapted (linear mV axis; off-scale outliers annotated as text instead of squeezed in).
# Selected-ℓ counts are tabulated separately (see selection_counts).
# `cells`/`modules` are calc_hyperparam_selection bundles (; fans, comp, rmse, thresh).
function plot_hyperparam_selection(cells, modules)
    fig = Figure(size = (700, 640), figure_padding = 8)
    wong = Makie.wong_colors()
    c_flag = wong[6]  # vermillion: flagged at init (> threshold)
    c_bulk = (:gray, 0.35)
    c_init = :gray30; c_adap = wong[1]

    axs = Matrix{Axis}(undef, 3, 2)
    for (col, lvl) in enumerate((cells, modules))
        # rows 1-2: dV/dSOC fans, initial vs adapted, composite in black
        for (row, stage) in ((1, :init), (2, :adapted))
            ax = Axis(fig[row, col]); axs[row, col] = ax
            fans = subset(lvl.fans, :stage => ByRow(==(stage)))
            for flagged in (false, true), r in eachrow(fans)  # bulk first, flagged on top
                r.flagged == flagged || continue
                lines!(ax, r.soc, r.dvdsoc; color = flagged ? (c_flag, 0.8) : c_bulk, linewidth = flagged ? 1.0 : 0.8)
            end
            lines!(ax, lvl.comp.soc, lvl.comp.dvdsoc; color = :black, linewidth = 2)
            ylims!(ax, -5, 30); xlims!(ax, 0, 100)
            ax.xticks = 0:25:100
            ax.xminorticks = IntervalsBetween(5)   # every 5 % SOC
            ax.xminorticksvisible = true
        end
        axs[2, col].xlabel = "SOC / %"

        # row 3: RMSE shift, initial vs adapted; off-scale outliers noted as text
        ax = Axis(fig[3, col]); axs[3, col] = ax
        xmax = 8.0
        density!(ax, lvl.rmse.init; color = (c_init, 0.2), strokecolor = c_init, strokewidth = 1.8, label = rich("initial ℓ", subscript("OCV")))
        density!(ax, lvl.rmse.adapted; color = (c_adap, 0.2), strokecolor = c_adap, strokewidth = 1.8, label = rich("adapted ℓ", subscript("OCV")))
        # off-scale units: one paired note each, named, numbers in the curve colors
        out = [r for r in eachrow(lvl.rmse) if r.init > xmax || r.adapted > xmax]
        for (i, r) in enumerate(out)
            text!(
                ax, 0.97, 0.95 - 0.15 * (i - 1);
                text = rich(
                    "$(r.name): ", rich(string(round(Int, r.init)), color = c_init), " → ",
                    rich(string(round(Int, r.adapted)), color = c_adap), " mV"
                ),
                space = :relative, align = (:right, :top), fontsize = 10,
            )
        end
        vlines!(ax, [lvl.thresh]; color = :black, linestyle = :dot, linewidth = 1.2)
        ax.xticks = 0:2:8
        xlims!(ax, 0, xmax); ylims!(ax, 0, nothing)
        ax.xlabel = "Composite-OCV RMSE / mV"
    end

    # fan-line legend (colors classify by the stage-1 fit; caption details)
    fanleg = [
        LineElement(color = (:gray, 0.7), linewidth = 1.2) => "≤ threshold (stage 1)",
        LineElement(color = (c_flag, 0.9), linewidth = 1.2) => "> threshold (stage 1)",
        LineElement(color = :black, linewidth = 2) => "composite",
    ]
    axislegend(axs[2, 1], first.(fanleg), last.(fanleg); position = :rt, framevisible = false, patchsize = (16, 10), rowgap = 0)

    Label(fig[0, 1], "Cells"; font = :bold, fontsize = 16, tellwidth = false)
    Label(fig[0, 2], "Modules (per cell)"; font = :bold, fontsize = 16, tellwidth = false)
    for col in 1:2
        axs[1, col].title = rich("Initial ℓ", subscript("OCV"), " (stage 1)")
        axs[2, col].title = rich("Adapted ℓ", subscript("OCV"), " (stage 2)")
        axs[1, col].titlefont = :regular; axs[2, col].titlefont = :regular
    end
    axs[1, 1].ylabel = "dV/dSOC / (mV/%)"; axs[2, 1].ylabel = "dV/dSOC / (mV/%)"; axs[3, 1].ylabel = "Density"
    foreach(ax -> hidexdecorations!(ax; ticks = false, grid = false), axs[1, :])
    for row in 1:3
        linkyaxes!(axs[row, 1], axs[row, 2])
        hideydecorations!(axs[row, 2]; ticks = false, grid = false)
    end
    axislegend(axs[3, 2]; position = :rt, framevisible = false, patchsize = (14, 10), rowgap = 0)
    for ax in vec(axs)
        ax.xgridvisible = false; ax.ygridvisible = false
    end
    for (tag, pos) in zip(("A", "B", "C", "D", "E", "F"), ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)))
        Label(fig[pos..., TopLeft()], tag; fontsize = 18, font = :bold, padding = (0, 25, 2, 0), halign = :left)
    end
    rowgap!(fig.layout, 6)
    return fig
end

# Paper figure (supplementary): all four GP hyperparameters of every unit — histograms in
# physical units (2 rows cells/modules × 4 equal columns ℓ_ocv/ℓ_r1/σ_ocv/σ_r1, ~40 bins
# each so bar widths match; ℓ_ocv log-x). Bars are stack-colored by the selected NOMINAL
# value (gray = 0.5 init, lipari steps for escalations) — the legend speaks the text's
# vocabulary, and the σ columns come out all-gray since σ is never adapted. Units beyond
# any cell capacity (ℓ_ocv > 100 Ah) are named; their 1-count bars would be invisible.
# Inputs are calc_scaled_hyperparams DataFrames.
function plot_hyperparam_scales(scaled_cells, scaled_mods)
    nom_esc = [0.3, 0.6, 0.85, 1.0, 1.5, 15.0]  # escalation-grid values (union of both ℓ grids)
    colors = Dict(v => get(cgrad(:lipari), t) for (v, t) in zip(nom_esc, range(0.8, 0.12, length = length(nom_esc))))
    colors[0.5] = RGBAf(0.65, 0.65, 0.65, 1)    # init: neutral gray
    nom_all = [0.5; nom_esc]                    # init drawn first (bottom of the stacks)

    function stacked_hist!(ax, groups, edges; logx = false)
        centers = logx ? sqrt.(edges[1:(end - 1)] .* edges[2:end]) : (edges[1:(end - 1)] .+ edges[2:end]) ./ 2
        base = zeros(length(centers))
        for (vals, color) in groups
            h = StatsBase.fit(Histogram, vals, edges).weights
            barplot!(ax, centers, Float64.(h); offset = base, width = diff(edges), color, gap = 0, strokewidth = 0.4, strokecolor = :white)
            base .+= h
        end
        return
    end

    # (value, nominal-for-color, xscale, xticks, xlims, bin edges, log-spaced?, label)
    specs = [
        (
            x -> x.ℓ_ocv, x -> x.nom_ocv, log10, ([10, 20, 50, 100, 200], ["10", "20", "50", "100", "200"]),
            (9, 220), 10 .^ (log10(9):0.035:log10(220)), true, rich("ℓ", subscript("OCV"), " / Ah"),
        ),
        (
            x -> x.ℓ_r1, x -> x.nom_r1, identity, (0:20:60, string.(0:20:60)),
            (5, 68), 5:1.5:68, false, rich("ℓ", subscript("R1"), " / Ah"),
        ),
        (
            x -> x.σ_ocv, x -> 0.5, identity, ([180, 230, 280], ["180", "230", "280"]),
            (175, 295), 178:2.9:294, false, rich("σ", subscript("OCV"), " / mV"),
        ),
        (
            x -> x.σ_r1, x -> 0.5, identity, (7:1:10, string.(7:1:10)),
            (6.1, 10.3), 6.2:0.1:10.2, false, rich("σ", subscript("R1"), " / mΩ"),
        ),
    ]

    fig = Figure(size = (700, 360), figure_padding = 8)
    axs = Matrix{Axis}(undef, 2, 4)
    for (col, (val, nom, xscale, xticks, lims, bins, logx, xlab)) in enumerate(specs)
        # Modules*: σ shown per cell (module value ÷ n) for a scale comparable to the cells
        for (row, (a, name)) in enumerate(((scaled_cells, "Cells"), (scaled_mods, "Modules*")))
            ax = Axis(fig[row, col]; xscale, xticks)
            axs[row, col] = ax
            groups = [(Float64[val(x) for x in eachrow(a) if nom(x) == v], colors[v]) for v in nom_all]
            stacked_hist!(ax, groups, collect(bins); logx)
            xlims!(ax, lims...); ylims!(ax, 0, nothing)
            col == 1 && (ax.ylabel = "$name / count")
            row == 2 ? (ax.xlabel = xlab) : hidexdecorations!(ax; ticks = false, grid = false)
            ax.xgridvisible = false; ax.ygridvisible = false
        end
        linkxaxes!(axs[:, col]...)
    end
    for row in 1:2
        linkyaxes!(axs[row, :]...)
        foreach(ax -> hideydecorations!(ax; ticks = false, grid = false), axs[row, 2:4])
    end
    for r in eachrow(scaled_cells)
        r.ℓ_ocv > 100 || continue
        lines!(axs[1, 1], [r.ℓ_ocv, r.ℓ_ocv], [45, 6]; color = :black, linewidth = 0.8)
        text!(axs[1, 1], r.ℓ_ocv, 50; text = r.name, align = (:right, :bottom), fontsize = 9)
    end
    fmt(v) = v == floor(v) ? string(Int(v)) : string(v)
    order = sort(nom_all)
    Legend(
        fig[3, 1:4],
        [PolyElement(color = colors[v]) for v in order],
        [(v == 0.5 ? "0.5 (init)" : fmt(v)) for v in order],
        "nominal ℓ / σ:";
        framevisible = false, patchsize = (12, 12), orientation = :horizontal,
        titleposition = :left, titlefont = :regular, colgap = 12,
    )
    rowgap!(fig.layout, 8); colgap!(fig.layout, 10)
    return fig
end


# Fitted ECM parameters across the fleet (consumes `calc_ecm_parameters`).
# Complements the ECM figure rather than repeating it: that one draws R_DC(q) curves for a handful
# of example units, this one carries the fleet SPREAD and the scalar parameters, which it does not
# show at all. Composite-free throughout — no SOC, no capacity, both of which come later.
#
# Cell and module fits are overlaid, modules on a per-cell basis (`n = 12` in the builder), because
# the COMPARISON is the result: R1, τ and k agree between the two levels, but module R0 sits ~40 %
# higher — the busbars and contacts that lie in a module's series path but not in a cell's own
# voltage measurement. Densities, not counts, since there are 324 cells against 27 modules.
function plot_ecm_parameters(df, df_mod = nothing; v_ref = 3.9, Δr = 0.06, Δτ = 5.0, T = 5:1:40, T0 = 25)
    wong = Makie.wong_colors()
    c_cell = wong[2]
    c_mod = wong[1]
    c_cell_med = wong[6]

    fig = Figure(size = (700, 450))

    pct(v) = (x = v * 100; isapprox(x, round(x); atol = 1.0e-9) ? string(round(Int, x)) : string(round(x; digits = 1)))
    function overlay!(ax, vc, vm; bins, ymax, ystep)
        ax.ytickformat = vs -> pct.(vs)
        ylims!(ax, nothing, ymax)
        ax.yticks = 0:ystep:ymax
        ax.yminorticks = IntervalsBetween(2)
        ax.yminorticksvisible = true
        hist!(ax, vc; bins, normalization = :probability, color = (c_cell, 0.85), strokewidth = 0.5, strokecolor = :white)
        isnothing(vm) && return
        hist!(ax, vm; bins, normalization = :probability, color = (c_mod, 0.55), strokewidth = 0.5, strokecolor = :white)
        return
    end

    shared(v, Δ) = (floor(minimum(v) / Δ) * Δ):Δ:(ceil(maximum(v) / Δ) * Δ)
    r = isnothing(df_mod) ? vcat(df.R0, df.R1) : vcat(df.R0, df.R1, df_mod.R0, df_mod.R1)
    edges = shared(r, Δr)
    τ_edges = shared(isnothing(df_mod) ? df.τ : vcat(df.τ, df_mod.τ), Δτ)

    axA = Axis(fig[1, 1]; xlabel = rich("R", subscript("1"), " ($(v_ref) V) / mΩ"), ylabel = "Share / %")
    overlay!(axA, df.R1, isnothing(df_mod) ? nothing : df_mod.R1; bins = edges, ymax = 0.25, ystep = 0.1)

    axC = Axis(fig[2, 1]; xlabel = rich("R", subscript("0"), " / mΩ"), ylabel = "Share / %")
    overlay!(axC, df.R0, isnothing(df_mod) ? nothing : df_mod.R0; bins = edges, ymax = 0.5, ystep = 0.2)
    linkxaxes!(axA, axC)

    axB = Axis(fig[1, 2]; xlabel = "τ / s", ylabel = "Share / %")
    overlay!(axB, df.τ, isnothing(df_mod) ? nothing : df_mod.τ; bins = τ_edges, ymax = 0.25, ystep = 0.1)

    # (D) the Arrhenius factor k(T) itself, unity at the reference temperature T0 = 25 °C
    axD = Axis(fig[2, 2]; xlabel = "Temperature / °C", ylabel = "Arrhenius factor k(T)")
    # k(T) = exp(k·(1/T − 1/T0)) is a closed form of the fitted `k`, not a separate quantity, so it
    # is rendered here rather than prepared upstream
    Tg = collect(T)
    Δ = 1 ./ (Tg .+ 273.15) .- 1 / (T0 + 273.15)
    kfac(ks) = [exp.(k .* Δ) for k in ks]
    fac = kfac(df.k)
    fac_mod = isnothing(df_mod) ? nothing : kfac(df_mod.k)

    for f in fac
        lines!(axD, Tg, f; color = (c_cell, 0.15), linewidth = 0.8)
    end
    if !isnothing(fac_mod)
        for f in fac_mod
            lines!(axD, Tg, f; color = (c_mod, 0.35), linewidth = 0.8)
        end
    end

    lines!(axD, Tg, [median(getindex.(fac, i)) for i in eachindex(Tg)]; color = c_cell_med, linewidth = 2)
    if !isnothing(fac_mod)
        lines!(
            axD, Tg, [median(getindex.(fac_mod, i)) for i in eachindex(Tg)];
            color = :black, linewidth = 2
        )
    end
    hlines!(axD, [1.0]; color = :black, linestyle = :dot, linewidth = 1)

    ylims!(axD, nothing, 2.5)
    xlims!(axD, extrema(Tg)...)
    axD.yticks = 1:1:2
    axD.yminorticks = IntervalsBetween(2)
    axD.yminorticksvisible = true

    entries = isnothing(fac_mod) ?
        ([[LineElement(color = (c_cell, 0.6)), LineElement(color = c_cell_med, linewidth = 2)]], ["Cell-level median"]) :
        (
            [
                [LineElement(color = (c_cell, 0.6)), LineElement(color = c_cell_med, linewidth = 2)],
                [LineElement(color = (c_mod, 0.7)), LineElement(color = :black, linewidth = 2)],
            ],
            ["Cell-level median", "Module-level median"],
        )
    axislegend(axD, entries...; position = :rt, framevisible = false, patchsize = (18, 10), rowgap = 0)

    for ax in (axA, axB, axC, axD)
        ax.xgridvisible = false
        ax.ygridvisible = false
        hidespines!(ax, :t, :r)
        ax.xminorticks = IntervalsBetween(2)
        ax.xminorticksvisible = true
    end
    if !isnothing(df_mod)
        axislegend(
            axA, [PolyElement(color = (c_cell, 0.85)), PolyElement(color = (c_mod, 0.55))],
            ["Cell-level", "Module-level"]; position = :rt, framevisible = false, patchsize = (16, 10), rowgap = 0,
        )
    end
    for (tag, ax) in (("A", axA), ("B", axB), ("C", axC), ("D", axD))
        text!(ax, 0.02, 0.98; text = tag, space = :relative, align = (:left, :top), font = :bold, fontsize = 20)
    end
    return fig
end
