function plot_module_dataset(df, m)
    fig = Figure(size = (900, 700))
    ax = [Axis(fig[i, 1]) for i in 1:3]

    # cell voltages
    for c in 1:12
        cell = @sprintf("%02d", c)
        df_cell = select(df, :timestamp_utc, "m$(m)_cell$cell" => :v)
        dropmissing!(df_cell)
        lines!(ax[1], df_cell.timestamp_utc, df_cell.v)
    end

    # current and coulomb counting
    df_i = select(df, :timestamp_utc, "m$(m)_current" => (-) => :i)
    dropmissing!(df_i)
    Δt = [Dates.value.(diff(df_i.timestamp_utc)); 0] * 1.0e-3
    df_i[!, :q] = cumsum(df_i.i .* Δt) ./ 3600

    lines!(ax[2], df_i.timestamp_utc, df_i.i)
    lines!(ax[3], df_i.timestamp_utc, df_i.q)

    ax[1].ylabel = "Module Voltage / V"
    ax[2].ylabel = "Module Current / A"
    ax[3].ylabel = "Module Charge / Ah"

    linkxaxes!(ax...)
    return fig
end

function compare_current_sources(df; m = 7, tek_offset = 0.0)
    df_bms = integrate_current(df; current_col = "m$(m)_current", negate = true)
    df_tek = integrate_current(df; current_col = "tek_m_cur_ref", negate = true, offset = tek_offset)

    fig = Figure(size = (900, 700))
    ax1 = Axis(fig[1, 1], ylabel = "Current / A", title = "BMS vs Oscilloscope Current (Module $(m))")
    ax2 = Axis(fig[2, 1], ylabel = "Charge / Ah")
    ax3 = Axis(fig[3, 1], ylabel = "Charge Error / Ah", xlabel = "Time")

    lines!(ax1, df_bms.timestamp_utc, df_bms.i, label = "BMS (m$(m)_current)")
    lines!(ax1, df_tek.timestamp_utc, df_tek.i, label = "Oscilloscope (tek)")

    lines!(ax2, df_bms.timestamp_utc, df_bms.q, label = "BMS")
    lines!(ax2, df_tek.timestamp_utc, df_tek.q, label = "Oscilloscope")

    df_err = innerjoin(
        select(df_bms, :timestamp_utc, :q => :q_bms),
        select(df_tek, :timestamp_utc, :q => :q_tek),
        on = :timestamp_utc
    )
    df_err[!, :Δq] = df_err.q_bms .- df_err.q_tek
    lines!(ax3, df_err.timestamp_utc, df_err.Δq)

    linkxaxes!(ax1, ax2, ax3)
    Legend(fig[4, 1], ax2, orientation = :horizontal)

    return fig
end


function plot_composite_ocv(fit, cells; xaxis = :soc)
    (; soc_grid, v_grid, Q_cell, s0, Q_full) = fit
    xscale = xaxis == :ah ? Q_full : 1.0
    xlabel = xaxis == :ah ? "Capacity / Ah" : "SOC"

    fig = Figure()
    ax1 = Axis(fig[1, 1]; ylabel = "Voltage / V")
    ax2 = Axis(fig[2, 1]; ylabel = "dV/d$(xlabel)", xlabel)

    Q_cell_μ = Measurements.value.(Q_cell)
    s0_μ = Measurements.value.(s0)
    # colorrange = (minimum(Q_cell_μ), maximum(Q_cell_μ))
    colormap = :lipari
    colorrange = (65, 87)
    lowclip = :black
    for i in eachindex(cells)
        q = cells[i].q
        v = cells[i].μ
        x = (q ./ Q_cell_μ[i] .+ s0_μ[i]) .* xscale
        lines!(ax1, x, v; color = Q_cell_μ[i], colorrange, colormap, alpha = 0.6)
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

    hidexdecorations!(ax1; grid = false, ticks = false)

    axislegend(ax1; position = :cb)
    Colorbar(fig[1:2, 2]; limits = colorrange, colormap, lowclip, label = "Cell Capacity / Ah")
    linkxaxes!(ax1, ax2)
    return fig
end


function plot_ocv_residuals(fit, cells)
    (; soc_grid, v_grid, Q_cell, s0) = fit
    order = sortperm(soc_grid)
    composite = LinearInterpolation(
        v_grid[order], soc_grid[order];
        extrapolation = ExtrapolationType.Constant,
    )

    fig = Figure(size = (900, 400))
    ax = Axis(
        fig[1, 1]; xlabel = "SOC", ylabel = "Residual / mV",
        title = "Per-cell OCV − composite OCV",
    )
    Q_cell_μ = Measurements.value.(Q_cell)
    s0_μ = Measurements.value.(s0)
    colorrange = (58, 70)
    for i in eachindex(cells)
        soc = cells[i].q ./ Q_cell_μ[i] .+ s0_μ[i]
        v = cells[i].μ
        mask = (soc .>= first(soc_grid)) .& (soc .<= last(soc_grid))
        r_mV = (v[mask] .- composite.(soc[mask])) .* 1000
        lines!(ax, soc[mask], r_mV; color = Q_cell_μ[i], alpha = 0.5, colorrange, linewidth = 1)
    end
    hlines!(ax, [0.0]; color = :black, linestyle = :dot)
    Colorbar(fig[1, 2], limits = colorrange, label = "Cell Capacity / Ah")
    return fig
end


function plot_reference_ocv(ref_v_grid, ref_soc_grid, cells, ref_fit; title = "Cells aligned to reference OCV")
    (; Q_cell, s0) = ref_fit

    fig = Figure(size = (900, 700))
    ax1 = Axis(fig[1, 1]; ylabel = "Voltage / V", title)
    ax2 = Axis(fig[2, 1]; xlabel = "SOC", ylabel = "dV/dSOC")

    Q_μ = Measurements.value.(Q_cell)
    s0_μ = Measurements.value.(s0)
    colorrange = extrema(Q_μ)
    colormap = :viridis
    for i in eachindex(cells)
        q = cells[i].q
        v = cells[i].μ
        soc = q ./ Q_μ[i] .+ s0_μ[i]
        lines!(ax1, soc, v; color = Q_μ[i], colorrange, colormap, alpha = 0.6)
        lines!(ax2, soc[2:end], diff(v) ./ diff(soc); color = Q_μ[i], colorrange, colormap, alpha = 0.4)
    end

    order = sortperm(ref_soc_grid)
    itp = PCHIPInterpolation(ref_v_grid[order], ref_soc_grid[order])
    soc_dense = range(ref_soc_grid[order[1]], ref_soc_grid[order[end]]; length = 500)
    dv_dsoc_raw = [DataInterpolations.derivative(itp, s) for s in soc_dense]
    hw = 15
    n = length(dv_dsoc_raw)
    dv_dsoc = [mean(dv_dsoc_raw[max(1, i - hw):min(n, i + hw)]) for i in 1:n]

    lines!(ax1, ref_soc_grid, ref_v_grid; color = :black, linewidth = 2, label = "Reference OCV")
    lines!(ax2, collect(soc_dense), dv_dsoc; color = :black, linewidth = 2)
    ylims!(ax2, 0, nothing)

    axislegend(ax1; position = :cb)
    Colorbar(fig[1:2, 2]; limits = colorrange, colormap, label = "Cell Capacity / Ah")
    linkxaxes!(ax1, ax2)
    return fig
end


function plot_ocv_residuals_ref(ref_v_grid, ref_soc_grid, cells, ref_fit; title = "Per-cell OCV − reference OCV")
    (; Q_cell, s0) = ref_fit
    order = sortperm(ref_soc_grid)
    composite = LinearInterpolation(
        ref_v_grid[order], ref_soc_grid[order];
        extrapolation = ExtrapolationType.Constant,
    )

    fig = Figure(size = (900, 400))
    ax = Axis(fig[1, 1]; xlabel = "SOC", ylabel = "Residual / mV", title)
    Q_μ = Measurements.value.(Q_cell)
    s0_μ = Measurements.value.(s0)
    # colorrange = extrema(Q_μ)
    colorrange = (75, 85)
    soc_lo, soc_hi = extrema(ref_soc_grid)
    for i in eachindex(cells)
        soc = cells[i].q ./ Q_μ[i] .+ s0_μ[i]
        v = cells[i].μ
        mask = (soc .>= soc_lo) .& (soc .<= soc_hi)
        r_mV = (v[mask] .- composite.(soc[mask])) .* 1000
        lines!(ax, soc[mask], r_mV; color = Q_μ[i], alpha = 0.5, colorrange, linewidth = 1)
    end
    hlines!(ax, [0.0]; color = :black, linestyle = :dot)
    Colorbar(fig[1, 2]; limits = colorrange, label = "Cell Capacity / Ah")
    return fig
end


function plot_ocv_residuals_by_module(fit, cells, ids)
    (; soc_grid, v_grid, Q_cell, s0) = fit
    order = sortperm(soc_grid)
    composite = LinearInterpolation(
        v_grid[order], soc_grid[order];
        extrapolation = ExtrapolationType.Constant,
    )

    fig = Figure(size = (900, 400))
    ax = Axis(
        fig[1, 1]; xlabel = "SOC", ylabel = "Residual / mV",
        title = "Per-cell OCV − composite OCV",
    )
    Q_cell_μ = Measurements.value.(Q_cell)
    s0_μ = Measurements.value.(s0)
    modules = sort(unique(id.m for id in ids))
    palette = cgrad(:tab10, length(modules); categorical = true)
    color_of = Dict(m => palette[k] for (k, m) in enumerate(modules))
    for i in eachindex(cells)
        soc = cells[i].q ./ Q_cell_μ[i] .+ s0_μ[i]
        v = cells[i].μ
        mask = (soc .>= first(soc_grid)) .& (soc .<= last(soc_grid))
        r_mV = (v[mask] .- composite.(soc[mask])) .* 1000
        lines!(ax, soc[mask], r_mV; color = color_of[ids[i].m], alpha = 0.5, linewidth = 1)
    end
    hlines!(ax, [0.0]; color = :black, linestyle = :dot)
    elems = [LineElement(; color = color_of[m]) for m in modules]
    Legend(fig[1, 2], elems, string.(modules), "Module")
    return fig
end


function plot_ocv_extrapolation(composite; V_min = 2.9, V_max = 4.1)
    soc = collect(composite.t)
    v = collect(composite.u)
    ocv_extrap = CubicSpline(v, soc; extrapolation = ExtrapolationType.Extension)
    sv_extrap = invert_ocv(ocv_extrap; n_samples = 100, extrapolation = ExtrapolationType.Extension)

    soc_at_Vmin = sv_extrap(V_min)
    soc_at_Vmax = sv_extrap(V_max)

    fig = Figure(size = (900, 500))
    ax = Axis(
        fig[1, 1], ylabel = "Voltage / V", xlabel = "SOC",
        title = "OCV with linear extrapolation"
    )

    soc_full = range(soc_at_Vmin, soc_at_Vmax; length = 500)
    lines!(ax, collect(soc_full), ocv_extrap.(soc_full), color = :red, linewidth = 2, linestyle = :dash, label = "Extrapolated")

    soc_data = range(first(soc), last(soc); length = 500)
    scatterlines!(ax, collect(soc_data), composite.(soc_data), color = :black, linewidth = 2, label = "Measured")

    hlines!(ax, [V_min, V_max], color = :gray, linestyle = :dot)
    ylims!(ax, V_min, V_max)
    axislegend(ax; position = :rc)
    return fig
end
