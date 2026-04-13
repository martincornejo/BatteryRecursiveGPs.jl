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

    fig = Figure(size = (900, 700))
    ax1 = Axis(
        fig[1, 1]; ylabel = "Voltage / V",
        title = "Composite OCV from aligned cells (Module 7)"
    )
    ax2 = Axis(fig[2, 1]; ylabel = "dV/d$(xlabel)", xlabel)

    for i in eachindex(cells)
        q = cells[i].q
        v = cells[i].μ
        x = (q ./ Measurements.value(Q_cell[i]) .+ Measurements.value(s0[i])) .* xscale
        lines!(ax1, x, v; color = (:gray, 0.5), label = "Cell OCV")
        lines!(ax2, x[2:end], diff(v) ./ diff(x); color = (:gray, 0.3))
    end

    x_comp = soc_grid .* xscale
    lines!(ax1, x_comp, v_grid; color = :black, linewidth = 2, label = "Composite OCV")
    lines!(ax2, x_comp[2:end], diff(v_grid) ./ diff(x_comp); color = :black, linewidth = 2)

    axislegend(ax1; position = :cb, merge = true)
    linkxaxes!(ax1, ax2)
    return fig
end


function plot_ocv_residuals(fit, cells)
    (; soc_grid, v_grid, Q_cell, s0) = fit
    composite = LinearInterpolation(
        v_grid, soc_grid;
        extrapolation = ExtrapolationType.Constant,
    )

    fig = Figure(size = (900, 400))
    ax = Axis(
        fig[1, 1]; xlabel = "SOC", ylabel = "Residual / mV",
        title = "Per-cell OCV − composite OCV",
    )
    for i in eachindex(cells)
        soc = cells[i].q ./ Measurements.value(Q_cell[i]) .+ Measurements.value(s0[i])
        v = cells[i].μ
        mask = (soc .>= first(soc_grid)) .& (soc .<= last(soc_grid))
        r_mV = (v[mask] .- composite.(soc[mask])) .* 1000
        lines!(ax, soc[mask], r_mV; color = (:gray, 0.5), linewidth = 1)
    end
    hlines!(ax, [0.0]; color = :black, linestyle = :dot)
    return fig
end


function plot_ocv_extrapolation(composite; V_min = 2.9, V_max = 4.1)
    soc = collect(composite.t)
    v = collect(composite.u)
    ocv_extrap = LinearInterpolation(v, soc; extrapolation = ExtrapolationType.Linear)
    sv_extrap = invert_ocv(ocv_extrap; n_samples = 1000, extrapolation = ExtrapolationType.Linear)

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
    lines!(ax, collect(soc_data), composite.(soc_data), color = :black, linewidth = 2, label = "Measured")

    hlines!(ax, [V_min, V_max], color = :gray, linestyle = :dot)
    axislegend(ax; position = :rc)
    return fig
end
