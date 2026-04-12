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


function plot_composite_ocv(composite, ocvs, params)
    fig = Figure(size = (900, 700))
    ax1 = Axis(
        fig[1, 1], ylabel = "Voltage / V",
        title = "Composite OCV from aligned cells (Module 7)"
    )
    ax2 = Axis(fig[2, 1], ylabel = "dV/dQ / (V/Ah)", xlabel = "Capacity / Ah")

    Q_shift = first(composite.t)
    for (i, (f, p)) in enumerate(zip(ocvs, params))
        Q0, s = p
        q = range(first(f.t), last(f.t); length = 300)
        q_aligned = q .* s .+ Q0 .- Q_shift
        v = f.(q)
        lines!(ax1, q_aligned, v, color = (:gray, 0.4), label = "Cell OCV")
        lines!(ax2, q_aligned[1:(end - 1)], diff(v) ./ diff(q_aligned), color = (:gray, 0.4))
    end

    q_comp = range(first(composite.t), last(composite.t); length = 500)
    v_comp = composite.(q_comp)
    q_plot = q_comp .- Q_shift
    lines!(ax1, q_plot, v_comp, color = :black, linewidth = 2, label = "Composite OCV")
    lines!(ax2, q_plot[1:(end - 1)], diff(v_comp) ./ diff(q_plot), color = :black, linewidth = 2)

    axislegend(ax1; position = :cb, merge = true)
    linkxaxes!(ax1, ax2)
    return fig
end


function plot_ocv_residuals(composite, ocvs, params)
    Q_lo = first(composite.t)
    Q_hi = last(composite.t)
    Q_shift = Q_lo

    fig = Figure(size = (900, 400))
    ax = Axis(
        fig[1, 1]; xlabel = "Capacity / Ah", ylabel = "Residual / mV",
        title = "Per-cell OCV − composite OCV"
    )
    for (i, (f, p)) in enumerate(zip(ocvs, params))
        Q0, s = p
        q = range(first(f.t), last(f.t); length = 300)
        q_aligned = collect(q .* s .+ Q0)
        v_cell = f.(q)
        mask = (q_aligned .>= Q_lo) .& (q_aligned .<= Q_hi)
        r_mV = (v_cell[mask] .- composite.(q_aligned[mask])) .* 1000
        lines!(ax, q_aligned[mask] .- Q_shift, r_mV; color = (:gray, 0.5), linewidth = 1)
    end
    hlines!(ax, [0.0]; color = :black, linestyle = :dot)
    return fig
end


function plot_ocv_extrapolation(composite; V_min = 2.9, V_max = 4.1)
    q = collect(composite.t)
    v = collect(composite.u)
    Q_shift = first(composite.t)
    ocv_extrap = LinearInterpolation(v, q; extrapolation = ExtrapolationType.Linear)
    qv_extrap = invert_ocv(ocv_extrap; n_samples = 1000, extrapolation = ExtrapolationType.Linear)

    Q_at_Vmin = qv_extrap(V_min)
    Q_at_Vmax = qv_extrap(V_max)

    fig = Figure(size = (900, 500))
    ax = Axis(
        fig[1, 1], ylabel = "Voltage / V", xlabel = "Capacity / Ah",
        title = "OCV with linear extrapolation"
    )

    q_full = range(Q_at_Vmin, Q_at_Vmax; length = 500)
    lines!(ax, q_full .- Q_shift, ocv_extrap.(q_full), color = :red, linewidth = 2, linestyle = :dash, label = "Extrapolated")

    Q_data_min = first(composite.t)
    Q_data_max = last(composite.t)
    q_data = range(Q_data_min, Q_data_max; length = 500)
    lines!(ax, q_data .- Q_shift, composite.(q_data), color = :black, linewidth = 2, label = "Measured")

    hlines!(ax, [V_min, V_max], color = :gray, linestyle = :dot)
    axislegend(ax; position = :rc)
    return fig
end
