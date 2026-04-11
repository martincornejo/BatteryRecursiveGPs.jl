using QuackIO
using DataFrames
using DataInterpolations
using Printf
using Statistics
using Dates

using GLMakie

include("ocv.jl")

df_d = read_parquet(DataFrame, "data/yuasa-ocv-test/combined_log_20260131_200948.parquet")
df_c = read_parquet(DataFrame, "data/yuasa-ocv-test/combined_log_20260201_181000.parquet")


# === Plotting helpers ===

function plot_module_dataset(df, m)
    fig = Figure()
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
    Δt = [Dates.value.(diff(df_i.timestamp_utc)); 0] * 1e-3
    df_i[!, :q] = cumsum(df_i.i .* Δt) ./ 3600

    lines!(ax[2], df_i.timestamp_utc, df_i.i)
    lines!(ax[3], df_i.timestamp_utc, df_i.q)

    ax[1].ylabel = "Module Voltage / V"
    ax[2].ylabel = "Module Current / A"
    ax[3].ylabel = "Module Charge / Ah"

    linkxaxes!(ax...)
    fig
end


function compare_current_sources(df; m=7, tek_offset=0.0)
    df_bms = integrate_current(df; current_col="m$(m)_current", negate=true)
    df_tek = integrate_current(df; current_col="tek_m_cur_ref", negate=true, offset=tek_offset)

    fig = Figure(size=(900, 700))
    ax1 = Axis(fig[1, 1], ylabel="Current / A", title="BMS vs Oscilloscope Current (Module $(m))")
    ax2 = Axis(fig[2, 1], ylabel="Charge / Ah")
    ax3 = Axis(fig[3, 1], ylabel="Charge Error / Ah", xlabel="Time")

    lines!(ax1, df_bms.timestamp_utc, df_bms.i, label="BMS (m$(m)_current)")
    lines!(ax1, df_tek.timestamp_utc, df_tek.i, label="Oscilloscope (tek)")

    lines!(ax2, df_bms.timestamp_utc, df_bms.q, label="BMS")
    lines!(ax2, df_tek.timestamp_utc, df_tek.q, label="Oscilloscope")

    df_err = innerjoin(
        select(df_bms, :timestamp_utc, :q => :q_bms),
        select(df_tek, :timestamp_utc, :q => :q_tek),
        on=:timestamp_utc
    )
    df_err[!, :Δq] = df_err.q_bms .- df_err.q_tek
    lines!(ax3, df_err.timestamp_utc, df_err.Δq)

    linkxaxes!(ax1, ax2, ax3)
    Legend(fig[4, 1], ax2, orientation=:horizontal)

    return fig
end


# === Current sensor comparison ===

tek_offset_d = find_tek_offset(df_d)
tek_offset_c = find_tek_offset(df_c)
compare_current_sources(df_d; tek_offset=tek_offset_d) |> display
compare_current_sources(df_c; tek_offset=tek_offset_c) |> display


# === Build composite OCV ===
begin
    composite, ocvs, params, stats = build_composite_ocv(
        df_c, df_d; tek_offset_c, tek_offset_d)

    Q_composite = last(composite.t) - first(composite.t)
    @printf("\n=== Composite OCV (Q(V) Linear Regression) ===\n")
    @printf("%-6s %8s %8s %8s %8s %10s %8s %8s\n",
            "Cell", "Q0/Ah", "s", "C/Ah", "R2", "RMS_Q/Ah", "SE(s)", "SE(Q0)")
    for (i, (p, st)) in enumerate(zip(params, stats))
        C_i = Q_composite / p[2]
        @printf("Cell %2d: %6.2f  %8.4f  %6.2f  %6.4f  %8.4f  %8.4f  %8.4f\n",
                i, p[1], p[2], C_i, st.R2, st.rms_q, st.se_s, st.se_Q0)
    end

    ocv_smooth = smooth_ocv(composite)
end

# === Plot composite OCV with individual curves ===

let
    fig = Figure(size=(900, 700))
    ax1 = Axis(fig[1, 1], ylabel="Voltage / V",
               title="Composite OCV from aligned cells (Module 7)")
    ax2 = Axis(fig[2, 1], ylabel="dV/dQ / (V/Ah)", xlabel="Capacity / Ah")

    for (i, (f, p)) in enumerate(zip(ocvs, params))
        Q0, s = p
        q = range(first(f.t), last(f.t); length=300)
        q_aligned = q .* s .+ Q0
        v = f.(q)
        lines!(ax1, q_aligned, v, color=(:gray, 0.4))
        lines!(ax2, q_aligned[1:end-1], diff(v) ./ diff(q_aligned), color=(:gray, 0.4))
    end

    q_comp = range(first(composite.t), last(composite.t); length=500)
    v_comp = composite.(q_comp)
    v_smooth = ocv_smooth.(q_comp)
    lines!(ax1, q_comp, v_comp, color=:black, linewidth=2, label="Composite OCV")
    lines!(ax1, q_comp, v_smooth, color=:red, linewidth=2, label="Smoothed OCV")
    lines!(ax2, q_comp[1:end-1], diff(v_comp) ./ diff(q_comp), color=:black, linewidth=2)
    lines!(ax2, q_comp[1:end-1], diff(v_smooth) ./ diff(q_comp), color=:red, linewidth=2)

    axislegend(ax1)
    linkxaxes!(ax1, ax2)
    fig
end


# === Estimate SOC range covered by the measurement ===

let
    q = collect(composite.t)
    v = collect(composite.u)
    ocv_extrap = LinearInterpolation(v, q; extrapolation=ExtrapolationType.Linear)

    qv_extrap = invert_ocv(ocv_extrap; n_samples=1000, extrapolation=ExtrapolationType.Linear)

    V_min, V_max = 2.75, 4.1
    Q_at_Vmin = qv_extrap(V_min)
    Q_at_Vmax = qv_extrap(V_max)
    Q_full = Q_at_Vmax - Q_at_Vmin

    Q_data_min = first(composite.t)
    Q_data_max = last(composite.t)
    Q_data = Q_data_max - Q_data_min

    @printf("\nSOC range estimation (linear extrapolation):\n")
    @printf("  Data covers:    %.1f - %.1f Ah (%.1f Ah)\n", Q_data_min, Q_data_max, Q_data)
    @printf("  Full range:     %.1f - %.1f Ah (%.1f Ah) [%.2fV - %.2fV]\n",
            Q_at_Vmin, Q_at_Vmax, Q_full, V_min, V_max)
    @printf("  SOC range used: %.1f%% - %.1f%%\n",
            100 * (Q_data_min - Q_at_Vmin) / Q_full,
            100 * (Q_data_max - Q_at_Vmin) / Q_full)

    fig = Figure(size=(900, 500))
    ax = Axis(fig[1, 1], ylabel="Voltage / V", xlabel="Capacity / Ah",
              title="OCV with linear extrapolation")

    q_full = range(Q_at_Vmin, Q_at_Vmax; length=500)
    lines!(ax, q_full, ocv_extrap.(q_full), color=:red, linewidth=2, linestyle=:dash, label="Extrapolated")

    q_data = range(Q_data_min, Q_data_max; length=500)
    lines!(ax, q_data, composite.(q_data), color=:black, linewidth=2, label="Measured")

    hlines!(ax, [V_min, V_max], color=:gray, linestyle=:dot)
    axislegend(ax; position=:rc)
    fig
end
