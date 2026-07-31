# Validation of the reconstructed cell OCV curves (RGP-ECM, this dataset) against the
# oscilloscope-measured OCV from the low-power cycling experiment (data/ocv-test).
#
# The validated module carries the accurate current probe: it is rig module 7 in the
# yuasa-ocv test and P1M9 (p=1, m=9) here. Cell index c refers to the same physical
# cell in both datasets.
#
# This file is self-contained: measured-OCV builders (parse the rig parquet),
# composite-decomposition validation, and the validation plots.


# === measured OCV builders (low-power charge/discharge, oscilloscope current) ===

function integrate_current(df; current_col, negate = true, offset = 0.0)
    transform = negate ? (x -> -(x .+ offset)) : (x -> x .+ offset)
    df_i = select(df, :timestamp_utc, current_col => transform => :i)
    dropmissing!(df_i)
    Δt = [Dates.value.(diff(df_i.timestamp_utc)); 0] * 1.0e-3  # in seconds
    df_i[!, :q] = cumsum(df_i.i .* Δt) ./ 3600  # in Ah
    return df_i
end

function find_tek_offset(df; m = 7, bms_thresh = 0.1)
    df_both = select(df, :timestamp_utc, "m$(m)_current" => :bms, :tek_m_cur_ref => :tek)
    dropmissing!(df_both)
    # Only use rest periods where BMS reads ~0; offset so that tek + offset ≈ 0 at rest
    df_rest = subset(df_both, :bms => ByRow(x -> abs(x) < bms_thresh))
    return -mean(df_rest.tek)
end

function invert_ocv(f; n_samples = 500, extrapolation = ExtrapolationType.Constant)
    q = range(first(f.t), last(f.t); length = n_samples)
    v = f.(q)
    # Ensure strict monotonicity: remove duplicate/decreasing V values
    mask = [true; diff(v) .> 0]
    return LinearInterpolation(q[mask], v[mask]; extrapolation)
end

# Midline OCV by averaging the charge and discharge branches in the VOLTAGE frame:
# at each voltage, the mean of the two branches' charge, q̄(v) = (q_chg(v)+q_dch(v))/2.
# Averaging at fixed voltage (not fixed charge) keeps the high-SOC end that q-frame
# averaging clips — the q-frame average is restricted to the charge range BOTH
# branches share, which (because the two experiments are offset in charge) truncates
# the top of the curve ~20 mV below either branch. The unknown charge offset between
# the experiments only shifts the (irrelevant) q origin here; it does not distort the
# shape. Returns (; q, μ) with μ the voltage grid and q the averaged charge.
function average_charge_discharge(fc, fd; n_samples = 500)
    gc = invert_ocv(fc)  # q(v), charge branch
    gd = invert_ocv(fd)  # q(v), discharge branch
    v_lo = max(first(gc.t), first(gd.t))
    v_hi = min(last(gc.t), last(gd.t))
    v = collect(range(v_lo, v_hi; length = n_samples))
    q = (gc.(v) .+ gd.(v)) ./ 2
    return (; q, μ = v)
end

function clean_ocv(df, id; dch::Bool, i_thresh = 0.5, current_col = "bms")
    (; m, c) = id

    cell = @sprintf("%02d", c)
    df_cell = select(df, :timestamp_utc, "m$(m)_cell$cell" => :v)
    dropmissing!(df_cell)

    if current_col == "tek"
        df_i = integrate_current(df; current_col = "tek_m_cur_ref", negate = true, offset = find_tek_offset(df; m))
    else
        df_i = integrate_current(df; current_col = "m$(m)_current", negate = true)
    end

    q0 = minimum(df_i.q)
    df_i[!, :q] = df_i.q .- q0

    df1 = innerjoin(df_cell, df_i, on = :timestamp_utc, makeunique = true)
    sort!(df1, :timestamp_utc)

    df2 = subset(df1, :i => ByRow(<=(i_thresh) ∘ abs))
    df2[!, :q_floor] = floor.(df2.q; digits = 1)

    df3 = combine(groupby(df2, :q_floor)) do subdf
        sort(subdf, :v; rev = dch) |> first
    end
    sort!(df3, :q)

    return LinearInterpolation(df3.v, df3.q; extrapolation = ExtrapolationType.Constant)
end


# === individual-cell OCV validation (RGP-OCV vs measured low-power OCV) ===
# `measured`/`reconstructed` are vectors of monotone v(q) curves `(; q, μ)` (Ah, V), one
# per cell in matching order (rig module 7 ≡ P1M9; cell index identical).
#
# Validation uses the composite-OCV DECOMPOSITION, NOT a per-cell affine fit. A per-cell
# fit conflates SOH, SOC, OCV shape and a cross-setup voltage/current calibration over
# inconsistent per-cell windows, which manufactures spurious systematic residuals (an
# S-shape, a uniform ~2% capacity bias, and apparent anomalies for low-SOC cells). Each
# dataset is instead decomposed by `fit_composite_ocv` into a shared OCV shape + per-cell
# `Q_cell` (SOH) and `s0` (SOC); the two datasets are aligned by ONE global gauge
# (soc_meas ≈ α·soc_rgp + β); and each cell is placed on the common absolute-SOC axis via
# its own (Q_cell, s0) — no per-cell fitting. Validation is then three separable,
# gauge-robust quantities: shared shape, per-cell SOH, per-cell SOC. Absolute capacity
# (~2%, current-sensor calibration) and absolute voltage (~mV) are unidentifiable
# nuisances, reported only relatively.

"""
    validate_cell_ocvs(measured, reconstructed) -> (; df, gauge, meas_comp, rgp_comp)

Composite-decomposition validation of individual-cell OCVs. Decomposes each dataset with
`fit_composite_ocv` (shared shape + per-cell `Q_cell`/`s0`), aligns the two datasets by one
global gauge `soc_meas ≈ α·soc_rgp + β`, and places every cell on the common absolute-SOC
axis via its own `(Q_cell, s0)` — no per-cell fitting. `df` has, per cell: SOH (`soh_*` =
`Q_cell`), SOC (`s0_*`), and the OCV residual on the common gauge (`ocv_rmse`, `ocv_max`,
mV). Also returns the global gauge and the two composite fits.
"""
function validate_cell_ocvs(measured, reconstructed)
    mf = fit_composite_ocv(measured; uq = false)
    rf = fit_composite_ocv(reconstructed; uq = false)

    # one global gauge between the datasets: soc_meas_comp(v) ≈ α·soc_rgp_comp(v) + β
    scm = LinearInterpolation(mf.soc_grid, mf.v_grid; extrapolation = ExtrapolationType.Constant)
    scr = LinearInterpolation(rf.soc_grid, rf.v_grid; extrapolation = ExtrapolationType.Constant)
    v_lo = max(minimum(mf.v_grid), minimum(rf.v_grid)) + 1.0e-3
    v_hi = min(maximum(mf.v_grid), maximum(rf.v_grid)) - 1.0e-3
    v = collect(range(v_lo, v_hi; length = 200))
    α, β = hcat(scr.(v), ones(length(v))) \ scm.(v)

    Qm = Measurements.value.(mf.Q_cell); s0m = Measurements.value.(mf.s0)
    Qr = Measurements.value.(rf.Q_cell); s0r = Measurements.value.(rf.s0)

    df = map(eachindex(measured)) do c
        # place each cell on the common (measured-gauge) absolute SOC — no per-cell fit
        socm = s0m[c] .+ collect(measured[c].q) ./ Qm[c]
        socr = α .* (s0r[c] .+ collect(reconstructed[c].q) ./ Qr[c]) .+ β
        vmf = LinearInterpolation(collect(measured[c].μ), socm; extrapolation = ExtrapolationType.Constant)
        vrf = LinearInterpolation(collect(reconstructed[c].μ), socr; extrapolation = ExtrapolationType.Constant)
        lo = max(minimum(socm), minimum(socr)) + 0.002
        hi = min(maximum(socm), maximum(socr)) - 0.002
        sg = collect(range(lo, hi; length = 150))
        r = (vrf.(sg) .- vmf.(sg)) .* 1000
        (;
            cell = c, soh_meas = Qm[c], soh_rgp = Qr[c], s0_meas = s0m[c], s0_rgp = s0r[c],
            ocv_rmse = sqrt(mean(abs2, r)), ocv_max = maximum(abs, r),
        )
    end |> DataFrame

    return (; df, gauge = (; α, β), meas_comp = mf, rgp_comp = rf)
end

function eval_cell_ocv_validation(res)
    df = res.df
    soh_m = df.soh_meas ./ mean(df.soh_meas); soh_r = df.soh_rgp ./ mean(df.soh_rgp)
    @printf("\n=== Individual-cell OCV validation (composite decomposition) ===\n")
    @printf("%-8s %9s %9s   %8s %8s   %11s\n", "Cell", "SOH_meas%", "SOH_rgp%", "s0_meas", "s0_rgp", "OCV rmse/mV")
    for r in eachrow(df)
        @printf(
            "Cell %2d: %+9.2f %+9.2f   %8.3f %8.3f   %11.2f\n",
            r.cell, (soh_m[r.cell] - 1) * 100, (soh_r[r.cell] - 1) * 100, r.s0_meas, r.s0_rgp, r.ocv_rmse
        )
    end
    return @printf(
        "\nSOH corr(RGP,meas)=%.2f | SOC corr=%.2f | per-cell OCV residual median=%.2f mV\n",
        cor(soh_r, soh_m), cor(df.s0_rgp, df.s0_meas), median(df.ocv_rmse)
    )
end


# === validation plots ===

# Consumes `validate_cell_ocvs(measured, reconstructed)` + the raw curves. Places each
# cell on the common absolute-SOC gauge (its own Q_cell/s0 + the global α,β) and shows:
# per-cell OCV overlay (measured vs RGP), per-cell residual RMS, and per-cell SOH & SOC.
# (Basic layout — refine later.)
function plot_cell_ocv_validation(res, measured, reconstructed)
    df = res.df
    (; α, β) = res.gauge
    cells = df.cell
    Qm = df.soh_meas; Qr = df.soh_rgp
    s0m = df.s0_meas; s0r = df.s0_rgp

    # per-cell absolute-SOC placement (measured gauge) and OCV interpolants soc → V
    socm(c) = s0m[c] .+ collect(measured[c].q) ./ Qm[c]
    socr(c) = α .* (s0r[c] .+ collect(reconstructed[c].q) ./ Qr[c]) .+ β
    vmf(c) = LinearInterpolation(collect(measured[c].μ), socm(c); extrapolation = ExtrapolationType.Constant)
    vrf(c) = LinearInterpolation(collect(reconstructed[c].μ), socr(c); extrapolation = ExtrapolationType.Constant)

    # per-cell OCV residual (RGP − measured), each over ITS OWN measured∩RGP SOC overlap
    res_curves = map(cells) do c
        scm_c = socm(c); scr_c = socr(c)
        lo = max(minimum(scm_c), minimum(scr_c)) + 0.002
        hi = min(maximum(scm_c), maximum(scr_c)) - 0.002
        s = collect(range(lo, hi; length = 200))
        (; soc = s, r = (vrf(c).(s) .- vmf(c).(s)) .* 1000)
    end
    rms_mV = median(df.ocv_rmse)

    c_meas = :gray60                # measured reference (neutral gray)
    c_rgp = Makie.wong_colors()[1]  # RGP-ECM (single accent)
    soh_m = (Qm ./ mean(Qm) .- 1) .* 100   # SOH deviation from module mean, %
    soh_r = (Qr ./ mean(Qr) .- 1) .* 100
    Δs0_m = (s0m .- mean(s0m)) .* 100      # initial-SOC offset from module mean, % (centered gauges)
    Δs0_r = (s0r .- mean(s0r)) .* 100

    fig = Figure(size = (700, 560), fontsize = 12)

    # left column (curves): (A) OCV overlay over (B) residual, sharing the SOC axis
    axA = Axis(fig[1, 1]; xlabel = "SOC / %", ylabel = "OCV / V", xgridvisible = false, ygridvisible = false)
    for c in cells
        lines!(axA, socm(c) .* 100, collect(measured[c].μ); color = c_meas, linewidth = 1.1)
        lines!(axA, socr(c) .* 100, collect(reconstructed[c].μ); color = c_rgp, linewidth = 1.3, linestyle = :dash)
    end
    axislegend(
        axA, [LineElement(color = c_meas), LineElement(color = :black, linestyle = :dash)],
        ["Measured", "RGP-ECM"]; position = :rb, framevisible = false, labelsize = 11,
    )
    axA.yticks = 3.2:0.2:4.2
    axA.yminorticks = IntervalsBetween(2)
    axA.yminorticksvisible = true

    axB = Axis(fig[2, 1]; xlabel = "SOC / %", ylabel = "ΔOCV / mV", xgridvisible = false, ygridvisible = false)
    for rc in res_curves
        lines!(axB, rc.soc .* 100, rc.r; color = (c_rgp, 0.6), linewidth = 1.3)
    end
    hlines!(axB, [0]; color = :black, linestyle = :dot)
    ylims!(axB, -20, 50)   # show the high-SOC rise (~+15 mV); the steep low-SOC knee clips off-panel
    text!(axB, 0.5, 0.98; text = "median RMS $(round(rms_mV; digits = 1)) mV", space = :relative, align = (:center, :top), fontsize = 11)
    axB.yticks = -20:20:60
    axB.yminorticks = IntervalsBetween(2)
    axB.yminorticksvisible = true
    for ax in (axA, axB)
        ax.xticks = 0:20:100
        ax.xminorticks = IntervalsBetween(2)
        ax.xminorticksvisible = true
    end
    linkxaxes!(axA, axB)

    # right column (scatters): (C) SOH parity, (D) initial-SOC parity — square, true 1:1
    Lh = 1.15 * maximum(abs, vcat(soh_m, soh_r))
    axC = Axis(fig[1, 2]; xlabel = "measured ΔSOH / %", ylabel = "RGP ΔSOH / %", aspect = 1, limits = (-Lh, Lh, -Lh, Lh), xgridvisible = false, ygridvisible = false)
    ablines!(axC, 0, 1; color = :gray, linestyle = :dash)
    scatter!(axC, soh_m, soh_r; color = c_rgp, markersize = 12, strokewidth = 0.5, strokecolor = :white)
    text!(axC, 0.05, 0.95; text = "r = $(round(cor(soh_r, soh_m); digits = 2))", space = :relative, align = (:left, :top), fontsize = 12)
    axC.xminorticksvisible = true
    axC.yminorticksvisible = true

    Ls = 1.15 * maximum(abs, vcat(Δs0_m, Δs0_r))
    axD = Axis(fig[2, 2]; xlabel = "measured ΔSOC / %", ylabel = "RGP ΔSOC / %", aspect = 1, limits = (-Ls, Ls, -Ls, Ls), xgridvisible = false, ygridvisible = false)
    ablines!(axD, 0, 1; color = :gray, linestyle = :dash)
    scatter!(axD, Δs0_m, Δs0_r; color = c_rgp, markersize = 12, strokewidth = 0.5, strokecolor = :white)
    text!(axD, 0.05, 0.95; text = "r = $(round(cor(s0r, s0m); digits = 2))", space = :relative, align = (:left, :top), fontsize = 12)
    axD.xminorticksvisible = true
    axD.yminorticksvisible = true
    axD.xminorticks = IntervalsBetween(5)
    axD.yminorticks = IntervalsBetween(5)

    for ax in (axA, axB, axC, axD)
        hidespines!(ax, :t, :r)
    end
    Label(fig[1, 1, TopLeft()], "A"; fontsize = 20, font = :bold, padding = (0, 0, 5, 0))
    Label(fig[2, 1, TopLeft()], "B"; fontsize = 20, font = :bold, padding = (0, 0, 5, 0))
    Label(fig[1, 2, TopLeft()], "C"; fontsize = 20, font = :bold, padding = (0, 0, 5, 0))
    Label(fig[2, 2, TopLeft()], "D"; fontsize = 20, font = :bold, padding = (0, 0, 5, 0))
    colsize!(fig.layout, 1, Relative(0.6))
    return fig
end


# === measured-OCV experiment diagnostics ===

# Current-sensor cross-check on the rig data: the BMS module current vs the oscilloscope
# (tek) reference probe, their integrated charge, and the running charge error between them.
function compare_current_sources(df; m = 7)
    df_bms = integrate_current(df; current_col = "m$(m)_current", negate = true)
    df_tek = integrate_current(df; current_col = "tek_m_cur_ref", negate = true, offset = find_tek_offset(df; m))

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

# Build the linearly-extrapolated OCV interpolants from a composite SOC → V curve
# (`composite.t` = SOC grid, `composite.u` = V grid, e.g.
# `LinearInterpolation(fit.v_grid, fit.soc_grid)`). Extends the curve linearly beyond the
# observed data so any voltage maps to an SOC and vice versa. Returns `(; v_of_soc, soc_of_v)`,
# two interpolants that are mutual inverses: `v_of_soc` is SOC → V, `soc_of_v` is V → SOC.
# Single source of truth for both `eval_soc_range` and `plot_ocv_extrapolation`.
function extrapolate_ocv(composite; n_samples = 100)
    soc = collect(composite.t)
    v = collect(composite.u)
    v_of_soc = LinearInterpolation(v, soc; extrapolation = ExtrapolationType.Linear)        # SOC → V
    soc_of_v = invert_ocv(v_of_soc; n_samples, extrapolation = ExtrapolationType.Linear)    # V → SOC
    return (; v_of_soc, soc_of_v)
end

# Estimate the usable SOC window of the measured composite by linearly extrapolating the
# OCV to the full voltage window [V_min, V_max]. Returns the extrapolated interpolants and
# the SOC bounds — `soc_of_v` maps any voltage to its SOC on the extrapolated composite and
# is exactly the `ref_soc_of_v` reference that `fit_cells_to_reference` consumes to scale the
# reconstructed cell OCVs onto this full-window gauge.
function eval_soc_range(composite; V_min = 2.9, V_max = 4.1)
    (; v_of_soc, soc_of_v) = extrapolate_ocv(composite)
    soc = collect(composite.t)
    v = collect(composite.u)

    soc_at_Vmin = soc_of_v(V_min)
    soc_at_Vmax = soc_of_v(V_max)
    soc_full = soc_at_Vmax - soc_at_Vmin

    soc_data_min = first(soc)
    soc_data_max = last(soc)
    soc_data = soc_data_max - soc_data_min

    V_data_min = first(v)
    V_data_max = last(v)

    @printf("\nSOC range estimation (linear extrapolation):\n")
    @printf(
        "  Data covers:    %.3f - %.3f (%.3f) [%.2fV - %.2fV]\n",
        soc_data_min, soc_data_max, soc_data, V_data_min, V_data_max
    )
    @printf(
        "  Full range:     %.3f - %.3f (%.3f) [%.2fV - %.2fV]\n",
        soc_at_Vmin, soc_at_Vmax, soc_full, V_min, V_max
    )
    @printf(
        "  SOC range used: %.1f%% - %.1f%%\n",
        100 * (soc_data_min - soc_at_Vmin) / soc_full,
        100 * (soc_data_max - soc_at_Vmin) / soc_full
    )
    return (; v_of_soc, soc_of_v, soc_at_Vmin, soc_at_Vmax, soc_full)
end

function plot_ocv_extrapolation(composite; V_min = 2.9, V_max = 4.1)
    (; v_of_soc, soc_of_v) = extrapolate_ocv(composite)
    soc = collect(composite.t)

    soc_at_Vmin = soc_of_v(V_min)
    soc_at_Vmax = soc_of_v(V_max)

    fig = Figure(size = (900, 500))
    ax = Axis(
        fig[1, 1], ylabel = "Voltage / V", xlabel = "SOC",
        title = "OCV with linear extrapolation"
    )

    soc_full = range(soc_at_Vmin, soc_at_Vmax; length = 500)
    lines!(ax, collect(soc_full), v_of_soc.(soc_full), color = :red, linewidth = 2, linestyle = :dash, label = "Extrapolated")

    soc_data = range(first(soc), last(soc); length = 500)
    scatterlines!(ax, collect(soc_data), composite.(soc_data), color = :black, linewidth = 2, label = "Measured")

    hlines!(ax, [V_min, V_max], color = :gray, linestyle = :dot)
    ylims!(ax, V_min, V_max)
    axislegend(ax; position = :rc)
    return fig
end

# Diagnostic: overlay the cleaned OCV (clean_ocv) on the raw (q, v) measurement for one
# cell/branch — shows how the per-bin extraction tracks the relaxed points within the
# noisy loaded data. `q` shares clean_ocv's frame (current-integrated, zeroed at min).
function plot_ocv_cleaning(df, id; dch::Bool, current_col = "tek", i_thresh = 0.5)
    col = current_col == "tek" ? "tek_m_cur_ref" : "m$(id.m)_current"
    offset = current_col == "tek" ? find_tek_offset(df; m = id.m) : 0.0
    df_i = integrate_current(df; current_col = col, negate = true, offset)
    df_i[!, :q] = df_i.q .- minimum(df_i.q)
    cell = @sprintf("%02d", id.c)
    raw = innerjoin(select(df, :timestamp_utc, "m$(id.m)_cell$cell" => :v), df_i, on = :timestamp_utc, makeunique = true)
    f = clean_ocv(df, id; dch, i_thresh, current_col)  # interp knots: f.t=q grid, f.u=v

    wong = Makie.wong_colors()
    fig = Figure(size = (800, 500))
    ax = Axis(
        fig[1, 1]; xlabel = "Charge / Ah", ylabel = "Voltage / V",
        title = "OCV cleaning — cell m$(id.m) c$(id.c) ($(dch ? "discharge" : "charge"))",
        xgridvisible = false, ygridvisible = false,
    )
    scatter!(ax, raw.q, raw.v; color = (:gray, 0.3), markersize = 4, label = "raw")
    lines!(ax, f.t, f.u; color = wong[2], linewidth = 2, label = "cleaned")
    axislegend(ax; position = :rb, framevisible = false)
    return fig
end
