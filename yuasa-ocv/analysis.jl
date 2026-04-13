function eval_fit_parameters(fit)
    (; params, Q_cell, s0) = fit
    @printf("\n=== Composite OCV (OLS UQ) ===\n")
    @printf(
        "%-6s %8s %8s %8s %10s %8s %10s\n",
        "Cell", "Q0/Ah", "s", "C/Ah", "σ_C/Ah", "s0", "σ_s0"
    )
    for i in eachindex(params)
        Q0, s = params[i]
        @printf(
            "Cell %2d: %6.2f  %8.4f  %6.2f  %8.4f  %8.4f  %8.4f\n",
            i, Q0, s,
            Measurements.value(Q_cell[i]),
            Measurements.uncertainty(Q_cell[i]),
            Measurements.value(s0[i]),
            Measurements.uncertainty(s0[i]),
        )
    end
    return
end

function eval_cell_parameters(
        fit;
        v_ref = (3.3, 4.05), soc_ref = (0.05, 0.95)
    )
    rescaled = rescale_composite_ocv(fit; v_ref, soc_ref)

    @printf("\n=== Cell capacity and initial SOC ===\n")
    @printf(
        "  SOC anchors: %.2f V → %d %%,  %.2f V → %d %%\n",
        v_ref[1], Int(soc_ref[1] * 100), v_ref[2], Int(soc_ref[2] * 100)
    )
    @printf("%-8s %8s %8s %8s %8s\n", "Cell", "Q/Ah", "σ_Q/Ah", "SOC_0", "σ_SOC_0")
    for i in eachindex(rescaled.Q_cell)
        @printf(
            "Cell %2d: %8.3f %8.4f %8.4f %8.4f\n",
            i, Measurements.value(rescaled.Q_cell[i]),
            Measurements.uncertainty(rescaled.Q_cell[i]),
            Measurements.value(rescaled.s0[i]),
            Measurements.uncertainty(rescaled.s0[i]),
        )
    end
    Cs = Measurements.value.(rescaled.Q_cell)
    return @printf("%-8s %8.3f %8.4f\n", "Mean", mean(Cs), std(Cs))
end

function eval_ocv_residuals(composite, ocvs, params)
    soc_lo = first(composite.t)
    soc_hi = last(composite.t)

    @printf("\n=== Per-cell residuals vs composite OCV (mV) ===\n")
    @printf("%-8s %6s %8s %8s %8s\n", "Cell", "N", "mean", "rms", "max|r|")

    all_r = Float64[]
    for (i, (f, p)) in enumerate(zip(ocvs, params))
        Q0, s = p
        q = range(first(f.t), last(f.t); length = 300)
        q_aligned = collect(q .* s .+ Q0)
        v_cell = f.(q)
        mask = (q_aligned .>= soc_lo) .& (q_aligned .<= soc_hi)
        r_mV = (v_cell[mask] .- composite.(q_aligned[mask])) .* 1000
        append!(all_r, r_mV)
        @printf(
            "Cell %2d: %6d %+8.2f %8.2f %8.2f\n",
            i, length(r_mV), mean(r_mV), sqrt(mean(abs2, r_mV)), maximum(abs, r_mV)
        )
    end
    return @printf(
        "%-8s %6d %+8.2f %8.2f %8.2f\n",
        "ALL", length(all_r),
        mean(all_r), sqrt(mean(abs2, all_r)), maximum(abs, all_r)
    )
end

function eval_soc_range(composite; V_min = 2.9, V_max = 4.1)
    soc = collect(composite.t)
    v = collect(composite.u)
    ocv_extrap = LinearInterpolation(v, soc; extrapolation = ExtrapolationType.Linear)
    sv_extrap = invert_ocv(ocv_extrap; n_samples = 1000, extrapolation = ExtrapolationType.Linear)

    soc_at_Vmin = sv_extrap(V_min)
    soc_at_Vmax = sv_extrap(V_max)
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
    return @printf(
        "  SOC range used: %.1f%% - %.1f%%\n",
        100 * (soc_data_min - soc_at_Vmin) / soc_full,
        100 * (soc_data_max - soc_at_Vmin) / soc_full
    )
end
