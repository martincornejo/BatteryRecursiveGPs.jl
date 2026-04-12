function eval_fit_parameters(params, uq)
    @printf("\n=== Composite OCV (joint-Hessian UQ) ===\n")
    @printf(
        "%-6s %8s %8s %8s %10s %8s %10s\n",
        "Cell", "Q0/Ah", "s", "C/Ah", "σ_C/Ah", "s0", "σ_s0"
    )
    for i in eachindex(params)
        Q0, s = params[i]
        C_i = Measurements.value(uq.est[i].Q)
        σ_C = Measurements.uncertainty(uq.est[i].Q)
        s0_i = Measurements.value(uq.est[i].s0)
        σ_s0 = Measurements.uncertainty(uq.est[i].s0)
        @printf(
            "Cell %2d: %6.2f  %8.4f  %6.2f  %8.4f  %8.4f  %8.4f\n",
            i, Q0, s, C_i, σ_C, s0_i, σ_s0
        )
    end
    return
end

function eval_cell_parameters(
        composite, params, uq, fit;
        V_ref = (3.3, 4.05), soc_ref = (0.05, 0.95)
    )
    qv = invert_ocv(composite)
    Q_ref = qv.(V_ref)
    Q_full_ref = (Q_ref[2] - Q_ref[1]) / (soc_ref[2] - soc_ref[1])
    Q_zero_ref = Q_ref[1] - soc_ref[1] * Q_full_ref

    @printf("\n=== Cell capacity and initial SOC ===\n")
    @printf(
        "  SOC anchors: %.2f V → %d %%,  %.2f V → %d %%\n",
        V_ref[1], Int(soc_ref[1] * 100), V_ref[2], Int(soc_ref[2] * 100)
    )
    @printf("  Q_full = %.3f Ah\n\n", Q_full_ref)
    @printf("%-8s %8s %8s %8s %8s\n", "Cell", "Q/Ah", "σ_Q/Ah", "SOC_0", "σ_SOC_0")
    for i in eachindex(params)
        Q0_i, s_i = params[i]
        C_i = Q_full_ref / s_i
        soc0_i = (Q0_i - Q_zero_ref) / Q_full_ref
        σ_Q = Measurements.uncertainty(uq.est[i].Q) * Q_full_ref / fit.Q_full
        σ_s0 = Measurements.uncertainty(uq.est[i].s0) * fit.Q_full / Q_full_ref
        @printf("Cell %2d: %8.3f %8.4f %8.4f %8.4f\n", i, C_i, σ_Q, soc0_i, σ_s0)
    end
    Cs = [Q_full_ref / p[2] for p in params]
    return @printf("%-8s %8.3f %8.4f\n", "Mean", mean(Cs), std(Cs))
end

function eval_ocv_residuals(composite, ocvs, params)
    Q_lo = first(composite.t)
    Q_hi = last(composite.t)

    @printf("\n=== Per-cell residuals vs composite OCV (mV) ===\n")
    @printf("%-8s %6s %8s %8s %8s\n", "Cell", "N", "mean", "rms", "max|r|")

    all_r = Float64[]
    for (i, (f, p)) in enumerate(zip(ocvs, params))
        Q0, s = p
        q = range(first(f.t), last(f.t); length = 300)
        q_aligned = collect(q .* s .+ Q0)
        v_cell = f.(q)
        mask = (q_aligned .>= Q_lo) .& (q_aligned .<= Q_hi)
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
    q = collect(composite.t)
    v = collect(composite.u)
    ocv_extrap = LinearInterpolation(v, q; extrapolation = ExtrapolationType.Linear)
    qv_extrap = invert_ocv(ocv_extrap; n_samples = 1000, extrapolation = ExtrapolationType.Linear)

    Q_at_Vmin = qv_extrap(V_min)
    Q_at_Vmax = qv_extrap(V_max)
    Q_full = Q_at_Vmax - Q_at_Vmin

    Q_data_min = first(composite.t)
    Q_data_max = last(composite.t)
    Q_data = Q_data_max - Q_data_min

    V_data_min = first(v)
    V_data_max = last(v)

    @printf("\nSOC range estimation (linear extrapolation):\n")
    @printf(
        "  Data covers:    %.1f - %.1f Ah (%.1f Ah) [%.2fV - %.2fV]\n",
        Q_data_min, Q_data_max, Q_data, V_data_min, V_data_max
    )
    @printf(
        "  Full range:     %.1f - %.1f Ah (%.1f Ah) [%.2fV - %.2fV]\n",
        Q_at_Vmin, Q_at_Vmax, Q_full, V_min, V_max
    )
    return @printf(
        "  SOC range used: %.1f%% - %.1f%%\n",
        100 * (Q_data_min - Q_at_Vmin) / Q_full,
        100 * (Q_data_max - Q_at_Vmin) / Q_full
    )
end
