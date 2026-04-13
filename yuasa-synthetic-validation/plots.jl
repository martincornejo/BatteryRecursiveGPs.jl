function plot_soc_estimation_state(data, sol, id, param_cells_state)
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]
    zt = fit_zscore()
    Ts = 1.0

    q̂ = first.(sol.xt)
    R̂ = first.(sol.Rt)
    t = sol.idx ./ 3600.0
    qμ = StatsBase.reconstruct(zt.q, q̂)
    qσ = StatsBase.reconstruct(zt.q, sqrt.(R̂))

    Q_est = Measurements.value(param_cells_state[id][:Q])
    s0_est = Measurements.value(param_cells_state[id][:soc])

    soc_kf = s0_est .+ qμ ./ Q_est
    soc_kf_lo = s0_est .+ (qμ .- qσ) ./ Q_est
    soc_kf_hi = s0_est .+ (qμ .+ qσ) ./ Q_est

    q_cc = cumsum(data.î) .* Ts ./ 3600
    soc_cc = s0_est .+ q_cc ./ Q_est

    soc_ref = data[:, "soc_cell_$id"]

    lines!(ax[1], t, soc_ref; color = :black, label = "True")
    lines!(ax[1], t, soc_kf; label = "KF")
    band!(ax[1], t, soc_kf_lo, soc_kf_hi; alpha = 0.4)
    lines!(ax[1], t, soc_cc; label = "CC")

    lines!(ax[2], t, (soc_kf .- soc_ref) .* 100; label = "KF")
    lines!(ax[2], t, (soc_cc .- soc_ref) .* 100; label = "CC")
    hlines!(ax[2], [0]; color = :black, linestyle = :dash)

    ax[1].ylabel = "SOC / p.u."
    ax[2].ylabel = "Error / %"
    ax[2].xlabel = "Time / h"
    Legend(fig[3, 1], ax[1]; orientation = :horizontal)

    xlims!(ax[1], t[begin], t[end])
    xlims!(ax[2], t[begin], t[end])
    linkxaxes!(ax...)
    return fig
end

function plot_q_estimation_state(data, sol)
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]
    zt = fit_zscore()

    q̂ = first.(sol.xt)
    R̂ = first.(sol.Rt)
    t = sol.idx * 1.0 / 3600
    qμ = StatsBase.reconstruct(zt.q, q̂)
    qσ = StatsBase.reconstruct(zt.q, sqrt.(R̂))

    lines!(ax[1], t, qμ, label = "KF estimate")
    band!(ax[1], t, qμ - qσ, qμ + qσ, alpha = 0.5)
    lines!(ax[1], t, data.q, label = "CMU")

    q_cc = cumsum(data.î) * 1.0 / 3600
    lines!(ax[2], t, qμ - data.q, label = "KF − osc")
    lines!(ax[2], t, q_cc - data.q, label = "CMU − osc")
    hlines!(ax[2], [0]; color = :black, linestyle = :dash)

    ax[1].ylabel = "Charge / Ah"
    ax[2].ylabel = "Error / Ah"
    ax[2].xlabel = "Time / h"
    Legend(fig[3, 1], ax[1], orientation = :horizontal)

    xlims!(ax[1], t[begin], t[end])
    xlims!(ax[2], t[begin], t[end])

    linkxaxes!(ax...)
    return fig
end

"""
    plot_r0_diagnostics(models, sols, param_cells, fR025)

R0 GP posterior (mean ± 2σ) for all cells vs the reference R0 curve at 25 °C,
plotted against SOC using `param_cells` for the charge-to-SOC mapping.
Bottom panel shows the residual in mΩ.
"""
function plot_r0_diagnostics(models, sols, param_cells, fR025)
    fig = Figure(size = (600, 500))
    ax = [Axis(fig[i, 1]) for i in 1:2]
    ax[1].title = "R0 GP posterior vs reference (25 °C)"
    ax[1].ylabel = "R0 / mΩ"
    ax[2].ylabel = "Residual / mΩ"
    ax[2].xlabel = "SOC / p.u."
    hidexdecorations!(ax[1]; ticks = false, grid = false)

    s = 0.0:0.005:1.0
    lines!(ax[1], s, fR025.(s) .* 1.0e3; color = :black, linestyle = :dash, label = "Reference")
    hlines!(ax[2], [0]; color = :black, linestyle = :dash)

    for id in sort(collect(keys(models)))
        (; q, μ, σ) = gp_r0(models[id], sols[id])

        Q_est = Measurements.value(param_cells[id][:Q])
        s0_est = Measurements.value(param_cells[id][:soc])
        soc_est = s0_est .+ q ./ Q_est

        rμ_mΩ = μ .* 1.0e3
        rσ_mΩ = σ .* 1.0e3
        r_ref = fR025.(soc_est) .* 1.0e3

        lines!(ax[1], soc_est, rμ_mΩ)
        band!(ax[1], soc_est, rμ_mΩ .- 2rσ_mΩ, rμ_mΩ .+ 2rσ_mΩ; alpha = 0.3)
        lines!(ax[2], soc_est, rμ_mΩ .- r_ref)
    end

    linkxaxes!(ax...)
    return fig
end

function plot_ocv_diagnostics(models, sols, param_cells, params_real, focv)
    fig = Figure(size = (800, 500))
    ax = [Axis(fig[i, j]) for i in 1:2, j in 1:2]

    titles = ["Estimated params", "Real params"]
    for j in 1:2
        ax[1, j].title = titles[j]
        ax[1, j].ylabel = "OCV / V"
        ax[2, j].ylabel = "Mismatch / mV"
        ax[2, j].xlabel = "SOC / p.u."
        hidexdecorations!(ax[1, j]; ticks = false, grid = false)
    end

    s = 0.0:0.005:1.0
    for j in 1:2
        lines!(ax[1, j], s, focv.(s); color = :black, linestyle = :dash)
        hlines!(ax[2, j], [0]; color = :black, linestyle = :dash)
    end

    for (id, model) in models
        (; q, μ, σ) = gp_ocv(model, sols[id])

        Q_est = Measurements.value(param_cells[id][:Q])
        s0_est = Measurements.value(param_cells[id][:soc])
        soc_est = s0_est .+ q ./ Q_est

        Q_r = params_real["cell_$id"]["Q"]
        s0_r = params_real["cell_$id"]["soc"]
        soc_r = s0_r .+ q ./ Q_r

        for (j, soc) in enumerate((soc_est, soc_r))
            lines!(ax[1, j], soc, μ)
            band!(ax[1, j], soc, μ .- 2σ, μ .+ 2σ; alpha = 0.3)
            lines!(ax[2, j], soc, (μ .- focv.(soc)) .* 1000)
        end
    end

    linkxaxes!(ax[1, 1], ax[2, 1])
    linkxaxes!(ax[1, 2], ax[2, 2])
    linkyaxes!(ax[1, 1], ax[1, 2])
    linkyaxes!(ax[2, 1], ax[2, 2])
    return fig
end

function plot_qs_scatter(param_cells, params_real)
    ids = sort(collect(keys(param_cells)))
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "Q_cell / Ah", ylabel = "Initial SOC / %")

    Q_true = [params_real["cell_$id"]["Q"] for id in ids]
    s_true = [params_real["cell_$id"]["soc"] for id in ids] .* 100
    scatter!(ax, Q_true, s_true; label = "Real")

    Qμ = [Measurements.value(param_cells[id][:Q]) for id in ids]
    Qσ = [Measurements.uncertainty(param_cells[id][:Q]) for id in ids]
    sμ = [Measurements.value(param_cells[id][:soc]) * 100 for id in ids]
    sσ = [Measurements.uncertainty(param_cells[id][:soc]) * 100 for id in ids]
    scatter!(ax, Qμ, sμ; label = "Estimation")
    errorbars!(ax, Qμ, sμ, Qσ; whiskerwidth = 10, direction = :x, color = Cycled(2))
    errorbars!(ax, Qμ, sμ, sσ; whiskerwidth = 10, direction = :y, color = Cycled(2))

    Legend(fig[2, 1], ax; orientation = :horizontal)
    return fig
end

function plot_composite_ocv(
        posteriors, param_cells, fit, focv;
        v_ref, soc_ref,
    )
    ids = sort(collect(keys(posteriors)))
    fig = Figure(size = (1000, 900))
    ax = Axis(
        fig[1, 1];
        xlabel = "SOC (lab gauge)", ylabel = "OCV / V",
        title = "Composite OCV vs lab reference — $(length(ids)) cells overlaid",
    )

    cell_curves = map(ids) do id
        p = posteriors[id]
        Qm = Measurements.value(param_cells[id][:Q])
        s0m = Measurements.value(param_cells[id][:soc])
        soc_cell = s0m .+ p.q ./ Qm
        (; id, soc_cell, μ = p.μ)
    end
    for c in cell_curves
        lines!(ax, c.soc_cell, c.μ; color = (:gray, 0.45), linewidth = 1)
    end

    lines!(
        ax, focv.t, focv.u;
        color = :black, linewidth = 2.5,
        linestyle = :dash, label = "Lab reference",
    )

    composite_interp = LinearInterpolation(fit.soc_grid, fit.v_grid)
    soc_at_ref = composite_interp.(v_ref)
    soc_span = (soc_ref[2] - soc_ref[1]) / (soc_at_ref[2] - soc_at_ref[1])
    soc_zero = soc_ref[1] - soc_span * soc_at_ref[1]
    soc_comp_lab = soc_span .* fit.soc_grid .+ soc_zero
    lines!(
        ax, soc_comp_lab, fit.v_grid;
        color = :crimson, linewidth = 2.5, label = "Composite (lab-free)",
    )

    vlines!(ax, [soc_ref...]; color = (:gray, 0.5), linestyle = :dot)
    axislegend(ax; position = :rb)

    ax2 = Axis(
        fig[2, 1];
        xlabel = "SOC (lab gauge)", ylabel = "ΔV / mV",
        title = "Composite − lab reference",
    )
    resid_mV = (fit.v_grid .- focv.(soc_comp_lab)) .* 1000
    lines!(ax2, soc_comp_lab, resid_mV; color = :crimson, linewidth = 2)
    hlines!(ax2, [0.0]; color = (:gray, 0.5), linestyle = :dot)

    ax3 = Axis(
        fig[3, 1];
        xlabel = "SOC (lab gauge)", ylabel = "ΔV / mV",
        title = "Per-cell GP − composite",
    )
    comp_interp = LinearInterpolation(fit.v_grid, soc_comp_lab)
    comp_lo, comp_hi = extrema(soc_comp_lab)
    colors = cgrad(:viridis, length(cell_curves); categorical = true)
    for (k, c) in enumerate(cell_curves)
        mask = (c.soc_cell .>= comp_lo) .& (c.soc_cell .<= comp_hi)
        any(mask) || continue
        soc_k = c.soc_cell[mask]
        resid_k = (c.μ[mask] .- comp_interp.(soc_k)) .* 1000
        lines!(
            ax3, soc_k, resid_k;
            color = colors[k], linewidth = 1.2, label = "cell $(c.id)",
        )
    end
    hlines!(ax3, [0.0]; color = (:gray, 0.5), linestyle = :dot)
    Legend(fig[1:3, 2], ax3, "Cell"; framevisible = false, labelsize = 10)

    rowsize!(fig.layout, 1, Relative(0.6))
    rowsize!(fig.layout, 2, Relative(0.2))
    rowsize!(fig.layout, 3, Relative(0.2))

    linkxaxes!(ax, ax2, ax3)
    return fig
end
