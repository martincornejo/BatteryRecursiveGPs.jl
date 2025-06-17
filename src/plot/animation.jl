## OCV Animation

begin
    ocv_curve = param_evo.ocv
    SOC_reconstructed = StatsBase.reconstruct(dt.soc, df.soc[1:N_points])

    N_frames = size(ocv_curve, 1)
    fig = Figure()

    # OCV curve
    ax1 = CairoMakie.Axis(fig[1, 1], title="OCV curve", xlabel="SOC", ylabel="V")
    ocv_plt_μ = lines!(ax1, collect(limit_predict[1]:step_predict:limit_predict[2]), ocv_curve[1].μ, label="OCV aprox")
    ocv_plt_band = band!(ax1, collect(limit_predict[1]:step_predict:limit_predict[2]), ocv_curve[1].μ - 2ocv_curve[1].σ, ocv_curve[1].μ - 2ocv_curve[1].σ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))
    ocv_plt_real = lines!(ax1, collect(limit_predict[1]:step_predict:limit_predict[2]), Y_predict_ocv, label="OCV real")
    ylims!(ax1, minimum(Y_predict_ocv), maximum(Y_predict_ocv))
    v_line_min = vlines!(ax1, minimum(StatsBase.reconstruct(dt.soc, df.soc[1:1])), color=:red, linestyle=:dash)
    v_line_max = vlines!(ax1, maximum(StatsBase.reconstruct(dt.soc, df.soc[1:1])), color=:red, linestyle=:dash)

    record(fig, "ocv_animation_synthetic.mp4", 1:N_frames; framerate=80) do i
        steps = 40 * i
        ax1.title = "OCV curve - Iteration $steps"
        ocv_plt_μ[2] = ocv_curve[i].μ
        ocv_plt_band[2] = ocv_curve[i].μ .- 2 .* ocv_curve[i].σ
        ocv_plt_band[3] = ocv_curve[i].μ .+ 2 .* ocv_curve[i].σ

        if steps <= size(SOC_reconstructed, 1)
            v_line_min[1] = minimum(SOC_reconstructed[1:steps])
            v_line_max[1] = maximum(SOC_reconstructed[1:steps])
        else
            v_line_min[1] = minimum(SOC_reconstructed[1:N_points])
            v_line_max[1] = maximum(SOC_reconstructed[1:N_points])
        end
    end
end

## V0 animations

begin
    v_curve = param_evo.v
    SOC_reconstructed = StatsBase.reconstruct(dt.soc, df.soc[1:N_points])

    N_frames = size(v_curve, 1)
    fig = Figure()

    # V

    ax3 = CairoMakie.Axis(fig[1, 1], title="Voltage", xlabel="time", ylabel="V")
    v_plt_μ = lines!(ax3, df.t[1:step_voltage:end], v_curve[1].μ, label="V aprox")
    v_plt_band = band!(ax3, df.t[1:step_voltage:end], v_curve[1].μ - 2v_curve[1].σ, v_curve[1].μ + 2v_curve[1].σ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))
    v_plt_real = lines!(ax3, df.t[1:step_voltage:end], df.v[1:step_voltage:end], label="V real")
    ylims!(ax3, minimum(df.v) - 0.1, maximum(df.v) + 0.7)
    axislegend(ax3)
    record(fig, "v_animation_synthetic.mp4", 1:N_frames; framerate=80) do i
        steps = 40 * i
        ax3.title = "V curve - Iteration $steps"
        v_plt_μ[2] = v_curve[i].μ
        v_plt_band[2] = v_curve[i].μ .- 2 .* v_curve[i].σ
        v_plt_band[3] = v_curve[i].μ .+ 2 .* v_curve[i].σ


    end
end


## R0 animation
begin
    r_curve = param_evo.r
    SOC_reconstructed = StatsBase.reconstruct(dt.soc, df.soc[1:N_points])

    N_frames = size(r_curve, 1)
    fig = Figure(size=(800, 400))

    # R
    ax2 = CairoMakie.Axis(fig[1, 1], title="R0 curve", xlabel="SOC", ylabel="V")
    r_plt_μ = lines!(ax2, collect(limit_predict[1]:step_predict:limit_predict[2]), r_curve[1].μ, label="R0 aprox")
    r_plt_band = band!(ax2, collect(limit_predict[1]:step_predict:limit_predict[2]), r_curve[1].μ - 2r_curve[1].σ, r_curve[1].μ + 2r_curve[1].σ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))
    ylims!(ax2, 0.0005, 0.0013)
    v_line_min_r = vlines!(ax2, minimum(StatsBase.reconstruct(dt.soc, df.soc[1:1])), color=:red, linestyle=:dash)
    v_line_max_r = vlines!(ax2, maximum(StatsBase.reconstruct(dt.soc, df.soc[1:1])), color=:red, linestyle=:dash)



    record(fig, "r_animation_profile2.mp4", 1:N_frames; framerate=80) do i
        steps = 10 * i

        ax2.title = "R0 curve - Iteration $steps"
        r_plt_μ[2] = r_curve[i].μ
        r_plt_band[2] = r_curve[i].μ .- 2 .* r_curve[i].σ
        r_plt_band[3] = r_curve[i].μ .+ 2 .* r_curve[i].σ

        ## Updating limiting lines
        if steps <= size(SOC_reconstructed, 1)
            v_line_min_r[1] = minimum(SOC_reconstructed[1:steps])
            v_line_max_r[1] = maximum(SOC_reconstructed[1:steps])
        else

            v_line_min_r[1] = minimum(SOC_reconstructed[1:end])
            v_line_max_r[1] = maximum(SOC_reconstructed[1:end])
        end

    end
end