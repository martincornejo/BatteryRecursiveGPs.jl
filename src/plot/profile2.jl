function plot_profile2(batt, l_ocv, l_r, σ_f1, σ_f2, Vrc, soc_test, dt)
    ##PLot
    limit_predict = [0, 1]
    step_predict = 0.01
    X_predict_soc = collect(limit_predict[1]:step_predict:limit_predict[2])
    X_predict_soc_n = X_predict_soc

    X_predict_soc_n = StatsBase.transform(dt.soc, X_predict_soc)


    rgp_ocv = batt.rgp_ocv
    rgp_r = batt.rgp_r

    μ_ocv = RecursiveGPs.predict(rgp_ocv, soc_test[1:50:end], train=false).μ
    μ_ocv = StatsBase.reconstruct(dt.v, μ_ocv)
    μ_r = RecursiveGPs.predict(rgp_r, soc_test[1:50:end], train=false).μ
    μ_r = StatsBase.reconstruct(dt.σ, μ_r)


    if Vrc == false
        V_p = μ_ocv + i_test[1:50:end] .* μ_r
    else
        V_p = μ_ocv + i_test[1:50:end] .* μ_r + Vrc[1:50:end]
    end

    V_r = v_test[1:50:end]
    soc_test_n = soc_test


    μ_ocv = RecursiveGPs.predict(rgp_ocv, X_predict_soc_n, train=false).μ
    Σ_ocv = RecursiveGPs.predict(rgp_ocv, X_predict_soc_n, train=false).Σ
    var_ocv = sqrt.(abs.(diag(Σ_ocv)))


    μ_r = RecursiveGPs.predict(rgp_r, X_predict_soc_n, train=false).μ
    Σ_r = RecursiveGPs.predict(rgp_r, X_predict_soc_n, train=false).Σ
    var_r = sqrt.(abs.(diag(Σ_r)))

    μ_ocv = StatsBase.reconstruct(dt.v, μ_ocv)
    var_ocv = StatsBase.reconstruct(dt.σ, var_ocv)
    r0 = dt.σ.scale[1] / dt.i.scale[1]
    μ_r = μ_r * r0
    var_r = var_r * r0
    soc_test_n = StatsBase.reconstruct(dt.soc, soc_test)
    V_p = StatsBase.reconstruct(dt.v, V_p)
    V_r = StatsBase.reconstruct(dt.v, v_test[1:50:end])



    error = abs.(V_p - V_r)
    ##  function
    fig = Figure(size=(1200, 800))

    #OCV
    ax1 = CairoMakie.Axis(fig[1, 1], title="GP updated for ocv l = $(l_ocv), σ =$(σ_ocv), noise = $(σ_f1) ", xlabel="soc", ylabel="ocv", xticks=limit_predict[1]:0.1:limit_predict[2])
    lines!(ax1, X_predict_soc, μ_ocv, label="OCV aprox")
    vlines!(ax1, [minimum(soc_test_n), maximum(soc_test_n)], color=:red, linestyle=:dash, label="Outsite test data")
    ylims!(ax1, 3.2, 4.2)
    band!(ax1, X_predict_soc, μ_ocv - 2var_ocv, μ_ocv + 2var_ocv; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))

    #R0
    ax2 = CairoMakie.Axis(fig[2, 1], title="GP updated for R0 l = $(l_r), σ =$(σ_r), noise = $(σ_f2)  ", xlabel="soc", ylabel="R0", xticks=limit_predict[1]:0.1:limit_predict[2])
    lines!(ax2, X_predict_soc, μ_r, label="R0 aprox")
    vlines!(ax2, [minimum(soc_test_n), maximum(soc_test_n)], color=:red, linestyle=:dash, label="Outsite test data")
    band!(ax2, X_predict_soc, μ_r - 2var_r, μ_r + 2var_r; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))
    ylims!(ax2, minimum(μ_r), maximum(μ_r) + 0.0001)
    #Voltage_profile
    ax3 = CairoMakie.Axis(fig[3, 1], title="Voltage profile")
    lines!(ax3, V_r, label="Real")
    lines!(ax3, V_p, label="Aprox")

    #Error
    ax4 = CairoMakie.Axis(fig[4, 1], title="Absolute Error")
    lines!(ax4, error, label="Abs error")
    ylims!(ax4, 0.0, 0.001)

    axislegend(ax1)
    axislegend(ax2)
    axislegend(ax3)
    axislegend(ax4)

    display(fig)
    return fig
end