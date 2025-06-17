function plot_profile2(batt, l_ocv, l_r, σ_f1, σ_f2, soc_test, dt, param_evo)

    soc_test = StatsBase.reconstruct(dt.soc, soc_test)
    limit_predict = [0, 1]
    step_predict = 0.01
    X_predict_soc = collect(limit_predict[1]:step_predict:limit_predict[2])
    X_predict_soc_n = StatsBase.transform(dt.soc, X_predict_soc)
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


    ##  function
    fig = Figure(size=(1200, 800))

    #OCV
    ax1 = CairoMakie.Axis(fig[1, 1], title="GP updated for ocv l = $(l_ocv), σ =$(σ_ocv), noise = $(σ_f1) ", xlabel="soc", ylabel="ocv", xticks=limit_predict[1]:0.1:limit_predict[2])
    lines!(ax1, X_predict_soc, μ_ocv, label="OCV aprox")
    vlines!(ax1, [minimum(soc_test), maximum(soc_test)], color=:red, linestyle=:dash, label="Outsite test data")
    ylims!(ax1, 3.2, 4.2)
    band!(ax1, X_predict_soc, μ_ocv - 2var_ocv, μ_ocv + 2var_ocv; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))


    ## Param evolution


    # Parameter evolution curves
    ax2 = CairoMakie.Axis(fig[2, 1], title="Parameter evolution", xlabel="iter", ylabel="R1")
    lines!(ax2, getindex.(param_evo.μ, 1), label="R1")
    ylims!(ax2, 0.0, 5e-3)

    ax3 = CairoMakie.Axis(fig[3, 1], title="Parameter evolution", xlabel="iter", ylabel="τ1")
    lines!(ax3, getindex.(param_evo.μ, 2), label="τ1")
    ylims!(ax3, 90, maximum(getindex.(param_evo.μ, 2)) + 20)


    #R0
    ax4 = CairoMakie.Axis(fig[4, 1], title="GP updated for R0 l = $(l_r), σ =$(σ_r), noise = $(σ_f2)  ", xlabel="soc", ylabel="R0", xticks=limit_predict[1]:0.1:limit_predict[2])
    lines!(ax4, X_predict_soc, μ_r, label="R0 aprox")
    vlines!(ax4, [minimum(soc_test), maximum(soc_test)], color=:red, linestyle=:dash, label="Outsite test data")
    band!(ax4, X_predict_soc, μ_r - 2var_r, μ_r + 2var_r; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))
    ylims!(ax4, minimum(μ_r) - 0.0001, 0.0002)




    axislegend(ax1)
    axislegend(ax2)
    axislegend(ax3)
    axislegend(ax4)

    display(fig)
    return fig
end