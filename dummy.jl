using Distributions
using MLUtils: DataLoader
using LinearAlgebra
using AbstractGPs
using CSV
using DataFrames
import ComponentArrays: ComponentArray
using DataInterpolations

using CairoMakie
using ColorSchemes

include("src/rgp.jl")
include("src/battModel.jl")
using .RecursiveGPs
using .battModel

begin
    df_p2 = CSV.read("data/profile2.csv", DataFrame)
    ## Train
    N_points = size(df_p2)[1]
    soc_test = df_p2[!, "soc"][1:N_points]
    i_test = df_p2[!, "i"][1:N_points]

    v_test = df_p2[!, "v"][1:N_points]
    batch_size = 1
    data = DataLoader((
            x=(
                soc=soc_test,
                i=i_test
            ),
            y=v_test
        ),
        batchsize=batch_size, shuffle=false
    )
end

begin
    ## Building GPs
    l_ocv = 0.2
    σ_ocv = 0.1

    l_r = 0.2
    σ_r = 0.1

    σ_f1 = 0.01
    σ_f2 = 0.00005


    limit_basis_ocv = [0, 1]
    step_basis_ocv = 0.01
    X_basis_ocv = collect(limit_basis_ocv[1]:step_basis_ocv:limit_basis_ocv[2])
    n_basis = size(X_basis_ocv)[1]
    X_basis_r = collect(limit_basis_ocv[1]:step_basis_ocv:limit_basis_ocv[2])

    gp_ocv = gp_ocv = GP(
        LinearKernel() +
        σ_ocv * with_lengthscale(SEKernel(), l_ocv)
    )
    rgp_ocv = RGPModel(gp_ocv, σ_f1, X_basis_ocv)

    gp_r = GP(σ_r * with_lengthscale(SEKernel(), l_r))
    rgp_r = RGPModel(gp_r, σ_f2, X_basis_r)

    batt = BATTModel(rgp_ocv, rgp_r)
end

begin
    ##Train
    for (n, batch) in enumerate(data)
        battery_learn!(batt, batch)
    end
end


begin
    ##PLot
    limit_predict = [0, 1]
    step_predict = 0.01

    X_predict_soc = collect(limit_predict[1]:step_predict:limit_predict[2])
    rgp_ocv = batt.rgp_ocv
    rgp_r = batt.rgp_r

    μ_ocv = predict(rgp_ocv, X_predict_soc).μ
    Σ_ocv = predict(rgp_ocv, X_predict_soc).Σ
    var_ocv = sqrt.(abs.(diag(Σ_ocv)))
    μ_r = predict(rgp_r, X_predict_soc).μ
    Σ_r = predict(rgp_r, X_predict_soc).Σ
    var_r = sqrt.(abs.(diag(Σ_r)))

    V_p = predict(rgp_ocv, soc_test[1:50:end]).μ + i_test[1:50:end] .* predict(rgp_r, soc_test[1:50:end]).μ
    V_r = v_test[1:50:end]
    ##  function
    fig = Figure(size=(1200, 800))
    ax1 = Axis(fig[1, 1], title="GP updated for ocv l = $(l_ocv), σ =$(σ_ocv), noise = $(σ_f1) ", xlabel="soc", ylabel="ocv", xticks=limit_predict[1]:0.1:limit_predict[2])
    lines!(ax1, X_predict_soc, μ_ocv, label="OCV aprox")
    vlines!(ax1, [minimum(soc_test), maximum(soc_test)], color=:red, linestyle=:dash, label="Outsite test data")
    ylims!(ax1, 3.2, 4.2)
    band!(ax1, X_predict_soc, μ_ocv - 2var_ocv, μ_ocv + 2var_ocv; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))

    ax2 = Axis(fig[2, 1], title="GP updated for R0 l = $(l_r), σ =$(σ_r), noise = $(σ_f2)  ", xlabel="soc", ylabel="R0", xticks=limit_predict[1]:0.1:limit_predict[2])
    lines!(ax2, X_predict_soc, μ_r, label="R0 aprox")
    vlines!(ax2, [minimum(soc_test), maximum(soc_test)], color=:red, linestyle=:dash, label="Outsite test data")
    lines!(ax2, [minimum(soc_test), minimum(soc_test)], [minimum(μ_r), maximum(μ_r)], color=:red, linestyle=:dash, label="Outsite test data")
    band!(ax2, X_predict_soc, μ_r - 2var_r, μ_r + 2var_r; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))

    ylims!(ax2, 0.0004, 0.0013)

    #ax3 = Axis(fig[3,1], title = "SOC histogram", xticks=limit_predict[1]:0.1:limit_predict[2])
    #hist!(ax3, soc_test; bins = 20, color =:gray)
    #xlims!(ax3, 0, 1)


    #save("pictures/profile2/ocv_parameters/profile2_r_l$(l_ocv)_sigma$(σ_ocv)_noise$(σ_f1)_n_basis_$(n_basis)_batch_size$(batch_size)_new_noise.png", fig)
    display(fig)
end

