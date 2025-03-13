# using Revise
using Distributions
using MLUtils: DataLoader
using LinearAlgebra

using KernelFunctions
import ComponentArrays: ComponentArray

using CairoMakie
using ColorSchemes

include("src/rgp.jl")
using .RecursiveGPs


begin # generate training data
    ## Dummy test
    limit_test = [-8, 8]
    limit_basis = [-8, 8]
    step_test = 0.03
    std_test = 0.01
    ## Test
    X_test = collect(limit_test[1]:step_test:limit_test[2])
    Y_test = sin.(X_test) + rand(Normal(0, std_test), size(X_test)[1])
    data = DataLoader((x=X_test, y=Y_test), batchsize=9, shuffle=true)
end

begin # create RGP
    limit_basis = [-10, 10]
    step_basis = 0.7
    X_basis = collect(limit_test[1]:step_basis:limit_test[2])
    σ = 0.01

    kernel = 2 * with_lengthscale(SEKernel(), 1)
    rgp = RGPModel(kernel, σ, X_basis)
end

for batch in data
    learn!(rgp, batch.x, batch.y)
end


begin # test regression
    limit_predict = [-10, 10]
    step_predict = 0.32
    X_predict = collect(limit_predict[1]:step_predict:limit_predict[2])
    Y_predict = sin.(X_predict)
    (; μ, Σ) = predict(rgp, X_predict)
    σ = sqrt.(diag(Σ))

    # plot
    fig = Figure()
    ax = Axis(fig[1, 1], title="GP updated for sin(x)", xlabel="x", ylabel="y", xticks=limit_predict[1]:1:limit_predict[2])
    lines!(ax, X_predict, Y_predict, label="sin(x) real")
    lines!(ax, X_predict, μ, label="sin(x) aprox")
    lines!(ax, [limit_test[1], limit_test[1]], [-1.5, 1.5], color=:red, linestyle=:dash, label="Outsite test data")
    lines!(ax, [limit_test[2], limit_test[2]], [-1.5, 1.5], color=:red, linestyle=:dash, label="Outsite test data")

    band!(ax, X_predict, μ - 2σ, μ + 2σ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))
    axislegend(ax)
    display(fig)
end