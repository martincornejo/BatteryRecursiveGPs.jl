using Distributions
using MLUtils: DataLoader
using LinearAlgebra
using AbstractGPs
using CSV
using DataFrames
using DataInterpolations
using CairoMakie
using ColorSchemes
using StatsBase
using Revise
using ComponentArrays
includet("src/battModel/rgp.jl")
includet("src/battModel/battModel.jl")
includet("src/battModel/kf_utils.jl")
includet("src/battModel/kf_core.jl")
includet("src/plot/profile2.jl")
using .RecursiveGPs
using .battModel
using .kf_core
using .kf_utils




function fit_zscore(df)
    v = StatsBase.fit(ZScoreTransform, df.v)
    σ = StatsBase.fit(ZScoreTransform, df.v, center=false)
    i = StatsBase.fit(ZScoreTransform, df.i, center=false)
    soc = StatsBase.fit(ZScoreTransform, df.soc)
    return (; v, σ, i, soc)
end

function normalize_data(df, dt)
    v = df.v#StatsBase.transform(dt.v, df.v)
    i = df.i#StatsBase.transform(dt.i, df.i)
    soc = StatsBase.transform(dt.soc, df.soc)
    return DataFrame(; df.t, v, i, soc)
end


begin
    RC = true
    if RC == true
        df_data = CSV.read("data/output_data_with_one_rc.csv", DataFrame)

    else
        df_data = CSV.read("data/output_data_without_rc.csv", DataFrame)
    end

    df_profile = CSV.read("data/profile.csv", DataFrame)
    df_ocv = CSV.read("data/ocv.csv", DataFrame)
    fi = ConstantInterpolation(df_profile.i, df_profile.t)

    if RC == true
        df = DataFrame(
            t=df_data.time,
            v=df_data.voltage,
            i=fi.(df_data.time),  # interpolated current
            soc=df_data.soc
        )
    end
    if RC == false
        df = DataFrame(
            t=df_data.time,
            v=df_data.voltage - 15e-3 * fi.(df_data.time),
            i=fi.(df_data.time),  # interpolated current
            soc=df_data.soc
        )
    end
    ## Normalizing data
    dt = fit_zscore(df)
    df = normalize_data(df, dt)


    ## Generating ocv curve
    df_ocv.soc = StatsBase.transform(dt.soc, df_ocv.soc)
    df_ocv.ocv = StatsBase.transform(dt.v, df_ocv.ocv)
    fi = ConstantInterpolation(df_profile.i, df_profile.t)
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc; extrapolation=ExtrapolationType.Constant)

    ## Building DataLoader
    N_points = size(df_data.soc)[1]
    X_soc = df.soc[1:N_points]
    X_i = df.i[1:N_points]
    Y_v = df.v[1:N_points]
    X_t = df.t[1:N_points]


    batch_size = 1
    data = DataLoader((
            x=(
                t=X_t,
                soc=X_soc,
                i=X_i
            ),
            y=Y_v
        ),
        batchsize=batch_size, shuffle=false
    )
end


## Building GP
begin
    l_ocv = 0.2
    σ_ocv = 0.1
    l_r = 0.2
    σ_r = 0.1

    σ_f1 = 0.01
    σ_f2 = 5e-8
    σ_model = 0.001

    limit_basis_ocv = [0, 1]
    step_basis_ocv = 0.01
    X_basis_ocv = collect(limit_basis_ocv[1]:step_basis_ocv:limit_basis_ocv[2])
    n_basis = size(X_basis_ocv)[1]
    X_basis_r = collect(limit_basis_ocv[1]:step_basis_ocv:limit_basis_ocv[2])

    X_basis_ocv = StatsBase.transform(dt.soc, X_basis_ocv)
    X_basis_r = StatsBase.transform(dt.soc, X_basis_r)


    gp_ocv = gp_ocv = GP(
        LinearKernel() +
        σ_ocv * with_lengthscale(SEKernel(), l_ocv)
    )


    mean_function_ocv = x -> focv(x)
    rgp_ocv = RGPModel(gp_ocv, σ_f1, X_basis_ocv)

    gp_r = GP(σ_r * with_lengthscale(SEKernel(), l_r))

    mean_function_r0 = x -> StatsBase.transform(dt.σ, [15e-3])[1]
    rgp_r = RGPModel(gp_r, σ_f2, X_basis_r; mean_function=mean_function_r0)


    #### Building RC
    μ_params = [30e-3, 40]

    batt = BATTModel(rgp_ocv, rgp_r, μ_params, dt, model_R=false)
end


## Training with batt
begin
    batt.param_noise.w.Σ = [
        1e-9 0;
        0 9e-3
    ]

    batt.param_noise.v.Σ = [5e-4]

    ## Parameter evolution
    param_evo = DataFrame(
        μ=Any[],
        Σ=Any[],
        vrc=Float64[],
        param_noise=Any[]
    )
    push!(param_evo, (
        μ=batt.μ_params,
        Σ=batt.Σ_params,
        vrc=0.0,
        param_noise=deepcopy(batt.param_noise)
    )
    )

    for (n, batch) in enumerate(data)
        if n % 10 == 0
            push!(param_evo, (
                μ=deepcopy(batt.μ_params),
                Σ=deepcopy(batt.Σ_params),
                vrc=deepcopy(batt.μ[end]),
                param_noise=deepcopy(batt.param_noise)
            )
            )
        end

        test_battery_learn_dual_kf!(batt, batch; rts=false)

    end
end




# Generating Data for plotting
begin

    using_real = true
    ## OCV curve
    limit_predict = [0, 1]
    step_predict = 0.015
    X_predict_soc = StatsBase.transform(dt.soc, collect(limit_predict[1]:step_predict:limit_predict[2]))
    Y_predict_ocv = StatsBase.reconstruct(dt.v, focv(X_predict_soc))
    μ, Σ = RecursiveGPs.predict(rgp_ocv, X_predict_soc)
    σ = sqrt.(abs.(diag(Σ)))
    μ = StatsBase.reconstruct(dt.v, μ)
    σ = StatsBase.reconstruct(dt.σ, σ)

    # R0_curve
    μ_r, Σ_r = RecursiveGPs.predict(rgp_r, X_predict_soc)
    σ_r = sqrt.(abs.(diag(Σ_r)))
    μ_r = StatsBase.reconstruct(dt.σ, μ_r)
    σ_r = StatsBase.reconstruct(dt.σ, σ_r)


    ## Voltage evolution
    """
    v = df.v[1:100:N_points]

    ocv_v = RecursiveGPs.predict(rgp_ocv, df.soc[1:100:N_points], train=false)[1]
    R0_v = RecursiveGPs.predict(rgp_r, df.soc[1:100:N_points], train=false)[1]
    i = df.i[1:100:N_points]
    if using_real == true
        v_aprox = StatsBase.reconstruct(dt.v, focv(df.soc[1:100:N_points])) + 15e-3 * i + param_evo.vrc
    else
        v_aprox = StatsBase.reconstruct(dt.v, ocv_v) + StatsBase.reconstruct(dt.σ, R0_v) .* i + param_evo.vrc
    end
    """
end


# Plotting
begin

    fig = Figure(size=(1200, 1200))

    # OCV curve
    ax1 = CairoMakie.Axis(fig[1, 1], title="OCV curve", xlabel="SOC", ylabel="V")
    lines!(ax1, collect(limit_predict[1]:step_predict:limit_predict[2]), μ, label="OCV aprox")
    band!(ax1, collect(limit_predict[1]:step_predict:limit_predict[2]), μ - 2σ, μ + 2σ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))
    lines!(ax1, collect(limit_predict[1]:step_predict:limit_predict[2]), Y_predict_ocv, label="OCV real")
    ylims!(ax1, minimum(Y_predict_ocv), maximum(Y_predict_ocv))
    vlines!(ax1, minimum(StatsBase.reconstruct(dt.soc, df.soc[1:N_points])), color=:red, linestyle=:dash)
    vlines!(ax1, maximum(StatsBase.reconstruct(dt.soc, df.soc[1:N_points])), color=:red, linestyle=:dash)
    axislegend(ax1)




    # Parameter evolution curves
    ax2 = CairoMakie.Axis(fig[2, 1], title="Parameter evolution", xlabel="iter", ylabel="R1")
    lines!(ax2, getindex.(param_evo.μ, 1))
    ylims!(ax2, 0.0, 50e-3)
    ax3 = CairoMakie.Axis(fig[3, 1], title="Parameter evolution", xlabel="iter", ylabel="τ1")
    lines!(ax3, getindex.(param_evo.μ, 2))

    hlines!(ax2, 15e-3, color=:red, linestyle=:dash)
    hlines!(ax3, 60, color=:red, linestyle=:dash)
    ylims!(ax3, 30.0, 80.0)
    display(fig)


    # Voltage curve
    """
    ax4 = Axis(fig[5, 1], title="Voltage", xlabel="iter", ylabel="V")

    lines!(ax4, v_aprox, label="Aprox")
    lines!(ax4, v, label="Real", linestyle=:dash)
    #ylims!(ax4, 3.5, 4.2)
    axislegend(ax4)

    # Error curve on every DataInterpolations
    ax5 = Axis(fig[6, 1], title="Voltage", xlabel="iter", ylabel="V")

    lines!(ax5, abs.(v .- v_aprox), label="Error")
    ylims!(ax5, 0.00, 0.0025)
    axislegend(ax5)


    """
end



begin
    fig = Figure(size=(1200, 1200))

    v_Σ = [d.v.Σ[1] for d in param_evo.param_noise]
    wR_Σ = [d.w.Σ[1, 1] for d in param_evo.param_noise]
    wτ_Σ = [d.w.Σ[2, 2] for d in param_evo.param_noise]

    ax1 = CairoMakie.Axis(fig[1, 1], title="Parameter evolution", xlabel="iter", ylabel="Estimation noise")
    lines!(ax1, v_Σ)

    ax2 = CairoMakie.Axis(fig[2, 1], title="Parameter evolution", xlabel="iter", ylabel="R1 noise")
    lines!(ax2, wR_Σ)

    ax3 = CairoMakie.Axis(fig[3, 1], title="Parameter evolution", xlabel="iter", ylabel="τ1 noise")
    lines!(ax3, wτ_Σ)

    display(fig)
end



#####################################################################################################################################################################
###############################################################################################################################################################################

#### Profile 2
begin
    ## normalize_data
    df = CSV.read("data/profile2.csv", DataFrame)

    dt = fit_zscore(df)
    df = normalize_data(df, dt)

    ## Train
    N_points = size(df)[1]
    soc_test = df.soc[1:N_points]
    i_test = df.i[1:N_points]

    v_test = df.v[1:N_points]
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
    σ_f2 = 5e-5
    σ_model = 0.1

    limit_basis_ocv = [0, 1]
    step_basis_ocv = 0.01
    X_basis_ocv = collect(limit_basis_ocv[1]:step_basis_ocv:limit_basis_ocv[2])
    n_basis = size(X_basis_ocv)[1]
    X_basis_r = collect(limit_basis_ocv[1]:step_basis_ocv:limit_basis_ocv[2])


    X_basis_ocv = StatsBase.transform(dt.soc, X_basis_ocv)
    X_basis_r = StatsBase.transform(dt.soc, X_basis_r)



    gp_ocv = gp_ocv = GP(
        LinearKernel() +
        σ_ocv * with_lengthscale(SEKernel(), l_ocv)
    )

    rgp_ocv = RGPModel(gp_ocv, σ_f1, X_basis_ocv)

    gp_r = GP(σ_r * with_lengthscale(SEKernel(), l_r))
    rgp_r = RGPModel(gp_r, σ_f2, X_basis_r)


    #### Building RC
    μ_params = [15e-4, 60.0]

    batt = BATTModel(rgp_ocv, rgp_r, μ_params, dt)
end



@time begin

    batt.i = i_test[1]
    Vrc = Vector{Float64}()
    for (n, batch) in enumerate(data)

        battery_learn!(batt, batch, rts=true)

        if n % 100 == 0
            for k in collect(99:-1:1)
                battery_rts_smoother!(batt, k)
            end
            empty_rts!(batt.histogram_model)
            if n > 600
                break
            end
        end


        #Vrc = push!(Vrc, Vrc1 + Vrc2)
    end
end

begin
    Vrc = false
    fig = plot_profile2(batt, l_ocv, l_r, σ_f1, σ_f2, Vrc, soc_test, dt)
    #save("pictures/Week_21_04_2025/TESTING_profile2_rc_Basis_vectors_mean_0_l$(l_ocv)_sigma$(σ_ocv)_noise$(σ_f1)_n_basis_$(n_basis)_batch_size$(batch_size)_$(name)_$(batt.R1)_$(batt.τ1)_$(batt.R2)_$(batt.τ2).png", fig)
end

