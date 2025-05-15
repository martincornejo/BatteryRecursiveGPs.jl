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
includet("src/rgp.jl")
includet("src/battModel.jl")
includet("src/plot/profile2.jl")
includet("src/synthetic.jl")
using .RecursiveGPs
using .battModel

function fit_zscore(df)
    v = StatsBase.fit(ZScoreTransform, df.v)
    σ = StatsBase.fit(ZScoreTransform, df.v, center=false)
    i = StatsBase.fit(ZScoreTransform, df.i, center=false)
    soc = StatsBase.fit(ZScoreTransform, df.soc)
    return (; v, σ, i, soc)
end

function normalize_data(df, dt)
    v = StatsBase.transform(dt.v, df.v)
    i = StatsBase.transform(dt.i, df.i)
    soc = StatsBase.transform(dt.soc, df.soc)
    return DataFrame(; df.t, v, i, soc)
end


begin
    RC = true
    if RC == true
        df_data = CSV.read("data/output_data_with_rc.csv", DataFrame)

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
    df.v = StatsBase.reconstruct(dt.v, df.v)
    df.i = StatsBase.reconstruct(dt.i, df.i)
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


    batch_size = 1
    data = DataLoader((
            x=(
                soc=X_soc,
                i=X_i
            ),
            v=Y_v
        ),
        batchsize=batch_size, shuffle=false
    )
end

## Building GP
begin
    l_ocv = 0.2

    σ_ocv = 0.1

    σ_f1 = 0.01

    limit_basis_ocv = [0, 1]
    step_basis_ocv = 0.01

    X_basis_ocv = collect(limit_basis_ocv[1]:step_basis_ocv:limit_basis_ocv[2])
    X_basis_ocv = StatsBase.transform(dt.soc, collect(limit_basis_ocv[1]:step_basis_ocv:limit_basis_ocv[2]))


    n_basis = size(X_basis_ocv)[1]
    kernel = GP(σ_ocv * with_lengthscale(SEKernel(), l_ocv) + LinearKernel())

    rgp_ocv = RGPModel(kernel, σ_f1, X_basis_ocv)

end


## Training
begin
    using_real = false
    # RC set-up
    if RC == true
        filler = zeros(size(rgp_ocv.Σ, 1), 1)
        filler_rc = zeros(1, size(rgp_ocv.Σ, 1))

        if using_real == true
            μ = [0.0]
            Q_vrc = 1e-4
            Σ = [Q_vrc]


        else
            μ = [rgp_ocv.μ; 0.0]
            Q_vrc = 1e-4
            Σ = vcat(
                [rgp_ocv.Σ filler],
                [filler_rc Q_vrc],
            )
        end
        i = df.i[1]
        soc = df.soc[1]
        μ_params_0 = [40, 40]
        μ_params = μ_params_0
        cov_noise = 1e-6
        Σ_params = cov_noise * I(size(μ_params, 1))
        ## Parameter evolution
        param_evo = DataFrame(
            μ=Any[],
            Σ=Any[],
            vrc=Float64[]
        )
        push!(param_evo, (μ=μ_params, Σ=Σ_params, vrc=0.0))
    end



    for (n, batch) in enumerate(data)
        if n % 10 == 0
            push!(param_evo, (
                μ=μ_params,
                Σ=Σ_params,
                vrc=μ[end],
            )
            )
        end
        μ, Σ, μ_params, Σ_params = dual_kf!(rgp_ocv, batch, i, μ, Σ, μ_params, Σ_params, dt, update_params=true)
        α_curr = (
            R1=μ_params[1],
            τ1=μ_params[2]
        )
        i = batch.x.i[1]
        soc = batch.x.soc
    end
end



begin
    ## OCV curve
    limit_predict = [0, 1]
    step_predict = 0.015
    X_predict_soc = StatsBase.transform(dt.soc, collect(limit_predict[1]:step_predict:limit_predict[2]))
    Y_predict_ocv = StatsBase.reconstruct(dt.v, focv(X_predict_soc))
    μ, Σ = RecursiveGPs.predict(rgp_ocv, X_predict_soc, train=false)
    σ = sqrt.(abs.(diag(Σ)))
    μ = StatsBase.reconstruct(dt.v, μ)
    σ = StatsBase.reconstruct(dt.σ, σ)

    ## Voltage evolution
    v = df.v[1:10:end]

    ocv_v = RecursiveGPs.predict(rgp_ocv, df.soc[1:10:end], train=false)[1]
    i = df.i[1:10:end]
    if using_real == true
        v_aprox = StatsBase.reconstruct(dt.v, focv(df.soc[1:10:end])) + 15e-3 * i + param_evo.vrc
    else
        v_aprox = StatsBase.reconstruct(dt.v, ocv_v) + 15e-3 * i + param_evo.vrc
    end


    fig = Figure(size=(1200, 1200))

    # OCV curve
    ax1 = Axis(fig[1, 1], title="OCV curve", xlabel="SOC", ylabel="V")
    lines!(ax1, collect(limit_predict[1]:step_predict:limit_predict[2]), μ, label="OCV aprox")
    band!(ax1, collect(limit_predict[1]:step_predict:limit_predict[2]), μ - 2σ, μ + 2σ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))
    lines!(ax1, collect(limit_predict[1]:step_predict:limit_predict[2]), Y_predict_ocv, label="OCV real")
    ylims!(ax1, minimum(Y_predict_ocv), maximum(Y_predict_ocv))
    axislegend(ax1)


    # Parameter evolution curves
    ax2 = Axis(fig[2, 1], title="Parameter evolution", xlabel="iter", ylabel="R1")
    lines!(ax2, getindex.(param_evo.μ, 1))
    ax3 = Axis(fig[3, 1], title="Parameter evolution", xlabel="iter", ylabel="τ1")
    lines!(ax3, getindex.(param_evo.μ, 2))

    hlines!(ax2, 15, color=:red, linestyle=:dash)
    hlines!(ax3, 60, color=:red, linestyle=:dash)

    if size(μ_params, 1) == 4
        ax6 = Axis(fig[6, 1], title="Parameter evolution", xlabel="iter", ylabel="R1")
        lines!(ax6, getindex.(param_evo.μ, 3))
        ax7 = Axis(fig[7, 1], title="Parameter evolution", xlabel="iter", ylabel="τ1")
        lines!(ax7, getindex.(param_evo.μ, 4))

        hlines!(ax6, 15, color=:red, linestyle=:dash)
        hlines!(ax7, 600, color=:red, linestyle=:dash)
    end


    # Voltage curve
    ax4 = Axis(fig[4, 1], title="Voltage", xlabel="iter", ylabel="V")

    lines!(ax4, v_aprox, label="Aprox")
    lines!(ax4, v, label="Real", linestyle=:dash)
    #ylims!(ax4, 3.5, 4.2)
    axislegend(ax4)

    # Error curve on every DataInterpolations
    ax5 = Axis(fig[5, 1], title="Voltage", xlabel="iter", ylabel="V")

    lines!(ax5, abs.(v .- v_aprox), label="Error")
    #ylims!(ax5, 0.00, 0.02)
    axislegend(ax5)


    #save("pictures/Week_28_04_25/different_noise_synthetic_Sig$(cov_noise)_l$(l_ocv)_sigma$(σ_ocv)_noise$(σ_f1)_R1_$(μ_params_0[1])_τ1_$(μ_params_0[2])_model_noise1e1.png", fig)
    display(fig)
end


#####################################################################################################################################################################
###############################################################################################################################################################################

#### Profile 2
begin
    ## normalize_data
    df = CSV.read("data/profile2.csv", DataFrame)
    normalize = true

    if normalize == true
        dt = fit_zscore(df)
        df = normalize_data(df, dt)
    end

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

    if normalize == true
        X_basis_ocv = StatsBase.transform(dt.soc, X_basis_ocv)
        X_basis_r = StatsBase.transform(dt.soc, X_basis_r)
    end


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

begin
    ##Train
    batt.i = i_test[1]
    Vrc = Vector{Float64}()
    for (n, batch) in enumerate(data)
        battery_learn!(batt, batch)
        #Vrc = push!(Vrc, Vrc1 + Vrc2)
    end
end


begin
    Vrc = false
    fig = plot_profile2(batt, l_ocv, l_r, σ_f1, σ_f2, Vrc, soc_test; dt=dt, normalize=normalize)

    if normalize == true
        name = "normalized"
    else
        name = "non_normalized"
    end

    #save("pictures/Week_21_04_2025/TESTING_profile2_rc_Basis_vectors_mean_0_l$(l_ocv)_sigma$(σ_ocv)_noise$(σ_f1)_n_basis_$(n_basis)_batch_size$(batch_size)_$(name)_$(batt.R1)_$(batt.τ1)_$(batt.R2)_$(batt.τ2).png", fig)
end

