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
includet("src/rgp_with_hyp.jl")
using .RecursiveGPsHYP
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

### Synthetic
function learn_2!(rgp_ocv::RGPModel, batch, i, μ, Σ)
    R0 = 15e-3
    R1 = 15e-3
    τ1 = 60.0
    R2 = 15e-3
    τ2 = 600.0

    ## Motion model
    ts = 1 ## PLaceholder, to be changed for new data.
    N_basis = size(rgp_ocv.X_basis, 1)

    A_batt = vcat(
        [I(N_basis) zeros(N_basis, 2)],
        [zeros(1, N_basis) exp(-ts / (τ1)) 0],
        [zeros(1, N_basis) 0 exp(-ts / (τ2))]
    )

    B_batt = [
        zeros(N_basis, 1);
        R1 * (1 - exp(-ts / (τ1)));
        R2 * (1 - exp(-ts / (τ2)))
    ]

    Q_batt = vcat(
        [zeros(N_basis, N_basis) zeros(N_basis, 2)],
        [zeros(1, N_basis) 10e-6 0],
        [zeros(1, N_basis) 0 10e-6]
    )

    μ_predict = A_batt * μ + B_batt * i
    Σ_predict = A_batt * Σ * A_batt' + Q_batt

    ## Update
    μ = μ_predict
    Σ = Σ_predict

    ocv = RecursiveGPs.predict(rgp_ocv, batch.x.soc)
    e = batch.v - (ocv.μ .+ R0 * batch.x.i .+ μ[end] .+ μ[end-1])
    H_ocv = cov(rgp_ocv.gp, batch.x.soc, rgp_ocv.X_basis) * rgp_ocv.inv_cov
    H_rc1 = 1
    H_rc2 = 1

    H = [H_ocv H_rc1 H_rc2]
    σ_model = 0.0
    S = H * Σ * H' + (ocv.Σ .+ σ_model^2) * I(size(batch.v, 1))
    Gk = Σ * H' * inv(S)
    new_μ = μ + Gk * (e)
    new_Σ = Σ - Gk * H * Σ

    ## Updating model
    size_ocv = size(rgp_ocv.μ)[1]

    rgp_ocv.μ = new_μ[1:size_ocv]
    rgp_ocv.Σ = new_Σ[1:size_ocv, 1:size_ocv]


    Σ = new_Σ
    μ = new_μ

    return μ, Σ
end


begin
    R0 = 15e-3
    normalize = false
    df_with_rc = CSV.read("data/output_data_with_rc.csv", DataFrame)
    df_profile = CSV.read("data/profile.csv", DataFrame)
    df_ocv = CSV.read("data/ocv.csv", DataFrame)

    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc; extrapolation=ExtrapolationType.Constant)
    fi = ConstantInterpolation(df_profile.i, df_profile.t)

    df = DataFrame(
        t=df_with_rc.time,
        v=df_with_rc.voltage - R0 * fi.(df_with_rc.time),
        i=fi.(df_with_rc.time),  # interpolated current
        soc=df_with_rc.soc
    )

    if normalize == true
        dt = fit_zscore(df)
        df = normalize_data(df, dt)
    end

    N_points = size(df_with_rc.soc)[1]
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


begin
    l_ocv = 0.2

    σ_ocv = 0.1

    σ_f1 = 0.01

    limit_basis_ocv = [0, 1]
    step_basis_ocv = 0.01

    mean_ = x -> StatsBase.transform(dt.v, [0.0])[1]
    X_basis_ocv = collect(limit_basis_ocv[1]:step_basis_ocv:limit_basis_ocv[2])
    n_basis = size(X_basis_ocv)[1]
    kernel = GP(σ_ocv * with_lengthscale(SEKernel(), l_ocv))

    rgp_ocv = RGPModel(kernel, σ_f1, X_basis_ocv)
    R0 = 15e-3
end

begin
    ### Uncomment if Augmented Kalman filter with RC
    ### Change DataLoader voltage to be v and not ocv, and learn_2! instead of learn!
    #filler = zeros(size(rgp_ocv.Σ, 1), 2)
    #filler_rc = zeros(1, size(rgp_ocv.Σ, 1))


    #μ = [rgp_ocv.μ; 0; 0]
    #Q_vrc = 1e-4
    #Σ = vcat(
    #[rgp_ocv.Σ filler],
    #[filler_rc Q_vrc 0],
    #[filler_rc 0 Q_vrc]
    #)
    #i = df.i[1]

    for (n, batch) in enumerate(data)
        #μ, Σ = learn_2!(rgp_ocv, batch, i, μ, Σ)
        RecursiveGPs.learn!(rgp_ocv, batch.x.soc, batch.v)
        #i = batch.x.i
    end
end

begin
    limit_predict = [0, 1]
    step_predict = 0.015
    X_predict_soc = collect(limit_predict[1]:step_predict:limit_predict[2])

    Y_predict_ocv = focv(X_predict_soc)


    μ, Σ = RecursiveGPs.predict(rgp_ocv, X_predict_soc)
    σ = sqrt.(abs.(diag(Σ)))

    #μ = StatsBase.reconstruct(dt.v, μ)
    #σ = StatsBase.reconstruct(dt.σ, σ)

    fig = Figure(size=(1200, 600))
    ax1 = Axis(fig[1, 1], title="GP updated for ocv", xlabel="soc", ylabel="ocv")

    lines!(ax1, X_predict_soc, μ, label="OCV aprox")
    band!(ax1, X_predict_soc, μ - 2σ, μ + 2σ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))

    lines!(ax1, X_predict_soc, Y_predict_ocv, label="Real")
    ylims!(ax1, minimum(focv.u), maximum(focv.u))
    vlines!(ax1, [minimum(X_soc), maximum(X_soc)], color=:red)
    axislegend(ax1)



    if normalize == true
        name = "normalized"
    else
        name = "non_normalized"
    end

    #save("pictures/Week_14_04_2025/Hyperparameter/synthetic_standard_l$(l_ocv)_sigma$(σ_ocv)_noise$(σ_f1)_n_basis_$(n_basis)_batch_size$(batch_size)_$(name)_stepping.png", fig)
    display(fig)
end


#####################################################################################################################################################################
###############################################################################################################################################################################

#### Profile 2
begin
    ## normalize_data
    df = CSV.read("data/profile2.csv", DataFrame)
    normalize = false

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

    gp_ocv = gp_ocv = GP(
        LinearKernel() +
        σ_ocv * with_lengthscale(SEKernel(), l_ocv)
    )
    rgp_ocv = RGPModel(gp_ocv, σ_f1, X_basis_ocv)

    gp_r = GP(σ_r * with_lengthscale(SEKernel(), l_r))
    rgp_r = RGPModel(gp_r, σ_f2, X_basis_r)

    batt = BATTModel(rgp_ocv, rgp_r, σ_model)
end

begin
    ##Train
    batt.i = i_test[1]
    for (n, batch) in enumerate(data)
        battery_learn_rc!(batt, batch)
    end
end


begin
    ##PLot
    limit_predict = [0, 1]
    step_predict = 0.01

    X_predict_soc = collect(limit_predict[1]:step_predict:limit_predict[2])
    rgp_ocv = batt.rgp_ocv
    rgp_r = batt.rgp_r

    μ_ocv = RecursiveGPs.predict(rgp_ocv, X_predict_soc).μ
    Σ_ocv = RecursiveGPs.predict(rgp_ocv, X_predict_soc).Σ
    var_ocv = sqrt.(abs.(diag(Σ_ocv)))
    μ_r = RecursiveGPs.predict(rgp_r, X_predict_soc).μ
    Σ_r = RecursiveGPs.predict(rgp_r, X_predict_soc).Σ
    var_r = sqrt.(abs.(diag(Σ_r)))

    V_p = RecursiveGPs.predict(rgp_ocv, soc_test[1:50:end]).μ + i_test[1:50:end] .* RecursiveGPs.predict(rgp_r, soc_test[1:50:end]).μ
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

    ylims!(ax2, 0.0005, 0.0013)

    if normalize == true
        name = "normalized"
    else
        name = "non_normalized"
    end


    #save("pictures/Week_14_04_2025/Adding_Vrc/profile2_rc_l$(l_ocv)_sigma$(σ_ocv)_noise$(σ_f1)_n_basis_$(n_basis)_batch_size$(batch_size)_$(name)_$(batt.R1)_$(batt.τ1)_$(batt.R2)_$(batt.τ2).png", fig)
    display(fig)
end

