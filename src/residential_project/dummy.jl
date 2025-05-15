using Distributions
using Dates
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
includet("../rgp.jl")
includet("../battModel.jl")
using .RecursiveGPs
using .battModel
using JLD2


function fit_zscore(df)
    v = StatsBase.fit(ZScoreTransform, df.v)
    σ = StatsBase.fit(ZScoreTransform, df.v, center=false)
    i = StatsBase.fit(ZScoreTransform, df.i, center=false)
    soc = StatsBase.fit(ZScoreTransform, df.q)
    return (; v, σ, i, soc)
end

function normalize_data(df, dt)
    v = StatsBase.transform(dt.v, df.v)
    i = StatsBase.transform(dt.i, df.i)
    q = StatsBase.transform(dt.soc, df.q)
    return DataFrame(; df.t, v, i, q)
end



function addCellVoltage(df, sys_id)

    meta = CSV.read("src/residential_project/nature_data/00_Data/00_Metadata/Metadata_Systems.csv", DataFrame)


    # Convert sys_id to integer for comparison
    sys_id_num = parse(Int, sys_id)

    # Find the matching row
    row_id = meta.ID .== sys_id_num

    # Get the number of cells in series
    cells_series = meta.Cell_number_in_series[row_id][1]

    # Add new column for cell voltage
    df.V_cell_in_V = df.V_in_V ./ cells_series

    # Return the modified dataframe
    return df
end



function addIOffset(df, correction=0.0)
    I_offset = mean(df.I_in_A) + correction
    return I_offset
end

function addCellQ(df, I_offset2=0.0, plot=true)
    I_offset = addIOffset(df)
    df.i = df.I_in_A .- (I_offset + I_offset2)
    df.q = cumsum(df.I_in_A .- I_offset) .* (60 / 3600)
    df.q = df.q .+ abs(minimum(df.q))


    if plot == true
        fig = Figure(size=(1200, 800))
        ax1 = Axis(fig[1, 1], title="cum sum I offset $(I_offset)")
        ax2 = Axis(fig[1, 2], title="Q")
        ax3 = Axis(fig[2, 1], title="V")
        ax4 = Axis(fig[2, 2], title="I")
        lines!(ax1, cumsum(df.i))
        lines!(ax2, df.q)
        lines!(ax3, df.V_cell_in_V)
        lines!(ax4, df.i)
        display(fig)
    end

    return df
end


## Loading relevant data
begin
    I_offsets2 = CSV.read("src/residential_project/nature_data/I_offsets.csv", DataFrame)
    rgp_models_ocv = Dict{String,RGPModel}()
    rgp_models_r = Dict{String,RGPModel}()
end


# Year evolution
# Months of interest
### Loading data and all that stuff
begin
    load_months = ["01", "02", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    data_yearly = DataFrame()
    for month in load_months
        sys_id = "11"
        year = "2021"
        time_step = "minute"
        folder_path = "src/residential_project/nature_data/00_Data/01_Operational_Data/Data_ID_$(sys_id)/$(sys_id)/$(year)_$(month)_System_ID_$(sys_id).csv"
        df = CSV.read(folder_path, DataFrame)
        df.Time = DateTime.(df.Time, "dd-u-yyyy HH:MM:SS")

        if time_step == "minute"
            df = filter(row -> Dates.second(row.Time) == 0, df)
            df.Time = (df.Time .- minimum(df.Time)) / 1000
            df.Time = parse.(Float64, replace.(string.(df.Time), " milliseconds" => ""))
        end
        I_offset2 = I_offsets2[I_offsets2.month.==parse(Int, month), :offset][1]
        df = addCellVoltage(df, sys_id)
        df = addCellQ(df, I_offset2, false)

        select!(df, Not(:P_in_W, :V_in_V, :T_Bat_in_C, :T_Room_in_C, :Interpolated))
        rename!(df, :V_cell_in_V => :v, :Time => :t)

        df.month = fill(month, nrow(df))  # Add month column
        append!(data_yearly, df)
    end

    data_yearly = groupby(data_yearly, :month)
end





## OCV and R
begin

    train_months = ["01"]

    ## Train percentage per month
    train = 1.0

    ## GP parameters
    n_basis_ocv = 101
    l_ocv = 0.1
    σ_ocv = 0.1
    l_r = 0.2
    σ_r = 0.5

    σ_f1 = 0.1
    σ_f2 = 5e-4
    σ_model = 0.1

    soc_floor = 40
    ocv_floor = 3.4


    fig = Figure(size=(1200, 800))
    ax1 = Axis(fig[1, 1], title="GP updated for ocv l = $(l_ocv), σ =$(σ_ocv), noise = $(σ_f1) ", xlabel="soc", ylabel="ocv")
    ax2 = Axis(fig[2, 1], title="GP updated for R0 l = $(l_r), σ =$(σ_r), noise = $(σ_f2) ", xlabel="soc", ylabel="R0")
    normal = true
    for month in train_months
        df = data_yearly[(month,)]
        ## Setting up training
        dt = fit_zscore(df)
        df = normalize_data(df, dt)
        n_train = Int(train * size(df, 1))
        df_train = df[1:n_train, :]
        df_test = df[n_train+1:end, :]


        ## Building GPs
        limit_basis_ocv = [minimum(df.q), maximum(df.q)]
        X_basis_ocv = collect(range(limit_basis_ocv[1], limit_basis_ocv[2], length=101))
        n_basis = size(X_basis_ocv)[1]
        X_basis_r = collect(range(limit_basis_ocv[1], limit_basis_ocv[2], length=101))

        gp_ocv = gp_ocv = GP(
            LinearKernel() +
            σ_ocv * with_lengthscale(SEKernel(), l_ocv)
        )

        rgp_ocv = RGPModel(gp_ocv, σ_f1, X_basis_ocv)

        gp_r = GP(σ_r * with_lengthscale(SEKernel(), l_r))
        rgp_r = RGPModel(gp_r, σ_f2, X_basis_r)

        batt = BATTModel(rgp_ocv, rgp_r, false, dt)



        ## Training loop
        data_train = DataLoader((
                x=(
                    soc=df_train.q,
                    i=df_train.i
                ),
                y=df_train.v
            ),
            batchsize=1, shuffle=false
        )

        for (n, batch) in enumerate(data_train)
            battery_learn!(batt, batch)
        end
        rgp_models_ocv[month] = rgp_ocv
        rgp_models_r[month] = rgp_r



        println(StatsBase.reconstruct(dt.soc, [minimum(df.q)]))
        println(StatsBase.reconstruct(dt.soc, [maximum(df.q)]))
        ## Ploting
        color_value = (12 - parse(Int, month) + 1) / 11  # 1 for first month, 1/12 for last month
        line_color = Makie.RGBA(0, 0, 1, color_value)  # Strong blue
        band_color = Makie.RGBA(0, 0, 0.8, 0.3 * color_value)

        X_predict_soc = StatsBase.transform(dt.soc,
            collect(range(0, 250, length=101))
        )
        X_predict_soc_n = X_predict_soc

        μ_ocv, Σ_ocv = RecursiveGPs.predict(rgp_ocv, X_predict_soc)
        var_ocv = sqrt.(abs.(diag(Σ_ocv)))
        μ_r, Σ_r = RecursiveGPs.predict(rgp_r, X_predict_soc)
        var_r = sqrt.(abs.(diag(Σ_r)))

        if normal == true
            μ_ocv = StatsBase.reconstruct(dt.v, μ_ocv)
            var_ocv = StatsBase.reconstruct(dt.σ, var_ocv)
            μ_r = StatsBase.reconstruct(dt.σ, μ_r)
            var_r = StatsBase.reconstruct(dt.σ, var_r)
            X_predict_soc_n = StatsBase.reconstruct(dt.soc, X_predict_soc)
        end

        ## Flooring
        soc_at_floor = X_predict_soc_n[findfirst(x -> x ≥ ocv_floor, μ_ocv)]
        X_predict_soc_n = X_predict_soc_n .+ (soc_floor - soc_at_floor)


        lines!(ax1, X_predict_soc_n, μ_ocv, label="OCV $(month)")
        band!(ax1, X_predict_soc_n, μ_ocv - 2var_ocv, μ_ocv + 2var_ocv)
        #vlines!(ax1, StatsBase.reconstruct(dt.soc, [minimum(df.q)]); color=:red, linestyle=:dash)
        #vlines!(ax1, StatsBase.reconstruct(dt.soc, [maximum(df.q)]); color=:red, linestyle=:dash)
        ylims!(ax1, 3.2, 4.3)

        ## R0
        lines!(ax2, X_predict_soc_n, μ_r, label="R0 $(month)")
        band!(ax2, X_predict_soc_n, μ_r - 2var_r, μ_r + 2var_r)
        #vlines!(ax2, StatsBase.reconstruct(dt.soc, [minimum(df.q)]); color=:red, linestyle=:dash)
        #vlines!(ax2, StatsBase.reconstruct(dt.soc, [maximum(df.q)]); color=:red, linestyle=:dash)
        ylims!(ax2, -0.2, 0.2)
        println("Month $(month) done")
    end

    axislegend(ax1)
    axislegend(ax2)
    display(fig)
end



begin
    n_train = 0.8
    sys_id = 11
    for month in ["04", "05", "06", "07", "08", "09"]
        df = data_yearly[(month,)]
        ## Setting up training
        dt = fit_zscore(df)
        df = normalize_data(df, dt)

        n_train = Int(train * size(df, 1))
        df_train = df[1:n_train, :]
        df_test = df[n_train+1:end, :]
        rgp_ocv = rgp_models_ocv[month]
        rgp_r = rgp_models_r[month]

        ## Generating images
        Step_comp = 12

        x_total = collect(1:Step_comp:size(df, 1))

        V_aprox_train = RecursiveGPs.predict(rgp_ocv, df_train.q[1:Step_comp:end])[1] +
                        df_train.i[1:Step_comp:end] .* RecursiveGPs.predict(rgp_r, df_train.q[1:Step_comp:end])[1]
        V_aprox_test = RecursiveGPs.predict(rgp_ocv, df_test.q[1:Step_comp:end])[1] +
                       df_test.i[1:Step_comp:end] .* RecursiveGPs.predict(rgp_r, df_test.q[1:Step_comp:end])[1]

        V_real = df.v[1:Step_comp:end]

        if normal == true
            V_aprox_train = StatsBase.reconstruct(dt.v, V_aprox_train)
            V_aprox_test = StatsBase.reconstruct(dt.v, V_aprox_test)
            V_real = StatsBase.reconstruct(dt.v, V_real)
        end
        V_aprox = vcat(
            V_aprox_train,
            V_aprox_test)
        ## Plooting
        fig = Figure(size=(1200, 800))
        x_total = collect(1:Step_comp:size(df, 1))

        ax1 = Axis(fig[1, 1], title="Train data month $(month)")
        lines!(ax1, V_aprox, label="V aprox")
        lines!(ax1, V_real, label="V real")
        ylims!(ax1, 3.2, 4.2)
        vlines!(ax1, n_train / Step_comp; color=:red, linestyle=:dash)

        ax2 = Axis(fig[2, 1], title="Error")
        lines!(ax2, abs.(V_real - V_aprox), label="Abs error")
        vlines!(ax2, n_train / Step_comp; color=:red, linestyle=:dash)


        axislegend(ax1)
        axislegend(ax2)
        display(fig)
        save("ID$(sys_id)_$(year)_$(month)_$(n_train).png", fig)
    end
end



begin
    jldsave("rgp_models.jld2"; rgp_models=rgp_models)
end




