using DataFrames
using CSV
using Dates
using DataInterpolations
using CairoMakie
using LowLevelParticleFilters
using Distributions
using LinearAlgebra
using MLUtils: DataLoader
using Statistics
using AbstractGPs
using StatsBase
using BenchmarkTools
using Profile
using Revise
using StaticArrays
using Optim
using UnPack
using ForwardDiff
using PreallocationTools
using Dates
using ProgressMeter
include("dataset.jl")
include("plot-dataset.jl")
includet("utils.jl")
includet("../battModel/rc.jl")
includet("../battModel/r0_ocv.jl")
includet("../battModel/batt.jl")
includet("../battModel/rgp.jl")
import ComponentArrays: ComponentVector, getaxes, ComponentMatrix, @static_unpack


files = Dict(
    :cell_voltage => "data-yuasa-cycles/cell_voltage_sbc-stage_container_yuasa50n_1753110875-1753156720.csv",
    :module_voltage => "data-yuasa-cycles/module_voltage_sbc-stage_container_yuasa50n_1753110875-1753156720.csv",
    :module_current_avg => "data-yuasa-cycles/debug_value_sbc-stage_container_yuasa50n_1753110875-1753156720.csv",
    :module_current_rms => "data-yuasa-cycles/module_current_sbc-stage_container_yuasa50n_1753110875-1753156720.csv",
    :module_temperature => "data-yuasa-cycles/module_temperature_sbc-stage_container_yuasa50n_1753110875-1753156720.csv",
    :phase_current => "data-yuasa-cycles/phase_current_sbc-stage_container_yuasa50n_1753110875-1753156720.csv",
)

data = load_yuasa_dataset(files)

tr = DateTime("2025-07-21T15:30"):Second(1):DateTime("2025-07-21T23:30") # timerange
dfs = make_module_dataframes(data, tr; cell_timestamps=false)

# Model without an RC
begin
    cell = 1
    m = 1
    p = 1
    r = Dict()
    r[:m] = Dict()
    df = dfs[(; p, m)]

    df_cell = df[:, [:module_current, :module_temperature, :time]]
    df_cell = rename(df,
        [:module_current => :i,
            :module_temperature => :T,
            :time => :t,
            Symbol("cell_voltage_$cell") => :v]
    )

    dt = fit_zscore(df_cell)
    df_cell = normalize_data(df_cell, dt)

    battModel_no_rc = create_batt_no_rc(df_cell, dt)
    N_points = size(df_cell, 1)

    r[:m][:p] = fit_module(df_cell, battModel_no_rc, dt)
end


# Plotting
cell = r[:m][:p]
plot_soc(cell)
plot_ocv(cell)
plot_r0(cell)
# Model with an RC

df_ocv = CSV.File("data/ocv.csv") |> DataFrame
focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc; extrapolation=ExtrapolationType.Constant)


begin
    p = 1
    ms = 1
    df_ocv = CSV.File("data/ocv.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc; extrapolation=ExtrapolationType.Constant)
    r = Dict()

    df_ocv = CSV.File("../../data/ocv.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc; extrapolation=ExtrapolationType.Constant)
    mean_r0(b) = 15e-3

    df = dfs[(; p, m)]
    battModel_no_rc = create_batt_rc(battModel_no_rc, mean_ocv, mean_r)
    r.[:m][:p] = fit_module(df, battModel, N_points)

end

# Plotting
plot_soc(r)
plot_ocv(r)
plot_r0(r)
plot_rc(r)



