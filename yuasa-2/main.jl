using DataFrames
using CSV
using Dates
using Intervals
using DataInterpolations
using StatsBase

# using GLMakie
using CairoMakie

using BatteryDigitalTwin
using StaticArrays
import ComponentArrays: ComponentVector, ComponentMatrix, getaxes

# include("model_yuasa.jl")
include("fit-model.jl")

# dataset
datadir = "data/data-yuasa-cycles-2/"
files = Dict(
    :cell_voltage => datadir * "cell_voltages.csv",
    :cell_soc => datadir * "cell_soc.csv",
    :module_voltage => datadir * "module_voltage.csv",
    :module_current => datadir * "module_current_average.csv",
    :derating_current => datadir * "derating_currents.csv",
    :battery_temperature => datadir * "battery_temperature.csv"
)

dateformat = dateformat"y-m-d H:M:S+00:00"
data = Dict(id => CSV.File(file; dateformat) |> DataFrame for (id, file) in files)

ti = Interval(DateTime("2025-12-10T14:00:20"), DateTime("2025-12-11T02:30:20"))


ids = [(; p, m, c) for p in 1:3, m in 1:9, c in 1:12] |> vec |> sort
# @time (; kfs, sols) = fit_models(data, ti, ids);
# begin
#     kfs = nothing
#     sols = nothing
#     GC.gc(true)
# end
@time (; models, sols) = fit_models_spawn(data, ti, ids);
fig2 = plot_ecms(models, sols)
# 566.760578 seconds (661.42 M allocations: 992.396 GiB, 17.62% gc time, 4.43% compilation time: 4% of which was recompilation)
# 115.219986 seconds (603.27 M allocations: 989.625 GiB, 45.33% gc time, 85 lock conflicts, 0.26% compilation time)

# now more allocations? 195.934984 seconds (894.08 M allocations: 1.032 TiB, 35.99% gc time, 55 lock conflicts, 268.37% compilation time: 4% of which was recompilation)

for p in 1:3, m in 1:9
    ids = [(; p, m, c) for c in 1:12] |> vec |> sort
    mod = Dict(id => models[id] for id in ids)
    sol = Dict(id => sols[id] for id in ids)
    fig = plot_ecms(mod, sol)
    fig.content[1].title = "Phase $p Module $m"
    fig |> display
end

# phase 1 module 6
for c in 1:12
    id = [(; p, m, c) for c in 1:12] |> vec |> sort
    mod = Dict(id => models[id] for id in ids)
    sol = Dict(id => sols[id] for id in ids)
    fig = plot_ecm(mod, sol)
    fig.content[1].title = "Phase $p Module $m"
    fig |> display
end


ids2 = [(; p, m) for p in 1:3, m in 1:9] |> vec |> sort
(models2, sols2) = fit_modules(data, ti, ids2)
fig3 = plot_ecms(models2, sols2; n = 12)

for id in ids
    plot_rc_param_trajectory(models[id], sols[id]) |> display
end

# f = extract_ocv(models[(; p=3, m=5, c=12)]);
f = let
    df = CSV.File("data/ocv/ocv-yuasa-ees-N208.csv") |> DataFrame
    # df = CSV.File("data/ocv/ocv-yuasa.csv") |> DataFrame
    ocv⁻¹ = LinearInterpolation(df.soc, df.ocv)
    ocv = LinearInterpolation(df.ocv, df.soc)
    (; ocv⁻¹, ocv)
end


vlim = (3.75, 3.9)
param_cells = Dict(
    id =>
        Dict(
            :Q => calc_Q(models[id], sols[id], f.ocv⁻¹; v = vlim),
            :soc => calc_soc0(models[id], sols[id], f.ocv⁻¹; v = vlim),
            :soh => calc_soh(models[id], sols[id], f.ocv⁻¹, 100; v = vlim),
        ) for id in ids
)

param_modules = Dict(
    id =>
        Dict(
            :Q => calc_Q(models2[id], sols2[id], f.ocv⁻¹; v = vlim, n = 12),
            :soc => calc_soc0(models2[id], sols2[id], f.ocv⁻¹; v = vlim, n = 12),
            :soh => calc_soh(models2[id], sols2[id], f.ocv⁻¹, 100; v = vlim, n = 12),
        ) for id in ids2
)

plot_ecms_norm(models, sols, f.ocv⁻¹, f.ocv; vlim)


# fig4 = plot_module_soh(param_cells, param_modules)
# fig5 = plot_cell_soh_hist(param_cells)
# fig6 = plot_module_inhomogenity(param_modules)

fig7 = let id = (; p = 1, m = 1, c = 1)
    plot_sim(models[id], sols[id])
end


let id = (; p = 3, m = 2, c = 1)
    plot_rc_param_trajectory(models[id], sols[id])
end


# plot_q_estimation requires a DataFrame with .q and .t columns — not applicable here
# let id = (; p=3, m=5, c=1)
#     plot_q_estimation(data, sols[id], models[id])
# end
