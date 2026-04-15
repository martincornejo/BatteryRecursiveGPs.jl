using CSV
using DataFrames
using Dates

using Printf
using Intervals
using StatsBase
using DataInterpolations
using StaticArrays
using Measurements

using CairoMakie

# Worker setup for distributed fitting (run only once)
using Distributed
addprocs(Sys.CPU_THREADS ÷ 2)
@everywhere begin
    using BatteryDigitalTwin
    include("fit-model.jl")
end
include("parallel.jl") # for threaded and distributed fit

include("dataset.jl")
include("../yuasa-ocv/ocv.jl")
include("../yuasa-ocv/analysis.jl")
include("../yuasa-ocv/plot.jl")

# === load data ===
dateformat = dateformat"y-m-dTHH:MM:SS.sss+00:00"
data = CSV.File("data/data-yuasa-peak-shaving/combined_log_20260327_082612.csv"; dateformat) |> DataFrame

profile = CSV.File("data/data-yuasa-peak-shaving/power_profile_peak_shaving_yuasa_1.csv") |> DataFrame


# === fit models ===
ti = Interval(DateTime("2026-03-27T08:43:00"), DateTime("2026-03-27T19:13:00"))
ids = [(; m, c) for m in 1:9, c in 1:12] |> vec |> sort
# ids = [(; m, c) for m in 1:1, c in 1:12] |> vec |> sort


@time (; models, sols) = fit_models_distributed(data, ti, ids);
rmprocs(workers()) # free processes
# begin
#     models = sols = nothing
#     @everywhere GC.gc(true)
#     @everywhere ccall(:malloc_trim, Int32, (Int32,), 0)
# end
# @everywhere @info Base.gc_live_bytes() / 1024^3  # live objects in GiB
# @time (; models, sols) = fit_models_threaded(data, ti, ids);

plot_ecms(models, sols) |> display

# === Composite OCV from GP posteriors ===

cells = map(ids) do id
    gp_ocv(models[id], sols[id])
end

fit = fit_composite_ocv(cells; n_v_grid = 100)

order = sortperm(fit.soc_grid)
composite = LinearInterpolation(
    fit.v_grid[order], fit.soc_grid[order];
    extrapolation = ExtrapolationType.Constant,
)

eval_fit_parameters(fit)
eval_cell_parameters(fit; v_ref = (3.5, 4.05), soc_ref = (0.1, 0.85))

ocvs = map(cells) do c
    LinearInterpolation(c.μ, c.q; extrapolation = ExtrapolationType.Constant)
end
eval_ocv_residuals(composite, ocvs, fit.params)
eval_soc_range(composite)

let fig = plot_composite_ocv(fit, cells)
    fig.content[1].title = "Composite OCV from GP posteriors (Module 1)"
    fig |> display
end

plot_ocv_residuals(fit, cells) |> display
