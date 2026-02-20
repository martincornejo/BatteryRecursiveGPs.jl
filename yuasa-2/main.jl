using DataFrames
using CSV
using Dates
using Intervals
using DataInterpolations
using StatsBase

# using GLMakie
using CairoMakie

using Measurements
using StaticArrays
using RecursiveGPs
using AbstractGPs
using LowLevelParticleFilters
import LowLevelParticleFilters as LLPF
using ForwardDiff
using LinearAlgebra
import ComponentArrays: ComponentVector, ComponentMatrix, getaxes

# using LogExpFunctions
# using Optimization
# using OptimizationOptimJL
# using LineSearches

# using NonlinearSolve

include("model_yuasa.jl")
include("fit-model.jl")

# dataset
datadir = "yuasa-2/data-yuasa-cycles-2/"
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

ti = Interval(DateTime("2025-12-10T14:02:00"), DateTime("2025-12-11T02:00:00"))

function fit_model(data, ti, id)
    θ0 = ComponentVector(; # tunable (hyper)params
        ocv=(; σ=0.5, ℓ=0.7),
        r0=(; σ=0.05, ℓ=0.5),
        vσ=3e-3,
    )
    ϑ = ComponentVector(; # non-tunable params
        Ts=1.0,
        r0μ=1.0e-3,
        rc=(;
            v0=0.0, σ0_v=1e-3, σ1_v=1.0e-4,
            r0=1.0e-3, σ0_r=0.5e-3, σ1_r=0.0,
            τ0=300.0, σ0_τ=30.0, σ1_τ=0.0,
        ),
    )
    θ = ComponentVector(; θ0..., ϑ...)

    (; p, m, c) = id
    u, y = cell_dataset(data, ti, p, m, c)
    zt = fit_zscore()
    kf = build_kf(θ, u, zt)

    stats = @timed begin
        sol = run_kf!(kf, u, y)
    end

    @info "Cell p:$(p), m:$(m), c:$(c) complete" stats.time

    (; kf, sol)
end

function fit_module(data, ti, id)
    n = 12
    θ0 = ComponentVector(; # tunable (hyper)params
        ocv=(; σ=0.5, ℓ=0.7),
        r0=(; σ=0.05, ℓ=0.5),
        vσ=n * 3e-3,
    )
    ϑ = ComponentVector(; # non-tunable params
        Ts=1.0,
        r0μ=n * 1.0e-3,
        rc=(;
            v0=n * 0.0, σ0_v=n * 1e-3, σ1_v=n * 1.0e-4,
            r0=n * 1.0e-3, σ0_r=n * 0.5e-3, σ1_r=n * 0.0,
            τ0=300.0, σ0_τ=30.0, σ1_τ=0.0,
        ),
    )
    θ = ComponentVector(; θ0..., ϑ...)

    (; p, m) = id
    zt = fit_zscore(n)
    u, y = module_dataset(data, ti, p, m)
    kf = build_kf(θ, u, zt)

    stats = @timed begin
        sol = run_kf!(kf, u, y)
    end

    @info "Module p:$(p), m:$(m), complete" stats.time

    (; kf, sol)
end

function fit_modules(data, ti, ids)
    kfs = Dict()
    sols = Dict()

    for id in ids
        (; kf, sol) = fit_module(data, ti, id)
        kfs[id] = kf
        sols[id] = sol
    end

    (; kfs, sols)
end

function fit_models(data, ti, ids)
    kfs = Dict()
    sols = Dict()

    for id in ids
        (; kf, sol) = fit_model(data, ti, id)
        kfs[id] = kf
        sols[id] = sol
    end

    (; kfs, sols)
end

function fit_models_spawn(data, ti, ids)
    kfs = Dict()
    sols = Dict()

    for batch in Iterators.partition(ids, Threads.nthreads())
        tasks = Dict(id => Threads.@spawn fit_model(data, ti, id) for id in batch)

        for (id, task) in tasks
            (; kf, sol) = fetch(task)
            kfs[id] = kf
            sols[id] = sol
        end
    end

    (; kfs, sols)
end



ids = [(; p, m, c) for p in 1:3, m in 1:9, c in 1:12] |> vec |> sort
# @time (; kfs, sols) = fit_models(data, ti, ids);
# begin
#     kfs = nothing
#     sols = nothing
#     GC.gc(true)
# end
@time (; kfs, sols) = fit_models_spawn(data, ti, ids);
fig2 = plot_ecms(kfs, sols)
# 566.760578 seconds (661.42 M allocations: 992.396 GiB, 17.62% gc time, 4.43% compilation time: 4% of which was recompilation)
# 115.219986 seconds (603.27 M allocations: 989.625 GiB, 45.33% gc time, 85 lock conflicts, 0.26% compilation time)

# now more allocations? 195.934984 seconds (894.08 M allocations: 1.032 TiB, 35.99% gc time, 55 lock conflicts, 268.37% compilation time: 4% of which was recompilation)

ids2 = [(; p, m) for p in 1:3, m in 1:9] |> vec |> sort
(kfs2, sols2) = fit_modules(data, ti, ids2)
fig3 = plot_ecms(kfs2, sols2; n=12)


# f = extract_ocv(kfs[(; p=3, m=5, c=12)]);
f = let
    df = CSV.File("data/ocv-yuasa-ees-N208.csv") |> DataFrame
    ocv⁻¹ = LinearInterpolation(df.soc, df.ocv)
    ocv = LinearInterpolation(df.ocv, df.soc)
    (; ocv⁻¹, ocv)
end


vlim = (3.8, 3.95)
param_cells = Dict(id =>
    Dict(
        :Q => calc_Q(kfs[id], f.ocv⁻¹; v=vlim),
        :soc => calc_soc0(kfs[id], f.ocv⁻¹; v=vlim),
        :soh => calc_soh(kfs[id], f.ocv⁻¹, 100; v=vlim),
    ) for id in ids
)

param_modules = Dict(id =>
    Dict(
        :Q => calc_Q(kfs2[id], f.ocv⁻¹; v=vlim, n=12),
        :soc => calc_soc0(kfs2[id], f.ocv⁻¹; v=vlim, n=12),
        :soh => calc_soh(kfs2[id], f.ocv⁻¹, 100; v=vlim, n=12),
    ) for id in ids2
)

plot_ecms_norm(kfs, sols, f.ocv⁻¹, f.ocv; vlim)


fig4 = plot_module_soh(param_cells, param_modules)
fig5 = plot_cell_soh_hist(param_cells)
fig6 = plot_module_inhomogenity(param_modules)

id = (; p=1, m=1, c=1)
fig7 = plot_sim(kfs[id], sols[id])


let id = (; p=3, m=2, c=1)
    plot_rc_param_trajectory(kfs[id], sols[id])
end


let id = (; p=3, m=5, c=1)
    plot_q_filter(kfs[id], sols[id])
end
