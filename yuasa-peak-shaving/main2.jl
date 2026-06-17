using CSV
using DataFrames
using Dates
using Printf
using Intervals
using StatsBase
using DataInterpolations
using StaticArrays
using Measurements
import ComponentArrays: ComponentVector, ComponentMatrix
using JSON

using RecursiveGPs: predict_gp

using CairoMakie

# Worker setup for distributed fitting (run only once)
using Distributed
addprocs(Sys.CPU_THREADS ÷ 2)
@everywhere using BatteryDigitalTwin

include("dataset.jl")
include("parallel.jl")
include("fit-model.jl")
include("analysis.jl")


# === load data ===
dateformat = dateformat"y-m-dTHH:MM:SS.sss+00:00"
data_1 = CSV.File("data/data-yuasa-peak-shaving/combined_log_20260327_082612.csv"; dateformat) |> DataFrame
data_2 = CSV.File("data/data-yuasa-peak-shaving/20260504_baseline.csv"; dateformat) |> DataFrame


# === fit 2RC models on data_1 and data_2 ===
ti_1 = Interval(DateTime("2026-03-27T08:41:13"), DateTime("2026-03-27T19:12:38"))
ti_2 = Interval(DateTime("2026-05-04T07:05:15"), DateTime("2026-05-04T19:36:56"))
ids = [(; m, c) for m in 1:9, c in 1:12] |> vec |> sort

@time (; models, sols) = fit_cells(data_1, ti_1, ids; model_type = Fenecon2RCModel, θ = default_θ_2rc());
@time res_d2 = fit_cells(data_2, ti_2, ids; model_type = Fenecon2RCModel, θ = default_θ_2rc());
models_d2 = res_d2.models
sols_d2 = res_d2.sols

# Refine data_1 fit using data_2
@time (; models_refined, sols_refined) = refine_cells_distributed(data_2, ti_2, models, ids)


# === Composite OCV per dataset, rescaled to common reference ===

cells_d1 = [gp_ocv(models[id], sols[id]) for id in ids]
cells_d2 = [gp_ocv(models_d2[id], sols_d2[id]) for id in ids]
cells_refined = [gp_ocv(models_refined[id], sols_refined[id]) for id in ids]

fit_d1 = fit_composite_ocv(cells_d1; n_v_grid = 100)
fit_d2 = fit_composite_ocv(cells_d2; n_v_grid = 100)
fit_refined = fit_composite_ocv(cells_refined; n_v_grid = 100)

# Common absolute reference: anchor each composite at the same (V, SOC) points
refs_common = (v_low = 3.5, soc_low = 0.1, v_high = 4.05, soc_high = 0.85)
abs_d1 = rescale_composite_ocv(fit_d1, refs_common)
abs_d2 = rescale_composite_ocv(fit_d2, refs_common)
abs_refined = rescale_composite_ocv(fit_refined, refs_common)


# === Export from independent data_1 and data_2 fits ===

params_d1 = extract_battery_params(models, sols, cells_d1, abs_d1, ids)
params_d2 = extract_battery_params(models_d2, sols_d2, cells_d2, abs_d2, ids)
params_refined = extract_battery_params(models_refined, sols_refined, cells_refined, abs_refined, ids)


let # Q distribution comparison
    fig = Figure()
    ax = Axis(fig[1, 1])
    bins = 75:0.5:90

    for params in (params_d1, params_d2, params_refined)
        Qs = reduce(vcat, [[cell["Q"] for cell in values(battery)] for battery in values(params)])
        hist!(ax, Qs; bins, alpha = 0.5)
    end

    fig
end

let m = 1, c = 1 # ocv
    fig = Figure()
    ax = Axis(fig[1, 1])

    cell_id = @sprintf("cell_%d", c)
    for params in (params_d1, params_d2, params_refined)
        ocv = params["battery_$m"][cell_id]["ocv_v"]
        soc = params["battery_$m"][cell_id]["ocv_soc"]
        lines!(ax, soc, ocv)
    end
    fig
end


# === State estimation on data_1 and data_2 with refined models ===

θ_state = (;
    q = (; σ0 = 1.0, σ1 = 1.0e-4),
    rc = (; σ0 = 1.0e-1, σ1 = 1.0e-3),
)

states_1 = estimate_states_distributed(data_1, ti_1, models_refined, ids; θ = θ_state)
states_2 = estimate_states_distributed(data_2, ti_2, models_refined, ids; θ = θ_state)

smoothed_1 = smooth_states_distributed(data_1, ti_1, models_refined, ids; θ = θ_state)
smoothed_2 = smooth_states_distributed(data_2, ti_2, models_refined, ids; θ = θ_state)


# === Export initial states from smoother (per run) ===

initial_1 = extract_initial_states(smoothed_1, abs_refined, ids)
initial_2 = extract_initial_states(smoothed_2, abs_refined, ids)


# === Filtered vs smoothed SOC, individual and combined RC voltages (single cell) ===

for i in eachindex(ids)
    id = ids[i]
    Q_i = Measurements.value(abs_refined.Q_cell[i])
    s0_i = Measurements.value(abs_refined.s0[i])

    fig = Figure(size = (1100, 1100))
    ax_soc = [Axis(fig[1, j]; ylabel = "SOC", title = j == 1 ? "Run 1" : "Run 2") for j in 1:2]
    ax_rc1 = [Axis(fig[2, j]; ylabel = "v_rc1 / V") for j in 1:2]
    ax_rc2 = [Axis(fig[3, j]; ylabel = "v_rc2 / V") for j in 1:2]
    ax_rc = [Axis(fig[4, j]; ylabel = "v_rc1 + v_rc2 / V", xlabel = "Time / h") for j in 1:2]
    linkxaxes!(ax_soc..., ax_rc1..., ax_rc2..., ax_rc...)

    colors = Makie.wong_colors()

    for (j, (states, smoothed)) in enumerate(((states_1, smoothed_1), (states_2, smoothed_2)))
        haskey(states, id) && haskey(smoothed, id) || continue

        st = states[id]
        zt = st.model.kf.p.kf.p.zt
        q_f = StatsBase.reconstruct(zt.q, first.(st.sol.xt))
        q_f_σ = StatsBase.reconstruct(zt.q, sqrt.(getindex.(st.sol.Rt, 1, 1)))
        v1_f = StatsBase.reconstruct(zt.σ, getindex.(st.sol.xt, 2))
        v1_f_σ = StatsBase.reconstruct(zt.σ, sqrt.(getindex.(st.sol.Rt, 2, 2)))
        v2_f = StatsBase.reconstruct(zt.σ, getindex.(st.sol.xt, 3))
        v2_f_σ = StatsBase.reconstruct(zt.σ, sqrt.(getindex.(st.sol.Rt, 3, 3)))
        v_f = v1_f .+ v2_f
        v_f_σ = StatsBase.reconstruct(zt.σ, sqrt.(getindex.(st.sol.Rt, 2, 2) .+ getindex.(st.sol.Rt, 3, 3) .+ 2 .* getindex.(st.sol.Rt, 2, 3)))
        soc_f = q_f ./ Q_i .+ s0_i
        soc_f_σ = q_f_σ ./ Q_i
        t_f = (st.sol.idx .- 1) ./ 3600

        sm = smoothed[id]
        zt_s = sm.model.kf.p.kf.p.zt
        q_s = StatsBase.reconstruct(zt_s.q, first.(sm.smoothed.xT))
        q_s_σ = StatsBase.reconstruct(zt_s.q, sqrt.(getindex.(sm.smoothed.RT, 1, 1)))
        v1_s = StatsBase.reconstruct(zt_s.σ, getindex.(sm.smoothed.xT, 2))
        v1_s_σ = StatsBase.reconstruct(zt_s.σ, sqrt.(getindex.(sm.smoothed.RT, 2, 2)))
        v2_s = StatsBase.reconstruct(zt_s.σ, getindex.(sm.smoothed.xT, 3))
        v2_s_σ = StatsBase.reconstruct(zt_s.σ, sqrt.(getindex.(sm.smoothed.RT, 3, 3)))
        v_s = v1_s .+ v2_s
        v_s_σ = StatsBase.reconstruct(zt_s.σ, sqrt.(getindex.(sm.smoothed.RT, 2, 2) .+ getindex.(sm.smoothed.RT, 3, 3) .+ 2 .* getindex.(sm.smoothed.RT, 2, 3)))
        soc_s = q_s ./ Q_i .+ s0_i
        soc_s_σ = q_s_σ ./ Q_i
        t_s = (0:(length(q_s) - 1)) ./ 3600

        lines!(ax_soc[j], t_f, soc_f; color = colors[1], alpha = 0.6, label = "Filtered")
        band!(ax_soc[j], t_f, soc_f .- 2soc_f_σ, soc_f .+ 2soc_f_σ; color = (colors[1], 0.2))
        lines!(ax_soc[j], t_s, soc_s; color = colors[2], label = "Smoothed")
        band!(ax_soc[j], t_s, soc_s .- 2soc_s_σ, soc_s .+ 2soc_s_σ; color = (colors[2], 0.2))

        lines!(ax_rc1[j], t_f, v1_f; color = colors[1], alpha = 0.6)
        band!(ax_rc1[j], t_f, v1_f .- 2v1_f_σ, v1_f .+ 2v1_f_σ; color = (colors[1], 0.2))
        lines!(ax_rc1[j], t_s, v1_s; color = colors[2])
        band!(ax_rc1[j], t_s, v1_s .- 2v1_s_σ, v1_s .+ 2v1_s_σ; color = (colors[2], 0.2))

        lines!(ax_rc2[j], t_f, v2_f; color = colors[1], alpha = 0.6)
        band!(ax_rc2[j], t_f, v2_f .- 2v2_f_σ, v2_f .+ 2v2_f_σ; color = (colors[1], 0.2))
        lines!(ax_rc2[j], t_s, v2_s; color = colors[2])
        band!(ax_rc2[j], t_s, v2_s .- 2v2_s_σ, v2_s .+ 2v2_s_σ; color = (colors[2], 0.2))

        lines!(ax_rc[j], t_f, v_f; color = colors[1], alpha = 0.6)
        band!(ax_rc[j], t_f, v_f .- 2v_f_σ, v_f .+ 2v_f_σ; color = (colors[1], 0.2))
        lines!(ax_rc[j], t_s, v_s; color = colors[2])
        band!(ax_rc[j], t_s, v_s .- 2v_s_σ, v_s .+ 2v_s_σ; color = (colors[2], 0.2))
        j == 1 && axislegend(ax_soc[j], position = :rb)
    end
    fig |> display
end

# === export results ===
let dir = joinpath(homedir(), "code/DigitalTwinBatteryMMC/data")
    JSON.json(joinpath(dir, "battery-params-yuasa-2rc-data1.json"), params_d1; pretty = 2)
    JSON.json(joinpath(dir, "battery-params-yuasa-2rc-data2.json"), params_d2; pretty = 2)
    JSON.json(joinpath(dir, "battery-params-yuasa-2rc-refined.json"), params_refined; pretty = 2)
end

let dir = joinpath(homedir(), "code/DigitalTwinBatteryMMC/data")
    JSON.json(joinpath(dir, "initial-states-yuasa-2rc-data1.json"), initial_1; pretty = 2)
    JSON.json(joinpath(dir, "initial-states-yuasa-2rc-data2.json"), initial_2; pretty = 2)
end
