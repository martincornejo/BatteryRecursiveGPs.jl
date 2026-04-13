# Oscilloscope current validation
#
# Investigates whether the current bias in the CMU current (î) vs the
# oscilloscope current (i) is the primary source of Q estimation bias.
# Strategy: replace î with i in the fitting pipeline and compare Q/s0 estimates.

using BatteryDigitalTwin
using RecursiveGPs: predict_gp
using CSV
using JSON
using DataFrames
using StatsBase
using Measurements
using DataInterpolations
using StaticArrays
using LinearAlgebra
import ComponentArrays: ComponentVector, ComponentMatrix
using CairoMakie
using Printf

include("fit-model.jl")
include("plots.jl")

# === setup
data = CSV.File("data/data-yuasa-synthetic/data-yuasa-synthetic.csv") |> DataFrame
params_real = JSON.parsefile("data/data-yuasa-synthetic/battery-params.json")

f = let
    df = CSV.File("data/ocv/ocv-yuasa.csv") |> DataFrame
    ocv⁻¹ = LinearInterpolation(df.soc, df.ocv)
    ocv = LinearInterpolation(df.ocv, df.soc)
    (; ocv⁻¹, ocv)
end

fR0(s, T; kT = 2000, T0 = 25) = (0.0015 + 0.0004 * sinpi(-0.2 + s * 1.5)) *
    exp(kT * (1 / (T + 273.15) - 1 / (T0 + 273.15)))
fR025(s) = fR0(s, 25)

θ = ComponentVector(;
    ocv = (; σ = 0.5, ℓ = 0.3),
    r0 = (; σ = 0.5, ℓ = 2.0),
    vσ = 2.0e-3, Ts = 1.0,
    r0μ = 1.5e-3,
    rc = (;
        v0 = 0.0, σ0_v = 1.0e-5, σ1_v = 5.0e-5,
        r0 = 1.5e-3, σ0_r = 5.0e-6, σ1_r = 0.0,
        τ0 = 250.0, σ0_τ = 1.0, σ1_τ = 0.0,
    ),
    cc = (; σ0 = 0.0, σ1 = 0.1e-5),
    arr = (; T0 = 25, k0 = 1500, σ0_k = 20, σ1_k = 0.0),
)

# === fit with oscilloscope current (i instead of î)
# Replace CMU current (î) with oscilloscope current (i) to isolate current-sensor bias.
data_osc = copy(data)
data_osc[!, "î"] = data_osc.i

(models_osc, sols_osc) = fit_models(data_osc, 1:12, θ);

posteriors_osc = Dict(id => extract_posterior(models_osc[id], sols_osc[id]) for id in 1:12)
ids_osc = sort(collect(keys(posteriors_osc)))
cells_osc = [(; q = collect(posteriors_osc[id].q), μ = collect(posteriors_osc[id].μ)) for id in ids_osc]
fit_osc = fit_composite_ocv(cells_osc)

v_ref_osc = let
    V_lab_lo = minimum(f.ocv.u)
    V_lab_hi = maximum(f.ocv.u)
    (clamp(fit_osc.v_grid[1], V_lab_lo, V_lab_hi),
     clamp(fit_osc.v_grid[end], V_lab_lo, V_lab_hi))
end
soc_ref_osc = (f.ocv⁻¹(v_ref_osc[1]), f.ocv⁻¹(v_ref_osc[2]))
rescaled_osc = rescale_composite_ocv(fit_osc; v_ref = v_ref_osc, soc_ref = soc_ref_osc)

param_cells_osc = Dict(
    ids_osc[i] => Dict(:Q => rescaled_osc.Q_cell[i], :soc => rescaled_osc.s0[i])
        for i in eachindex(ids_osc)
)

println("\n=== Q/s0 with oscilloscope current ===")
df_params_osc = params_to_df(param_cells_osc, params_real)
show(df_params_osc, allrows = true)

plot_qs_scatter(param_cells_osc, params_real)

df_ocv_osc = ocv_residuals_to_df(models_osc, sols_osc, param_cells_osc, params_real, f.ocv)
show(df_ocv_osc, allrows = true)

fig_ocv_osc = plot_ocv_diagnostics(models_osc, sols_osc, param_cells_osc, params_real, f.ocv)

# === OCV residual vs charge (cell 1)
fig_ocv_res = let id = 1
    (; q, μ) = gp_ocv(models_osc[id], sols_osc[id])
    Q_est = Measurements.value(param_cells_osc[id][:Q])
    s0_est = Measurements.value(param_cells_osc[id][:soc])
    soc_est = s0_est .+ q ./ Q_est
    slims = extrema(f.ocv.t)
    mask = findall(slims[1] .<= soc_est .<= slims[2])
    r = (μ[mask] .- f.ocv.(soc_est[mask])) .* 1000  # mV

    fig = Figure()
    ax = Axis(
        fig[1, 1]; xlabel = "Charge / Ah", ylabel = "OCV residual / mV",
        title = "Cell $(id) — oscilloscope current"
    )
    lines!(ax, q[mask], r)
    hlines!(ax, [0]; linestyle = :dash, color = :black)
    fig
end
