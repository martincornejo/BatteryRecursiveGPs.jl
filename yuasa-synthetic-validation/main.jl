using BatteryDigitalTwin
using CSV
using JSON
using DataFrames
using StatsBase
using Measurements
using DataInterpolations
using StaticArrays
import ComponentArrays: ComponentVector, ComponentMatrix, getaxes

include("fit-model.jl")

# === load data
data = CSV.File("data/data-yuasa-synthetic/data-yuasa-synthetic.csv") |> DataFrame

f = let
    df = CSV.File("data/ocv/ocv-yuasa.csv") |> DataFrame
    ocv⁻¹ = LinearInterpolation(df.soc, df.ocv)
    ocv = LinearInterpolation(df.ocv, df.soc)
    (; ocv⁻¹, ocv)
end

fR0(s, T; kT=20, T0=25) = (0.0015 + 0.0004 * sinpi(-0.2 + s * 1.5)) * exp(kT * (1 / T - 1 / T0))
fR025(s) = fR0(s, 25) # reference R0 at 25°C


# === fit models
kfs, sols = fit_models(data, 1:12);

# === analysis
vlim = (3.7, 4.0)
plot_ecms(kfs, sols) |> display
plot_ecms_norm(kfs, sols, f.ocv⁻¹, f.ocv, fR025; vlim) |> display

let id = 1
    fig = plot_sim(kfs[id], sols[id])
    fig.content[1].title = "Cell $(id)"
    fig |> display
end
for id in 1:12
    fig = plot_q_estimation(kfs[id], sols[id], data)
    fig.content[1].title = "Cell $(id)"
    fig |> display
end
let id = 1
    fig = plot_rc_param_trajectory(kfs[id], sols[id]; r1=1.2e-3, τ1=300)
    fig.content[1].title = "Cell $(id)"
    fig |> display
end


param_cells = Dict(id =>
    Dict(
        :Q => calc_Q(kfs[id], sols[id], f.ocv⁻¹; v=vlim),
        :soc => calc_soc0(kfs[id], sols[id], f.ocv⁻¹; v=vlim),
        :soh => calc_soh(kfs[id], sols[id], f.ocv⁻¹, 100; v=vlim),
    ) for id in 1:12
)
params_real = JSON.parsefile("data/data-yuasa-synthetic/battery-params.json")

let
    fig = Figure()
    ax = Axis(fig[1, 1])

    Q = [params_real["cell_$id"]["Q"] for id in 1:12]
    s = [params_real["cell_$id"]["soc"] for id in 1:12]
    scatter!(ax, Q, s)

    Qμ = [param_cells[id][:Q] for id in 1:12] .|> Measurements.value
    Qσ = [param_cells[id][:Q] for id in 1:12] .|> Measurements.uncertainty
    sμ = [param_cells[id][:soc] for id in 1:12] .|> Measurements.value
    sσ = [param_cells[id][:soc] for id in 1:12] .|> Measurements.uncertainty
    scatter!(ax, Qμ, sμ)
    errorbars!(ax, Qμ, sμ, Qσ, whiskerwidth=10, direction=:x, color=Cycled(2))
    errorbars!(ax, Qμ, sμ, sσ, whiskerwidth=10, direction=:y, color=Cycled(2))

    fig
end

# TODO
# - plot/validate soc estimation (cell, pack)
# - plot/validate arrhenius (cell, pack)
