using BatteryDigitalTwin
using CSV
using JSON
using DataFrames
using StatsBase
using Measurements
using DataInterpolations
using StaticArrays
import ComponentArrays: ComponentVector, ComponentMatrix
using CairoMakie
using Printf

include("fit-model.jl")

# === load data
data = CSV.File("data/data-yuasa-synthetic/data-yuasa-synthetic.csv") |> DataFrame

f = let
    df = CSV.File("data/ocv/ocv-yuasa.csv") |> DataFrame
    ocv⁻¹ = LinearInterpolation(df.soc, df.ocv)
    ocv = LinearInterpolation(df.ocv, df.soc)
    (; ocv⁻¹, ocv)
end

fR0(s, T; kT=2000, T0=25) = (0.0015 + 0.0004 * sinpi(-0.2 + s * 1.5)) * exp(kT * (1 / (T + 273.15) - 1 / (T0 + 273.15)))
fR025(s) = fR0(s, 25) # reference R0 at 25°C


# === fit models
θ = ComponentVector(;
    ocv=(; σ=0.5, ℓ=0.3),
    r0=(; σ=0.5, ℓ=2.0),
    vσ=2e-3, Ts=1.0,
    r0μ=1.5e-3,
    rc=(;
        v0=0.0, σ0_v=1.0e-5, σ1_v=5.0e-5,
        r0=1.5e-3, σ0_r=5.0e-6, σ1_r=0.0,
        τ0=250.0, σ0_τ=1.0, σ1_τ=0.0,
    ),
    cc=(; σ0=0.0, σ1=0.1e-5),
    arr=(; T0=25, k0=1500, σ0_k=20, σ1_k=0.0),
)
(; models, sols) = fit_models(data, 1:12, θ);
(; models, sols) = refine_models(data, 1:12, models, sols);

# === analysis
vlim = (3.7, 4.0)
fig1 = plot_ecms(models, sols) #|> display
fig2 = plot_ecms_norm(models, sols, f.ocv⁻¹, f.ocv, fR025; vlim) #|> display

let id = 1
    fig = plot_sim(models[id], sols[id])
    fig.content[1].title = "Cell $(id)"
    fig |> display
end
let id = 1
    fig = plot_q_estimation(data, sols[id], models[id])
    fig.content[1].title = "Cell $(id)"
    fig |> display
end
let id = 1
    fig = plot_rc_param_trajectory(models[id], sols[id]; r1=1.2e-3, τ1=300)
    fig.content[1].title = "Cell $(id)"
    fig |> display
end
let id = 1
    fig = plot_arrhenius_param_trajectory(models[id], sols[id]; k=2000)
    fig.content[1].title = "Cell $(id)"
    fig |> display
end


param_cells = Dict(id =>
    let (; Q, s0) = calc_wls(models[id], sols[id], f.ocv⁻¹, f.ocv)
        Dict(:Q => Q, :soc => s0)
    end
    for id in 1:12
)
params_real = JSON.parsefile("data/data-yuasa-synthetic/battery-params.json")

report_params(param_cells, params_real)
report_ocv_residuals(models, sols, param_cells, params_real, f.ocv)

fig3 = let
    fig = Figure()
    ax = Axis(fig[1, 1])

    Q = [params_real["cell_$id"]["Q"] for id in 1:12]
    s = [params_real["cell_$id"]["soc"] for id in 1:12] * 100
    scatter!(ax, Q, s; label="Real")

    Qμ = [param_cells[id][:Q] for id in 1:12] .|> Measurements.value
    Qσ = [param_cells[id][:Q] for id in 1:12] .|> Measurements.uncertainty
    sμ = 100 * [param_cells[id][:soc] for id in 1:12] .|> Measurements.value
    sσ = 100 * [param_cells[id][:soc] for id in 1:12] .|> Measurements.uncertainty
    scatter!(ax, Qμ, sμ; label="Estimation")
    errorbars!(ax, Qμ, sμ, Qσ, whiskerwidth=10, direction=:x, color=Cycled(2))
    errorbars!(ax, Qμ, sμ, sσ, whiskerwidth=10, direction=:y, color=Cycled(2))

    ax.xlabel = "Q_cell / Ah"
    ax.ylabel = "Initial SOC / %"

    Legend(fig[2, 1], ax, orientation=:horizontal)

    fig
end

fig4 = plot_ocv_diagnostics(models, sols, param_cells, params_real, f.ocv) #|> display

# TODO
# - plot/validate soc estimation (cell, pack)

# state estimation with EKF
let id = 1
    fig = plot_q_estimation(data, sols[id], models[id])
    fig.content[1].title = "Cell $(id)"
    fig |> display
end
begin
    θ_state = (; q=(; σ0=1e-3, σ1=0.5e-5), rc=(; σ0=1e-4, σ1=1e-4))
    soc_model = YuasaStateModel(models[1]; q0=0.0, Ts=1.0, θ=θ_state)
    u, y = cell_dataset(data, 1)

    sol_state = run_kf!(soc_model, u, y)

    plot_q_estimation_state(data, sol_state) |> display
    report_q_estimation(data, sol_state, 1, param_cells)
end

