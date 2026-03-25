using CSV
using Dates
using DataFrames
using Printf
using Intervals
using StatsBase
using DataInterpolations
using StaticArrays
using Measurements

using BatteryDigitalTwin

import ComponentArrays: ComponentVector, ComponentMatrix, getaxes

using CairoMakie
# using GLMakie

include("fit-model.jl")

# === load data ===
dateformat = dateformat"y-m-dTHH:MM:SS.sss+00:00"
data = CSV.File("data/data-yuasa-phase-test/combined_log_20260209_062859.csv"; dateformat) |> DataFrame

f = let
    df = CSV.File("data/ocv/ocv-yuasa-ees-N208.csv") |> DataFrame
    # df = CSV.File("data/ocv/ocv-yuasa-ees.csv") |> DataFrame
    # df = CSV.File("data/ocv/ocv-yuasa.csv") |> DataFrame
    ocv⁻¹ = LinearInterpolation(df.soc, sort(df.ocv))
    ocv = LinearInterpolation(sort(df.ocv), df.soc)
    (; ocv⁻¹, ocv)
end

# names(df)

# let m = 2
for m in 1:9
    df = transform(data, All() .=> x -> coalesce.(x, NaN), renamecols=false)

    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:3]
    for i in 1:12
        cell = @sprintf("%02d", i)
        lines!(ax[1], df.timestamp_utc, df[:, "m$(m)_cell$(cell)"])
    end

    lines!(ax[2], df.timestamp_utc, df[:, "m$(m)_current"])
    lines!(ax[3], df.timestamp_utc, df[:, "m$(m)_temp01"])
    lines!(ax[3], df.timestamp_utc, df[:, "m$(m)_temp02"])

    fig |> display
end


ti = Interval(DateTime("2026-02-09T06:29:00"), DateTime("2026-02-09T20:13:00"))
# ids = [(; m, c) for m in 1:9, c in 1:12] |> vec |> sort
ids = [(; m, c) for m in 1:1, c in 1:12] |> vec |> sort
kfs, sols = fit_models_spawn(data, ti, ids)

for id in ids
    plot_sim(kfs[id], sols[id]) |> display
end

plot_ecms(kfs, sols)

vlim = (3.5, 4.0)
plot_ecms_norm(kfs, sols, f.ocv⁻¹, f.ocv; vlim)

# kf, sol = fit_model(data, ti, (;m=1, c=1))
param_cells = Dict(id =>
    Dict(
        :Q => calc_Q(kfs[id], sols[id], f.ocv⁻¹; v=vlim),
        :soc => calc_soc0(kfs[id], sols[id], f.ocv⁻¹; v=vlim),
        :soh => calc_soh(kfs[id], sols[id], f.ocv⁻¹, 100; v=vlim),
    ) for id in ids
)

Qs = [param_cells[id][:Q] for id in ids]
socs = [param_cells[id][:soc] for id in ids]

let
    Qμ = Measurements.value.(Qs)
    Qσ = Measurements.uncertainty.(Qs)
    x = 1:length(Qs)

    fig = Figure()
    ax = Axis(fig[1, 1])
    barplot!(ax, x, Qμ)
    errorbars!(ax, x, Qμ, Qσ; whiskerwidth=10, color=:black)
    # ylims!(ax, 70, 80)
    # ax.yticks = 70:80
    ax.xticks = x
    fig
end

fig1 = let
    # fig = Figure(size=(500, 450))
    fig = Figure()
    # ax = [Axis(fig[i, 1]) for i in 1:2]
    ax = [Axis(fig[i, 1]) for i in 1:3]

    df = transform(data, All() .=> x -> coalesce.(x, NaN), renamecols=false)

    # lines!(ax[1], data.timestamp_utc, -data.tek_m_cur_ref)
    for i in 1:9
        lines!(ax[1], df.timestamp_utc, df.mb_SystemSignedActivePowerSetPointAC * 1e-3)
        lines!(ax[2], df.timestamp_utc, df[!, "m$(i)_current"], alpha=0.5)
        # lines!(ax[1], df.timestamp_utc, -df[!, "m$(i)_current_rms"], alpha=0.5)
    end

    lines!(ax[3], df.timestamp_utc, df.mb_CurrentLimitDischargeSystem)
    lines!(ax[3], df.timestamp_utc, -df.mb_CurrentLimitChargeSystem)

    ax[1].title = "Measurement"
    ax[1].ylabel = "Power Target / kW"
    ax[2].ylabel = "Battery Current / A"
    ax[3].ylabel = "Max Current / A"
    # ax[2].xlabel = "Time / h"

    ylims!(ax[1], -40, 40)

    linkxaxes!(ax...)
    fig
end

