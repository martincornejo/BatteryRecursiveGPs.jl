using RecursiveGPs
using AbstractGPs
using StaticArrays

using Distributions
using LowLevelParticleFilters
import LowLevelParticleFilters as LLPF

using DataFrames
using CSV
using DataFrames
using Random
using DataInterpolations

using StatsBase

using BenchmarkTools

using ForwardDiff
using LinearAlgebra

import ComponentArrays: ComponentVector, ComponentMatrix, getaxes

using CairoMakie

using PreallocationTools

using OrdinaryDiffEq
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

using CairoMakie

include("synthetic-data.jl")
include("model.jl")

## === dataset
begin
    # read OCV look-up-table and current profile, define internal resistance function
    df_ocv = CSV.File("data/ocv.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc, extrapolation=ExtrapolationType.Constant)

    df = CSV.File("data/profile.csv") |> DataFrame
    fi = ConstantInterpolation(df.i, df.t)

    fR0(s) = 0.01 + 0.005 * s + 0.005 * sinpi(-0.2 + s * 1.5)

    # simulate battery operation
    tspan = (0, 24 * 3600) # one day
    Ts = 1.0 # # time sampling

    df = generate_timeseries(; Ts, tspan, fi, focv, fR0)
    plot_timeseries(df) |> display

    # create input / output data 
    zt = fit_zscore(df)
    dfn = normalize_data(df, zt)

    us = [(; i, î) for (i, î) in zip(df.i, dfn.i)]
    ys = [SA[y] for y in dfn.v]
end

# === model
focv´ = let # focv prior
    df_ocv = CSV.File("data/ocv.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc)
    soc´ = 0.0:0.05:1.0
    ocv´ = StatsBase.transform(zt.v, focv(soc´))
    focv´ = LinearInterpolation(ocv´, soc´; extrapolation=ExtrapolationType.Constant)
end

θ = (; # hyperparams
    ocv=(; σ=0.1, ℓ=0.05),
    r0=(; σ=0.001, ℓ=0.5),
    soc=(; σ2=1e-7),
)

kf = build_kf(θ, focv´, zt)

# for (u, y) in zip(us, ys)
#     LLPF.update!(kf, u, y)
# end

sol = forward_trajectory(kf, us, ys)

plot_soc_estimation(sol, df)

plot_ecm(kf, df, zt, focv, fR0)
