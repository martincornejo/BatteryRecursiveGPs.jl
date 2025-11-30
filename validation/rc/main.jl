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
include("model_rc.jl")
include("rc.jl")
## === dataset
begin
    # read OCV look-up-table and current profile, define internal resistance function
    df_ocv = CSV.File("data/ocv.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc, extrapolation=ExtrapolationType.Constant)

    df = CSV.File("data/profile.csv") |> DataFrame
    # fi = ConstantInterpolation(df.i * 1.5, df.t)
    fi = ConstantInterpolation(df.i * -1.5, df.t)

    fR0(s) = 0.01 + 0.005 * s + 0.005 * sinpi(-0.2 + s * 1.5)

    # simulate battery operation
    ttest = 2.5 * 24 * 3600
    ttrain = 2 * 24 * 3600
    tspan = (0, ttest) # one day
    Ts = 1.0 # # time sampling

    df = generate_timeseries(; Ts, tspan, fi, focv, fR0, soc0=0.65)
    plot_timeseries(df) |> display

    # create input / output data 
    zt = fit_zscore(df)
    dfn = normalize_data(df, zt)

    df_train = subset(df, :t => ByRow(<=(ttrain)))
    df_train_n = subset(dfn, :t => ByRow(<=(ttrain)))
    df_test = subset(df, :t => ByRow(>(ttrain)))
    df_test_n = subset(dfn, :t => ByRow(>(ttrain)))

    us = [(; i, î) for (i, î) in zip(df_train.i, df_train_n.i)]
    ut = [(; i, î) for (i, î) in zip(df_test.i, df_test_n.i)]
    ys = [SA[y] for y in df_train_n.v]
end

# === model
focv´ = let # focv prior
    df_ocv = CSV.File("data/ocv.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc)
    soc´ = 0.0:0.05:1.0
    ocv´ = StatsBase.transform(zt.v, focv(soc´))
    focv´ = LinearInterpolation(ocv´, soc´; extrapolation=ExtrapolationType.Constant)
end

focv⁻¹ = let
    df_ocv = CSV.File("data/ocv.csv") |> DataFrame
    LinearInterpolation(df_ocv.soc, df_ocv.ocv)
end

focv2´ = let
    df_ocv = CSV.File("data/ocv-2.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc)
    soc´ = 0.0:0.05:1.0
    ocv´ = StatsBase.transform(zt.v, focv(soc´))
    focv´ = LinearInterpolation(ocv´, soc´; extrapolation=ExtrapolationType.Constant)
end

focv2⁻¹ = let
    df_ocv = CSV.File("data/ocv-2.csv") |> DataFrame
    LinearInterpolation(df_ocv.soc, df_ocv.ocv)
end

focv3´ = let
    df_ocv = CSV.File("data/ocv-3.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc)
    soc´ = 0.0:0.05:1.0
    ocv´ = StatsBase.transform(zt.v, focv(soc´))
    focv´ = LinearInterpolation(ocv´, soc´; extrapolation=ExtrapolationType.Constant)
end

focv3⁻¹ = let
    df_ocv = CSV.File("data/ocv-3.csv") |> DataFrame
    LinearInterpolation(df_ocv.soc, df_ocv.ocv)
end

# ==== 
# perfect input
θ = (; # hyperparams
    ocv=(; σ=0.1, ℓ=0.1),
    r0=(; σ=0.001, ℓ=0.5),
    soc=(; σ1=0.01, σ2=1e-8, soc0=0.65),
    rc =(;ts = 1.0, τ0 = 40, R0 = 10e-3)
)
ϑ = (; Ts=10.0, q=4.8)
kf = build_kf(θ, ϑ, focv´, zt; n=21)

plot_ecm(kf, df, zt, focv, fR0; closeup=true)

sol = run_sim!(kf, us, ys, ut)

plot_ecm(kf, df, zt, focv, fR0; closeup=true)

calc_ocv_mae(kf, df_train, zt, focv)

plot_sim(sol, df, ttrain)

# ===
# wrong initial OCV
soc0 = df_train[begin, :s]
soc0´ = 0.7 # focv2⁻¹(df_train[begin, :v])
θ = (; # hyperparams
    ocv=(; σ=0.1, ℓ=0.1),
    r0=(; σ=0.001, ℓ=0.5),
    soc=(; σ1=0.05, σ2=1e-7, soc0=soc0´),
)
ϑ = (; Ts=10.0, q=4.8)
kf2 = build_kf(θ, ϑ, focv2´, zt; n=21)

plot_ecm(kf2, df, zt, focv, fR0; focv_prior=focv2´)

sol2 = run_sim!(kf2, us, ys, ut)

Δs = soc0 - focv2⁻¹(focv(soc0))
plot_ecm(kf2, df, zt, focv, fR0; focv_prior=focv2´, soc_shift=0.0, closeup=true)
plot_ecm(kf2, df, zt, focv, fR0; focv_prior=focv2´, soc_shift=Δs, closeup=true)

calc_ocv_mae(kf2, df_train, zt, focv; soc_shift=0.0)
calc_ocv_mae(kf2, df_train, zt, focv; soc_shift=Δs)

plot_sim(sol2, df, ttrain; soc_shift=0.0)
plot_sim(sol2, df, ttrain; soc_shift=Δs)


# ===
# current noise
let
    Random.seed!(12345)
    ϵ = rand(Normal(0.0, 2.5), nrow(df)) # very high white noise
    df[!, :iϵ] = df.i + ϵ

    plot_coulomb_error_noise(df; q=4.8) |> display

    zt = fit_zscore(df)
    dfn = normalize_data(df, zt)
    dfn[!, :iϵ] = StatsBase.transform(zt.i, df.iϵ)

    df_train = subset(df, :t => ByRow(<=(ttrain)))
    df_train_n = subset(dfn, :t => ByRow(<=(ttrain)))
    df_test = subset(df, :t => ByRow(>(ttrain)))
    df_test_n = subset(dfn, :t => ByRow(>(ttrain)))

    # noise is only added to coulomb counting!
    us = [(; i, î) for (i, î) in zip(df_train.iϵ, df_train_n.i)]
    ut = [(; i, î) for (i, î) in zip(df_test.iϵ, df_test_n.i)]
    ys = [SA[y] for y in df_train_n.v]
end


soc0 = df_train[begin, :s]
soc0´ = 0.7 # focv2⁻¹(df_train[begin, :v])
θ = (; # hyperparams
    ocv=(; σ=0.1, ℓ=0.1),
    r0=(; σ=0.001, ℓ=0.5),
    soc=(; σ1=0.01, σ2=1e-7, soc0=soc0´),
)
ϑ = (; Ts=10.0, q=4.8)
kf2 = build_kf(θ, ϑ, focv2´, zt; n=21)

plot_ecm(kf2, df, zt, focv, fR0; focv_prior=focv2´)

@time sol2 = run_sim!(kf2, us, ys, ut)

Δs = soc0 - focv2⁻¹(focv(soc0))
plot_ecm(kf2, df, zt, focv, fR0; focv_prior=focv2´, soc_shift=0.0, closeup=true)
plot_ecm(kf2, df, zt, focv, fR0; focv_prior=focv2´, soc_shift=Δs, closeup=true)

calc_ocv_mae(kf2, df_train, zt, focv; soc_shift=0.0)
calc_ocv_mae(kf2, df_train, zt, focv; soc_shift=Δs)

plot_sim(sol2, df, ttrain; soc_shift=0.0)
plot_sim(sol2, df, ttrain; soc_shift=Δs)


## ==
begin
    # read OCV look-up-table and current profile, define internal resistance function
    df_ocv = CSV.File("data/ocv.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc, extrapolation=ExtrapolationType.Constant)

    df = CSV.File("data/profile.csv") |> DataFrame
    df = df[1:10:end, :]
    # fi = ConstantInterpolation(df.i * 1.5, df.t)
    fi = ConstantInterpolation(df.i * -1.5, df.t)

    fR0(s) = 0.01 + 0.005 * s + 0.005 * sinpi(-0.2 + s * 1.5)

    # simulate battery operation
    ttest = 2.5 * 24 * 3600
    ttrain = 2 * 24 * 3600
    tspan = (0, ttest) # one day
    Ts = 10.0 # # time sampling

    df = generate_timeseries(; Ts, tspan, fi, focv, fR0, soc0=0.65)
    plot_timeseries(df) |> display

    # create input / output data 
    zt = fit_zscore(df)
    dfn = normalize_data(df, zt)

    df_train = subset(df, :t => ByRow(<=(ttrain)))
    df_train_n = subset(dfn, :t => ByRow(<=(ttrain)))
    df_test = subset(df, :t => ByRow(>(ttrain)))
    df_test_n = subset(dfn, :t => ByRow(>(ttrain)))

    us = [(; i, î) for (i, î) in zip(df_train.i, df_train_n.i)]
    ut = [(; i, î) for (i, î) in zip(df_test.i, df_test_n.i)]
    ys = [SA[y] for y in df_train_n.v]
end


soc0 = df_train[begin, :s]
# soc0´ = focv2⁻¹(df_train[begin, :v])
θ = (; # hyperparams
    ocv=(; σ=0.1, ℓ=0.1),
    r0=(; σ=0.001, ℓ=0.5),
    soc=(; σ1=0.01, σ2=1e-7, soc0=soc0),
)
ϑ = (; Ts=10.0, q=4.8)
kf = build_kf(θ, ϑ, focv2´, zt; n=21)

plot_ecm(kf2, df, zt, focv, fR0; focv_prior=focv2´)

@time sol2 = run_sim!(kf2, us, ys, ut)

Δs = soc0 - focv2⁻¹(focv(soc0))
plot_ecm(kf2, df, zt, focv, fR0; focv_prior=focv2´, soc_shift=0.0, closeup=true)
plot_ecm(kf2, df, zt, focv, fR0; focv_prior=focv2´, soc_shift=Δs, closeup=true)

calc_ocv_mae(kf2, df_train, zt, focv; soc_shift=0.0)
calc_ocv_mae(kf2, df_train, zt, focv; soc_shift=Δs)

plot_sim(sol2, df, ttrain; soc_shift=0.0)
plot_sim(sol2, df, ttrain; soc_shift=Δs)


function make_io_data(df, dfn, ttrain)
    df_train = subset(df, :t => ByRow(<=(ttrain)))
    df_train_n = subset(dfn, :t => ByRow(<=(ttrain)))
    df_test = subset(df, :t => ByRow(>(ttrain)))
    df_test_n = subset(dfn, :t => ByRow(>(ttrain)))

    # noise is only added to coulomb counting!
    us = [(; i, î) for (i, î) in zip(df_train.i, df_train_n.i)]
    ut = [(; i, î) for (i, î) in zip(df_test.i, df_test_n.i)]
    ys = [SA[y] for y in df_train_n.v]

    (; ys, us, ut)
end

function animate_filter(kf, df, dfn)
    Δs = soc0 - focv2⁻¹(focv(soc0))
    for ttrain in 0:3600:df[end, :t]
        kf_ = deepcopy(kf)
        (; ys, us, ut) = make_io_data(df, dfn, ttrain)
        sol = run_sim!(kf_, us, ys, ut)
        plot_sim(sol, df, ttrain; soc_shift=Δs) |> display
        # plot_ecm(kf_, df, zt, focv, fR0; focv_prior=focv2´, soc_shift=Δs, closeup=true) |> display
    end
end
