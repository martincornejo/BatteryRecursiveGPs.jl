using RecursiveGPs
using AbstractGPs
using StaticArrays

using Distributions
using LowLevelParticleFilters
import LowLevelParticleFilters as LLPF

using CSV
using DataFrames
using Random

using StatsBase

using BenchmarkTools

using ForwardDiff
using LinearAlgebra

import ComponentArrays: ComponentVector, ComponentMatrix, getaxes

using CairoMakie

using PreallocationTools

using StaticArrays

using DataInterpolations

include("synthetic-data.jl")
include("kf-params.jl")
include("kf-state.jl")
include("plot.jl")


## === dataset
# begin
#     df = CSV.read("data/profile.csv", DataFrame)
#     zt = fit_zscore(df)
#     dfn = normalize_data(df, zt)

#     horizon = 24 * 3600
#     df_train = dfn[1:horizon, :]
#     df_ = df[1:horizon, :]

#     # ys = [SA[y+0.01randn()+0.1] for y in df_train.v]
#     ys = [SA[y] for y in df_train.v]
#     us = [(; i=x.i) for x in eachrow(df_train)]
# end

begin # read OCV look-up-table and current profile
    df_ocv = CSV.File("data/ocv.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc, extrapolation=ExtrapolationType.Constant)
    f´ocv = LinearInterpolation(df_ocv.soc, df_ocv.ocv)

    soc_prior = 0.0:0.05:1.0
    # ocv_prior = StatsBase.transform(zt.v, focv(soc_prior) .+ 0.02)
    ocv_prior = StatsBase.transform(zt.v, focv(soc_prior))
    focv_prior = LinearInterpolation(ocv_prior, soc_prior; extrapolation=ExtrapolationType.Constant)

end

# === model
#
begin
    kf1 = build_kf_params(focv_prior)

    soc0 = f´ocv(df[begin, :v])
    kf2 = build_kf_state(kf1, soc0)
end

plot_ecm(kf1, zt, focv, fR0; closeup=false)

for (u, v) in zip(us, ys)
    i = u.i
    LLPF.update!(kf2, (; i), v)
    s = kf2.x[1]
    LLPF.update!(kf1, (; i, s), v)
end

# smoothsol = smooth(kf2, us, ys)
# plot_soc_estimation(smoothsol, df)

begin
    # ys = [SA[y+0.01randn()+0.1] for y in df_train.v]
    s´ = smoothsol.xT .|> first
    us2 = [(; i=x.i, s) for (x, s) in zip(eachrow(dfn), s´)]
end


for (i, (u, y)) in enumerate(zip(us2, ys))
    # @info i kf.x[1]
    kf1(u, y)
end


plot_ecm(kf1, zt, focv, fR0)





x0 = SA[first(s´)]
θ = (; kf, Ts=1.0, q=4.8)

d0 = MvNormal(x0, 0.1I)
R1 = @SMatrix [0.0001;;]
kf3 = ExtendedKalmanFilter(dynamics_state, measurement_state, R1, R2_state, d0; nx=1, nu=1, ny=1, p=θ)

smoothsol = smooth(kf3, us, ys)


let fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]
    s´ = smoothsol.xT .|> first

    horizon = 24 * 3600
    df_ = df[1:horizon, :]

    s = 0.4 .+ df_.s ./ 4.8
    lines!(ax[1], df_.t, s´)
    lines!(ax[1], df_.t, s)
    lines!(ax[2], df_.t, s - s´)
    fig
end


let fig = Figure()
    ax = Axis(fig[1, 1])
    s´ = smoothsol.xT .|> first

    horizon = 24 * 3600
    df_ = df[1:horizon, :]

    scatter!(ax, s´, df_.s)
    fig
end
