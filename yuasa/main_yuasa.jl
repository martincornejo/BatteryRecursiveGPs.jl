using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks
using OrdinaryDiffEq

using StatsBase
using CSV, DataFrames, DataInterpolations

using Distributions
using RecursiveGPs
using AbstractGPs
using StaticArrays
using LowLevelParticleFilters
import LowLevelParticleFilters as LLPF
using ForwardDiff
using LinearAlgebra
import ComponentArrays: ComponentVector, ComponentMatrix, getaxes

using CairoMakie

include("dataset_yuasa.jl")
include("model_yuasa.jl")
include("plots.jl")

begin # read OCV look-up-table 
    df_ocv = CSV.File("data/ocv-yuasa.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc, extrapolation=ExtrapolationType.Constant)
    focv⁻¹ = LinearInterpolation(df_ocv.soc, df_ocv.ocv, extrapolation=ExtrapolationType.Constant)
end

begin
    fR0(s) = 0.001 + 0.0002 * sinpi(-0.2 + s * 1.5)
    # fR0(s, T) = (0.001 + 0.0002 * sinpi(-0.2 + s * 1.5)) * exp(kT * (1 / T - 1 / T0))
end

begin # read current profile
    df = CSV.File("data/yuasa-p1-m1.csv") |> DataFrame
    select!(df, :time => :t, :module_current => :i, :module_temperature => :T)
    fi = ConstantInterpolation(-df.i, df.t)
end


params = Dict(
    :cell_1 => Dict(:soc => 0.5, :Q => 63, :R0 => 1.5e-3, :R1 => 0.8e-3, :τ1 => 60, :focv => focv),
    :cell_2 => Dict(:soc => 0.51, :Q => 64, :R0 => 1.5e-3, :R1 => 0.8e-3, :τ1 => 60, :focv => focv),
    :cell_3 => Dict(:soc => 0.53, :Q => 65, :R0 => 1.5e-3, :R1 => 0.8e-3, :τ1 => 60, :focv => focv),
    :cell_4 => Dict(:soc => 0.52, :Q => 64, :R0 => 1.5e-3, :R1 => 0.8e-3, :τ1 => 60, :focv => focv),
    :cell_5 => Dict(:soc => 0.51, :Q => 66, :R0 => 1.5e-3, :R1 => 0.8e-3, :τ1 => 60, :focv => focv),
    :cell_6 => Dict(:soc => 0.53, :Q => 65, :R0 => 1.5e-3, :R1 => 0.8e-3, :τ1 => 60, :focv => focv),
    :cell_7 => Dict(:soc => 0.53, :Q => 65, :R0 => 1.5e-3, :R1 => 0.8e-3, :τ1 => 60, :focv => focv),
    :cell_8 => Dict(:soc => 0.52, :Q => 66, :R0 => 1.5e-3, :R1 => 0.8e-3, :τ1 => 60, :focv => focv),
    :cell_9 => Dict(:soc => 0.51, :Q => 64, :R0 => 1.5e-3, :R1 => 0.8e-3, :τ1 => 60, :focv => focv),
    :cell_10 => Dict(:soc => 0.53, :Q => 67, :R0 => 1.5e-3, :R1 => 0.8e-3, :τ1 => 60, :focv => focv),
    :cell_11 => Dict(:soc => 0.54, :Q => 66, :R0 => 1.5e-3, :R1 => 0.8e-3, :τ1 => 60, :focv => focv),
    :cell_12 => Dict(:soc => 0.54, :Q => 65, :R0 => 1.5e-3, :R1 => 0.8e-3, :τ1 => 60, :focv => focv)
)

tspan = (0, 8 * 3600)
df = simulate_module(params, tspan; Ts=10.0)
plot_dataset(df)

df_cell = select_cell_dataset(df, 1)
zt = fit_zscore(df_cell)
dfn = normalize_data(zt, df_cell)

us = [(; i, q) for (i, q) in zip(dfn.i, dfn.q)]
ys = [SA[y] for y in dfn.v]



θ = (; # tunable (hyper)params
    ocv=(; σ=0.5, ℓ=0.5),
    r0=(; σ=0.001, ℓ=1.5),
    rc=(;
        σ0_v=1e-3, σ1_v=1.0e-4,
        σ0_r=50e-3, σ1_r=3e-6,
        σ0_τ=50, σ1_τ=2e-6,
        # σ0_v=1e-3, σ1_v=10,
        # σ0_r=5e-3, σ1_r=1e-3,
        # σ0_τ=10, σ1_τ=2e-3,
    ),
    vσ=3e-3,
)
ϑ = (; # non-tunable params
    Ts=10.0,
    r0=(; r0=1.0e-3),
    rc=(; v0=0.0, r0=0.8e-3, τ0=60.0,)
)


kf = build_kf(θ, ϑ, df, zt)
# @time run_kf!(kf, us, ys)

sol = LLPF.forward_trajectory(kf, us, ys)

# @benchmark run_kf!($kf, $us, $ys)
# @profview_allocs prof_kf(kf, us, ys)

# 

StatsBase.reconstruct(zt.r, getindex.(getindex.(ComponentVector.(sol.xt, xid), :rc), :r)) |> lines

StatsBase.reconstruct(zt.σ, getindex.(getindex.(ComponentVector.(sol.xt, xid), :rc), :v)) |> lines

# TODO
# - fit the individual cells and validate
# - fit model with module data
# - compare SOH estimation
# - compare SOC estimation

# let 
#     cells = [Symbol("cell_$i") for i in 1:12]
#     socs = [params[cell][:soc] for cell in cells]
#     barplot(socs, color=socs)
# end


let fig = Figure(size=(600, 600))
    colors = Makie.wong_colors()
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / mΩ"

    # predict new points -> mean and std
    qmin, qmax = df.q |> extrema
    bgp = qmin:0.01:qmax
    bgpn = StatsBase.transform(zt.q, bgp)
    ocv = predict_gp(kf, bgpn, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
    ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)


    # plot results 
    # lines!(ax[1], 0:0.01:1, f1.(0:0.01:1), color=colors[1], label="f1(x)")
    lines!(ax[1], bgp, ocvμ)
    band!(ax[1], bgp, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha=0.8)
    # scatter!(ax[1], df_train.s, df.y, color=(:red, 0.5), label="Data")
    # axislegend(ax[1]; merge=true, position=:lt)

    # predict new points -> mean and std
    r0 = predict_gp(kf, bgpn, :r0)
    rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1e3
    rσ = StatsBase.reconstruct(zt.r, r0.σ) * 1e3

    # plot results 
    # lines!(ax[2], 0:0.01:1, f2.(0:0.01:1), color=colors[1], label="f2(x)")
    lines!(ax[2], bgp, rμ)
    band!(ax[2], bgp, rμ + 2rσ, rμ - 2rσ, alpha=0.8)
    # axislegend(ax[2]; merge=true, position=:lt)
    # end
    fig
end

