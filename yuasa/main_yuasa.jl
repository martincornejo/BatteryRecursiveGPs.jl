using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks
using OrdinaryDiffEq

using Random
using StatsBase
using CSV, DataFrames, DataInterpolations

using Measurements
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
include("analysis.jl")
include("plots.jl")

include("model_yuasa_q.jl")
include("soc_estimation.jl")

## dataset
begin # read OCV look-up-table 
    df_ocv = CSV.File("data/ocv-yuasa.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc, extrapolation=ExtrapolationType.Constant)
    focv⁻¹ = LinearInterpolation(df_ocv.soc, df_ocv.ocv, extrapolation=ExtrapolationType.Constant)
end

begin
    fR0(s) = 0.001 + 0.0004 * sinpi(-0.2 + s * 1.5)
    # fR0(s) = 0.001 + 0.0002 * sinpi(-0.2 + s * 1.5)
    # fR0(s, T) = (0.001 + 0.0002 * sinpi(-0.2 + s * 1.5)) * exp(kT * (1 / T - 1 / T0))
end

begin # read current profile
    df = CSV.File("data/yuasa-p1-m1.csv") |> DataFrame
    select!(df, :time => :t, :module_current => :i, :module_temperature => :T)
    # fi = ConstantInterpolation(-df.i, df.t)
    fi = ConstantInterpolation(-df.i * 0.6, df.t)
end

params = Dict(
    # :cell_1 => Dict(:soc => 0.5, :Q => 38, :R0 => 1.5e-3, :R1 => 0.8e-3, :τ1 => 60, :focv => focv),
    :cell_1 => Dict(:soc => 0.5, :Q => 63, :R1 => 0.8e-3, :τ1 => 60, :focv => focv, :fR0 => fR0),
    :cell_2 => Dict(:soc => 0.51, :Q => 64, :R1 => 0.8e-3, :τ1 => 60, :focv => focv, :fR0 => fR0),
    :cell_3 => Dict(:soc => 0.53, :Q => 65, :R1 => 0.8e-3, :τ1 => 60, :focv => focv, :fR0 => fR0),
    :cell_4 => Dict(:soc => 0.52, :Q => 64, :R1 => 0.8e-3, :τ1 => 60, :focv => focv, :fR0 => fR0),
    :cell_5 => Dict(:soc => 0.51, :Q => 66, :R1 => 0.8e-3, :τ1 => 60, :focv => focv, :fR0 => fR0),
    :cell_6 => Dict(:soc => 0.53, :Q => 65, :R1 => 0.8e-3, :τ1 => 60, :focv => focv, :fR0 => fR0),
    :cell_7 => Dict(:soc => 0.53, :Q => 65, :R1 => 0.8e-3, :τ1 => 60, :focv => focv, :fR0 => fR0),
    :cell_8 => Dict(:soc => 0.52, :Q => 66, :R1 => 0.8e-3, :τ1 => 60, :focv => focv, :fR0 => fR0),
    :cell_9 => Dict(:soc => 0.51, :Q => 64, :R1 => 0.8e-3, :τ1 => 60, :focv => focv, :fR0 => fR0),
    :cell_10 => Dict(:soc => 0.53, :Q => 67, :R1 => 0.8e-3, :τ1 => 60, :focv => focv, :fR0 => fR0),
    :cell_11 => Dict(:soc => 0.54, :Q => 66, :R1 => 0.8e-3, :τ1 => 60, :focv => focv, :fR0 => fR0),
    :cell_12 => Dict(:soc => 0.54, :Q => 65, :R1 => 0.8e-3, :τ1 => 60, :focv => focv, :fR0 => fR0)
)

Ts = 10.0
tspan = (0, 8 * 3600)
df = simulate_module(params, tspan; Ts)
plot_dataset(df)


## cell 1
res = let
    res = Dict()
    for cell_id in 1:12
        df_cell = select_cell_dataset(df, cell_id)
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
            ),
            vσ=3e-3,
        )
        ϑ = (; # non-tunable params
            Ts=10.0,
            r0=(; r0=1.0e-3),
            rc=(; v0=0.0, r0=0.8e-3, τ0=60.0,)
        )

        kf = build_kf(θ, ϑ, df_cell, zt)


        v_sim = run_sim!(kf, us, ys, [])

        vμ = StatsBase.reconstruct(zt.v, v_sim.vμ)
        vσ = StatsBase.reconstruct(zt.σ, v_sim.vσ)

        plot_simulation(vμ, vσ, df_cell) |> display
        plot_ecm(kf, df_cell, zt; focv, Q=params[Symbol("cell_$cell_id")][:Q]) |> display


        Q = params[Symbol("cell_$cell_id")][:Q]
        Q´ = calc_Q(kf, df_cell, zt, focv⁻¹) |> Measurements.value
        s = params[Symbol("cell_$cell_id")][:soc]
        s´ = calc_soc0(kf, df_cell, zt, focv⁻¹) |> Measurements.value

        @info "cell $cell_id:" Q Q´ s s´

        us2 = [(; i, î=î + 0.5) for (i, î) in zip(dfn.i, df_cell.i)]
        # us2 = [(; i, î) for (i, î) in zip(dfn.i, df_cell.i)]
        ys2 = [SA[y] for y in dfn.v]

        kf2 = build_kf_state(kf, s´, s´, Q´, Ts, zt)

        sol = LLPF.forward_trajectory(kf2, us2, ys2)
        sμ = sol.xt .|> first
        sσ = sol.Rt .|> first .|> sqrt

        î = [u.î for u in us2]
        ŝ = s´ .+ cumsum(î) * Ts / 3600 / Q´

        plot_soc_estimation(df_cell.t, sμ, sσ, df_cell.s, ŝ) |> display

        res[Symbol("cell_$cell_id")] = Dict(
            :kf => kf,
            :Q => Q´,
            :soc => s´,
            :v => vμ .± vσ,
            :s => sμ .± sσ,
        )
    end
    res
end

plot_simulation_cell(df, res) |> display
plot_soc_estimation_cell(df, res, params) |> display


## module
let
    zt = fit_zscore(df)
    dfn = normalize_data(zt, df)

    us = [(; i, q) for (i, q) in zip(dfn.i, dfn.q)]
    ys = [SA[y] for y in dfn.v]

    θ = (; # tunable (hyper)params
        ocv=(; σ=0.5, ℓ=0.5),
        r0=(; σ=0.001, ℓ=1.5),
        rc=(;
            σ0_v=1e-3, σ1_v=1.0e-4,
            σ0_r=50e-3, σ1_r=3e-6,
            σ0_τ=50, σ1_τ=2e-6,
        ),
        q=(;
            q0=0.0,
            σ1=0.0,
        ),
        vσ=3e-3,
    )
    ϑ = (; # non-tunable params
        Ts=10.0,
        r0=(; r0=12 * 1.0e-3),
        rc=(; v0=0.0, r0=12 * 0.8e-3, τ0=60.0,)
    )

    kf = build_kf(θ, ϑ, df, zt)

    v_sim = run_sim!(kf, us, ys, [])

    vμ = StatsBase.reconstruct(zt.v, v_sim.vμ)
    vσ = StatsBase.reconstruct(zt.σ, v_sim.vσ)

    plot_simulation(vμ, vσ, df) |> display
    plot_ecm(kf, df, zt, Q=0) |> display

    Q´ = calc_Q(kf, df, zt, focv⁻¹; n=12) |> Measurements.value
    s´ = calc_soc0(kf, df, zt, focv⁻¹; n=12) |> Measurements.value

    @info "module:" Q´ s´

    # us = [(; i, î=î + randn()) for (i, î) in zip(dfn.i, df.i)]
    us2 = [(; i, î=î + 0.5) for (i, î) in zip(dfn.i, df.i)]
    ys2 = [SA[y] for y in dfn.v]

    kf2 = build_kf_state(kf, 0.5, s´, Q´, Ts, zt)
    sol = LLPF.forward_trajectory(kf2, us2, ys2)

    î = [u.î for u in us2]
    ŝ = s´ .+ cumsum(î) * Ts / 3600 / Q´
    soc_module = calc_module_soc(df, params)

    μ = sol.xt .|> first
    σ = sol.Rt .|> first .|> sqrt
    plot_soc_estimation(df.t, μ, σ, soc_module, ŝ) |> display
end



# TODO
# - fit the individual cells and validate
# - fit model with module data
# - compare SOH estimation
# - compare SOC estimation

# TODO:
# - Extract SOH and analyze error
# - simulate and analyze model error
# - perform SOC estimation and analyze error

# TODO:
# - differentiate between capacity loss and virtual (temporary) capacity loss

# let 
#     cells = [Symbol("cell_$i") for i in 1:12]
#     socs = [params[cell][:soc] for cell in cells]
#     barplot(socs, color=socs)
# end

# calc_soh(kf, df, zt, focv⁻¹, 100)
# calc_Q(kf, df, zt, focv⁻¹)
# calc_soc0(kf, df, zt, focv⁻¹)

# StatsBase.reconstruct(zt.r, getindex.(getindex.(ComponentVector.(sol.xt, xid), :rc), :r)) |> lines
# StatsBase.reconstruct(zt.σ, getindex.(getindex.(ComponentVector.(sol.xt, xid), :rc), :v)) |> lines
# r1 = StatsBase.reconstruct(zt.r, [getindex(getindex(ComponentVector(kf.x, xid), :rc), :r)]) |> first

## cell 1
# simulataneous parameter and state estimation
begin

    cell_id = 1
    df_cell = select_cell_dataset(df, cell_id)
    zt = fit_zscore(df_cell)
    dfn = normalize_data(zt, df_cell)

    Random.seed!(42)
    us = [(; i) for i in dfn.i]
    # us = [(; i=i + 0.01, q) for (i, q) in zip(dfn.i, dfn.q)]
    ys = [SA[y] for y in dfn.v]

    θ = (; # tunable (hyper)params
        ocv=(; σ=1.2, ℓ=0.8),
        r0=(; σ=0.8, ℓ=0.8),
        # ocv=(; σ=0.5, ℓ=0.5),
        # r0=(; σ=0.001, ℓ=1.5),
        rc=(;
            σ0_v=1e-3, σ1_v=1.0e-4,
            σ0_r=50e-3, σ1_r=3e-6,
            σ0_τ=50, σ1_τ=2e-6,
        ),
        # q=(; σ1=5e-6,),
        # q=(; σ1=1e-4),
        q=(; σ1=0.0),
        vσ=3e-3,
    )
    ϑ = (; # non-tunable params
        Ts=10.0,
        r0=(; r0=1.0e-1),
        rc=(; v0=0.0, r0=0.8e-3, τ0=60.0,)
    )

    kf = build_kf_q(θ, ϑ, df_cell, zt)

    # plot_ecm(kf, df_cell, zt; focv, Q=params[:cell_1][:Q], external_cc=false) |> display

    if false
        v_sim = run_sim!(kf, us, ys, [])
        vμ = StatsBase.reconstruct(zt.v, v_sim.vμ)
        vσ = StatsBase.reconstruct(zt.σ, v_sim.vσ)

        plot_simulation(vμ, vσ, df_cell) |> display
        plot_ecm(kf, df_cell, zt, Q=0) |> display
    else
        evo = save_train_kf(kf, us, ys; step_size=1)

        # plot_q_estimation(evo, df_cell, zt) |> display
        plot_ecm(kf, df_cell, zt; focv, Q=params[:cell_1][:Q], external_cc=false) |> display
    end

    # begin
    #     fig = Figure()
    #     ax = [Axis(fig[i, 1]) for i in 1:2]
    #     î = StatsBase.reconstruct(zt.i, [u.i for u in us])
    #     q̂ = cumsum(î) * Ts / 3600
    #     lines!(ax[1], df_cell.t / 3600, q̂)
    #     lines!(ax[1], df_cell.t / 3600, df_cell.q)
    #     lines!(ax[2], df_cell.t / 3600, q̂ - df_cell.q; color=:red)
    #     fig
    # end |> display


    Q´ = calc_Q(kf, df_cell, zt, focv⁻¹) |> Measurements.value
    s´ = calc_soc0(kf, df_cell, zt, focv⁻¹) |> Measurements.value

    @info "cell $cell_id:" Q´ s´
end
#