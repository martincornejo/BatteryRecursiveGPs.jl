using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks
using OrdinaryDiffEq
using NonlinearSolve
using Random
using StatsBase
using CSV, DataFrames, DataInterpolations
using OptimizationOptimJL
using Optimization
using LineSearches
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
include("utils.jl")
include("model_yuasa_q.jl")
include("soc_estimation.jl")
include("q_estimation.jl")

## dataset
begin # read OCV look-up-table 
    df_ocv = CSV.File("data/ocv-yuasa.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc, extrapolation=ExtrapolationType.Constant)
    focv⁻¹ = LinearInterpolation(df_ocv.soc, df_ocv.ocv, extrapolation=ExtrapolationType.Constant)
end

begin
    fR0(s) = 0.001 + 0.0004 * sinpi(-0.2 + s * 1.5)
end

begin # read current profile
    df = CSV.File("data/yuasa-p1-m1.csv") |> DataFrame
    select!(df, :time => :t, :module_current => :i, :module_temperature => :T)
    fi = ConstantInterpolation(-df.i, df.t)
    #fi = ConstantInterpolation(-df.i * 0.6, df.t)
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

function loss_function(u,p)
    (;ϑ, df, zt, us,ys) = p
    cost = 0.0
    θ_cons = softplus.(u)
    kf = build_kf(θ_cons, ϑ, df, zt)
    
    for (u, y) in zip(us,ys)
        ll, e = correct!(kf, u, y, kf.p, t)
        predict!(kf, u)
        cost += dot(e,1,e)
    end
    cost
 
end

begin
    cell_id = 1
    df_cell = select_cell_dataset(df, cell_id)
    zt = fit_zscore(df_cell)
    dfn = normalize_data(zt, df_cell)

    Random.seed!(42)
    us = [(;  i = i, q = q) for (i, q) in zip(dfn.i, dfn.q)]
    #us = [(; i) for i in dfn.i]
    #us = [(; i=i + 0.01) for i in dfn.i]
    ys = [SA[y] for y in dfn.v]
    
    ## Tunable
    θ0 = ComponentVector(
        ocv =ComponentVector(
            σ = 0.52,
            ℓ = 0.5,
        ),
        r0 = ComponentVector(
            σ = 0.8,
            ℓ = 0.8,
        ),
        q = ComponentVector(
            σ1 = 1e-5,
        ),
        vσ = 3e-3,
    )

    ## Non-tunable
    ϑ = ComponentVector(
        Ts = 10.0,
        r0 = ComponentVector(
            r0 = 1.0e-1
        ),
        rc = ComponentVector(
            σ0_v = 1e-3,
            σ1_v = 1.0e-4,
            σ0_r = log(0.8 * 1e-3) - log(1.4e-3),
            σ1_r = 3e-6,
            σ0_τ = log(60) - log(120),
            σ1_τ = 2e-6,
            v0 = 0.0,
            r0 = 1.4e-3,
            τ0 = 120.0,
        )
    )

end

begin
    p = (;ϑ, df, zt, us,ys)
    u0  = inv_softplus.(θ0)
    loss_function(u0,p)
end

begin
    adtype = AutoForwardDiff()
    f = OptimizationFunction(loss_function, adtype)
    prob = OptimizationProblem(f, u0, p)
    
    alg = LBFGS(linesearch=LineSearches.BackTracking())
    sol = solve(prob,
        alg,
        reltol=1e-4,
        show_trace = true
    ) 

    θ = softplus.(sol.u)
end 
println(θ)

title_sim = "q estimation, perfect i vσ = $(round(θ.vσ, digits = 3))"
title_q = "σ1 = $(round(θ.q.σ1, digits = 3))"
title_ecm = "ocv, σ = $(round(θ.ocv.σ, digits = 3)) , ℓ =$(round(θ.ocv.ℓ, digits = 3)),
R0, σ = $(round(θ.r0.σ, digits = 3)) , ℓ =$(round(θ.r0.ℓ, digits = 3))"

begin
    kf = build_kf(θ, ϑ, df_cell, zt)
    (;evo, v_sim) = run_sim!(kf, us, ys, [], predict_fun = model_predict)
    vμ = StatsBase.reconstruct(zt.v, v_sim.vμ)
    vσ = StatsBase.reconstruct(zt.σ, v_sim.vσ)
    
    plot_simulation(vμ, vσ, df_cell, title = title_sim) |> display
    
    #plot_q_estimation(evo.evoμ, evo.evoΣ, df_cell, zt, title = title_q) |> display
    plot_ecm(kf, df_cell, zt; focv,fR0, Q=params[:cell_1][:Q], external_cc=true, title = title_ecm) |> display

    plot_rc_evo(kf,evo.evoμ,evo.evoΣ,zt, title = title_sim) |> display

    Q´ = calc_Q(kf, df_cell, zt, focv⁻¹) |> Measurements.value
    s´ = calc_soc0(kf, df_cell, zt, focv⁻¹) |> Measurements.value

    @info "cell $cell_id:" Q´ s´
end
