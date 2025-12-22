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
include("q_estimation.jl")
include("model_yuasa_q.jl")

# TODOs:
# Push correction in RecursiveGPs- DONE
# Test different solvers (and cofings) for NonLinearSolve -CURRENT
# Test dual-KF approach for q estimation ---> Current
# Test cross-validation for fitting
# Test with Optimization.jl --> set bounds

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
    θ = ComponentVector(
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
    title_sim = "ECM.Pre-q, perfect i vσ = $(round(θ.vσ, digits = 3))"
    title_q = "ECM.Pre-q,With updating curve σ1 = $(round(θ.q.σ1, digits = 7))"
    title_ecm = "ECM.Pre-q,ocv, σ = $(round(θ.ocv.σ, digits = 3)) , ℓ =$(round(θ.ocv.ℓ, digits = 3)),
    R0, σ = $(round(θ.r0.σ, digits = 3)) , ℓ =$(round(θ.r0.ℓ, digits = 3))"
    kf_ecm = build_kf(θ,ϑ,df,zt)

    ## Training kf_ecm
    (;evo, v_sim) = run_sim!(kf_ecm, us, ys, [], predict_fun = model_predict_2)

    ### Restarting rc value to starting points, 43 is hard coded value
    kfx = ComponentVector(kf_ecm.x, kf_ecm.p.xid)
    ocv = measurement_gp(kf_ecm.p.ocv, kfx.ocv,us[1].q)
    r0 = measurement_gp(kf_ecm.p.r0, kfx.r0, us[1].q)
    kf_ecm.x[43] = first(ys[1] .- ocv .- r0 .* us[1].i)

    vμ = StatsBase.reconstruct(zt.v, v_sim.vμ)
    vσ = StatsBase.reconstruct(zt.σ, v_sim.vσ)
    plot_simulation(vμ, vσ, df_cell, title = title_sim) |> display

    plot_ecm(kf_ecm, df_cell, zt; focv,fR0, Q=params[:cell_1][:Q], external_cc=true, title = title_ecm) |> display
    plot_rc_evo(kf_ecm,evo.evoμ,evo.evoΣ,zt, title = title_sim) |> display
end

begin
    ## Creating kf_dual
    kf_dual = build_kf_dual(kf_ecm, θ, ϑ, zt)
    
    ## Training kf_dual and kf_ecm at same time
    vμ_ecm, vσ_ecm = Float64[], Float64[]
    evoμ_ecm, evoΣ_ecm = [], []
    xid_ecm, Σid_ecm = kf_ecm.p.xid, kf_ecm.p.Σid
    
    vμ_dual, vσ_dual = Float64[], Float64[]
    evoμ_dual, evoΣ_dual = [], []
    xid_dual, Σid_dual = kf_dual.p.xid, kf_dual.p.Σid

    n = 1
    for (u, y) in zip(us, ys)
        (vμᵢ_dual, vσᵢ_dual) = model_predict_2(kf_ecm, u)
        push!(vμ_dual, vμᵢ_dual)
        push!(vσ_dual, vσᵢ_dual)
        kf_dual(u, y)
        
        u = (;i = u.i, q = kf_dual.x[1]) ## Updating u.i so is the q value
        (vμᵢ_ecm, vσᵢ_ecm) = model_predict_2(kf_ecm, u)
        push!(vμ_ecm, vμᵢ_ecm)
        push!(vσ_ecm, vσᵢ_ecm)
        kf_ecm(u, y)

      
        push!(evoμ_dual, copy(ComponentVector(kf_dual.x, xid_dual)))
        push!(evoΣ_dual, copy(ComponentMatrix(kf_dual.R, Σid_dual)))
        push!(evoμ_ecm, copy(ComponentVector(kf_ecm.x, xid_ecm)))
        push!(evoΣ_ecm, copy(ComponentMatrix(kf_ecm.R, Σid_ecm)))
 
    end

    v_sim_dual = (; vμ_dual, vσ_dual)
    evo_dual = (; evoμ_dual, evoΣ_dual)

    v_sim_ecm = (; vμ_ecm, vσ_ecm)
    evo_ecm = (; evoμ_ecm, evoΣ_ecm)
    nothing
end



### Plooting ecm outputs
begin
    title_sim = "ECM.Post-q, perfect i vσ = $(round(θ.vσ, digits = 3))"
    title_q = "ECM.Post-q, With updating curve σ1 = $(round(θ.q.σ1, digits = 7))"
    title_ecm = "ECM.Post-q, ocv, σ = $(round(θ.ocv.σ, digits = 3)) , ℓ =$(round(θ.ocv.ℓ, digits = 3)),
    R0, σ = $(round(θ.r0.σ, digits = 3)) , ℓ =$(round(θ.r0.ℓ, digits = 3))"

    vμ = StatsBase.reconstruct(zt.v, v_sim_ecm.vμ_ecm)
    vσ = StatsBase.reconstruct(zt.σ, v_sim_ecm.vσ_ecm)
    plot_simulation(vμ, vσ, df_cell, title = title_sim) |> display
    plot_ecm(kf_ecm, df_cell, zt; focv,fR0, Q=params[:cell_1][:Q], external_cc=true, title = title_ecm) |> display
    plot_rc_evo(kf_ecm,evo_ecm.evoμ_ecm,evo_ecm.evoΣ_ecm,zt, title = title_sim) |> display

    Q´ = calc_Q(kf_ecm, df_cell, zt, focv⁻¹) |> Measurements.value
    s´ = calc_soc0(kf_ecm, df_cell, zt, focv⁻¹) |> Measurements.value

    @info "cell $cell_id:" Q´ s´
end


### Plooting dual outputs
begin
    title_sim = "CC.Post-q, perfect i vσ = $(round(θ.vσ, digits = 3))"
    title_q = "CC.With updating curve σ1 = $(round(θ.q.σ1, digits = 7))"
    title_ecm = "CC.ocv, σ = $(round(θ.ocv.σ, digits = 3)) , ℓ =$(round(θ.ocv.ℓ, digits = 3)),
    R0, σ = $(round(θ.r0.σ, digits = 3)) , ℓ =$(round(θ.r0.ℓ, digits = 3))"

    vμ = StatsBase.reconstruct(zt.v, v_sim_dual.vμ_dual)
    vσ = StatsBase.reconstruct(zt.σ, v_sim_dual.vσ_dual)
    plot_simulation(vμ, vσ, df_cell, title = title_sim) |> display
    plot_q_estimation(evo_dual.evoμ_dual, evo_dual.evoΣ_dual, df_cell, zt, title = title_q) |> display

    @info "cell $cell_id:" Q´ s´
end