module BatteryRecursiveGPs

using RecursiveGPs
using AbstractGPs
using LowLevelParticleFilters
import LowLevelParticleFilters as LLPF
using LinearAlgebra
using Distributed: WorkerPool, workers, remotecall

using StatsBase
using StaticArrays
import ComponentArrays: ComponentVector, ComponentMatrix, getaxes

using DataInterpolations
using DataFrames
using Measurements
using Printf

abstract type AbstractBatteryModel end

# state-only models: frozen ECM parameters, online estimation of charge (+ RC voltage)
abstract type AbstractBatteryStateModel <: AbstractBatteryModel end

include("models/components.jl")
include("models/yuasa.jl")
include("models/yuasa2rc.jl")
include("models/fenecon.jl")
include("models/fenecon2rc.jl")
include("models/rcgp.jl")
include("models/rcgp2rc.jl")
include("models/yuasa_soc.jl")
include("models/fenecon_soc.jl")
include("models/fenecon2rc_soc.jl")
include("models/rcgp_soc.jl")
include("models/rcgp2rc_soc.jl")
include("runner.jl")
include("fit_model.jl")
include("analysis.jl")
include("composite_ocv.jl")

# model types
export AbstractBatteryModel, AbstractBatteryStateModel
export YuasaModel, Yuasa2RCModel, FeneconModel, Fenecon2RCModel, RCGPModel, RCGP2RCModel
export YuasaStateModel, FeneconStateModel, Fenecon2RCStateModel, RCGPStateModel, RCGP2RCStateModel

# runner
export run_kf!, reduce_sol, reinit_kf!
export run_kf_smoother!, smooth_kf!
export fit_model, fit_models_threaded, fit_models_distributed
export eval_model

# model components
export ColoumbCounting, dynamics_cc
export Arrhenius, arrhenius_factor
export R0, RC, dynamics_rc, RC_VTau, dynamics_rc_vτ

# analysis
export calc_deltaq, calc_Q, calc_soc0, calc_soh # single cell
export gls_fit, calc_wls, gp_ocv, gp_r0, gp_r1, gp_r2, charge_trajectory, voltage_error # improved GLS-based estimation
export calc_Q_pack, calc_soc_pack, calc_soh_pack, calc_Q_utilization # battery pack
export fit_composite_ocv, rescale_composite_ocv, fit_cells_to_reference # composite OCV from cell posteriors

end # module BatteryRecursiveGPs
