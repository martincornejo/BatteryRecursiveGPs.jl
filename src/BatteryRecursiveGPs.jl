module BatteryRecursiveGPs

using RecursiveGPs
using AbstractGPs
using LowLevelParticleFilters
import LowLevelParticleFilters as LLPF
using LinearAlgebra
using Distributed: WorkerPool, workers, remotecall

using StatsBase
using StaticArrays
import ComponentArrays: ComponentVector, ComponentMatrix

using DataInterpolations
using Measurements

"""
Supertype of the battery models. A subtype wraps an `ExtendedKalmanFilter` in its `kf` field
and defines [`reduce_sol`](@ref) and, for full models, `reinit_kf!`.
"""
abstract type AbstractBatteryModel end

"""
Supertype of the state-only models: the ECM parameters are frozen from an earlier fit and only
the charge (and RC voltage) are estimated online.
"""
abstract type AbstractBatteryStateModel <: AbstractBatteryModel end

include("models/components.jl")
include("models/fenecon.jl")
include("models/fenecon2rc.jl")
include("models/yuasa.jl")
include("models/fenecon_soc.jl")
include("models/fenecon2rc_soc.jl")
include("models/yuasa_soc.jl")
include("runner.jl")
include("fit_model.jl")
include("analysis.jl")
include("composite_ocv.jl")

# model types
export AbstractBatteryModel, AbstractBatteryStateModel
export FeneconModel, Fenecon2RCModel, YuasaModel
export FeneconStateModel, Fenecon2RCStateModel, YuasaStateModel

# runner
export run_kf!, reduce_sol, reinit_kf!
export run_kf_smoother!, smooth_kf!
export fit_model, fit_models_threaded, fit_models_distributed, fit_ocv_curve
export eval_model

# model components
export CoulombCounting, dynamics_cc
export Arrhenius, arrhenius_factor
export R0, RC, dynamics_rc, RC_VTau, dynamics_rc_vτ

# analysis
export gp_ocv, gp_r1, charge_trajectory, voltage_error # posteriors in physical units
export calc_Q_pack, calc_soc_pack, calc_soh_pack # battery pack
export fit_composite_ocv, rescale_composite_ocv, fit_cells_to_reference # composite OCV from cell posteriors

end # module BatteryRecursiveGPs
