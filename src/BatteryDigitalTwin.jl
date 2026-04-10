module BatteryDigitalTwin

using RecursiveGPs
using AbstractGPs
using LowLevelParticleFilters
import LowLevelParticleFilters as LLPF
using LinearAlgebra

using StatsBase
using StaticArrays
import ComponentArrays: ComponentVector, ComponentMatrix, getaxes

using DataInterpolations
using Measurements
using Printf

using CairoMakie # use Makie instead?

abstract type AbstractBatteryModel end

include("models/components.jl")
include("models/yuasa.jl")
include("models/fenecon.jl")
include("models/yuasa_soc.jl")
include("runner.jl")
include("analysis.jl")
include("plot.jl")

# model types
export AbstractBatteryModel
export YuasaModel, FeneconModel, YuasaStateModel

# runner
export run_kf!, reinit_kf!

# model components
export ColoumbCounting, dynamics_cc
export Arrhenius, arrhenius_factor
export R0, RC, dynamics_rc

# analysis
export calc_deltaq, calc_Q, calc_soc0, calc_soh # single cell
export gls_fit, calc_wls, gp_ocv # improved GLS-based estimation
export calc_Q_pack, calc_soc_pack, calc_soh_pack, calc_Q_utilization # battery pack

# plotting — model-agnostic
export plot_sim
export plot_ecm, plot_ecm!, plot_ecms
export plot_q_trajectory, plot_q_estimation, plot_q_estimation_state
export plot_soc_trajectories
export plot_module_soh, plot_cell_soh_hist, plot_module_inhomogenity

# plotting — model-specific (YuasaModel)
export plot_ecms_norm
export plot_rc_param_trajectory, plot_arrhenius_param_trajectory
export animate_ecm_evolution, animate_model

end # module BatteryDigitalTwin
