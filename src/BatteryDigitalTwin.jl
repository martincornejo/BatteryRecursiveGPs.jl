module BatteryDigitalTwin

using RecursiveGPs
using AbstractGPs
using LowLevelParticleFilters
import LowLevelParticleFilters as LLPF
using LinearAlgebra

using StatsBase
using StaticArrays
import ComponentArrays: ComponentVector, ComponentMatrix, getaxes

using Measurements
# using ForwardDiff

using CairoMakie # use Makie instead?

include("model.jl")
include("soc_estimation.jl")
include("analysis.jl")
include("plot.jl")

# modeling
export build_kf, run_kf!
export build_kf_state # soc estimation only

# model components
export CoulombCounting, dynamics_cc
export Arrhenius, arrhenius_factor
export R0, RC, dynamics_rc

# analysis
export calc_Q, calc_soc0, calc_soh # single cell
export calc_Q_pack, calc_soc_pack, calc_soh_pack, calc_Q_utilization # battery pack

# ploting
export plot_sim # single cell
export plot_ecm
export plot_rc_param_trajectory
export plot_q_estimation, plot_q_estimation_state
export plot_ecms, plot_ecms_norm # battery pack
export plot_soc_trajectories

export animate_model # single cell --> animation

end # module BatteryDigitalTwin
