module RecursiveGPs

export RGP, measurement_gp, uncertainty_gp
export make_ekf, predict_gp

using Statistics
using LinearAlgebra
using AbstractGPs
using StaticArrays
using ForwardDiff # for jacobians
using PreallocationTools
using ComponentArrays

using LowLevelParticleFilters
import LowLevelParticleFilters as LLPF

include("rgp.jl")
include("kf.jl")
include("model.jl")


end