module RecursiveGPs

export RGP, measurement_gp, uncertainty_gp

using Statistics
using LinearAlgebra
using AbstractGPs
using Distributions # is this necessary?
using StaticArrays
# using ComponentArrays

using LowLevelParticleFilters

include("rgp.jl")
include("kf.jl")


end