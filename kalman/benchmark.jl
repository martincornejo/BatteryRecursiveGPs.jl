using LowLevelParticleFilters
using Distributions
using LinearAlgebra
using MLUtils: DataLoader
using DataFrames
using CSV
using Statistics
using AbstractGPs
using DataInterpolations
using CairoMakie
using StatsBase
using BenchmarkTools
using Profile
using Revise
using Optim
includet("battModel/rc.jl")
includet("battModel/r0_ocv.jl")
includet("battModel/batt.jl")
includet("battModel/rgp.jl")
import ComponentArrays: ComponentVector, getaxes, ComponentMatrix

#### BENCHMARK RGP ###

begin
    x = collect(0:0.05:1)
    y = sin.(x) + 4 * x .^ 2 - x / 32
end

begin
    l_ocv = 0.2
    σ_ocv = 0.8
    b0 = collect(0:0.2:1)  # basis vector for OCV
    tr_b = ZScoreTransform(1, 1, [0.0], [1.0])
    b0 = StatsBase.transform(tr_b, b0)

    gp_ocv = GP(ZeroMean(), LinearKernel() + σ_ocv * with_lengthscale(SEKernel(), l_ocv))
    ocv = RGP(
        gp_ocv, b0;
        σ2=1e-5)
end

begin
    b = [0.4]
    println(cov(gp_ocv, b, b0))
    println(gp_ocv.kernel.(b, b0))
end

@benchmark begin
    gp_ocv.kernel.($b, $b0)
end

@benchmark begin
    cov(gp_ocv, $b, $b0)
end

#### dynamics_gp ###
begin
    x = zeros(size(b0))
    u = (; b=rand(1)
    )
    p = ocv.p
    t = 1
end

@benchmark begin
    measurement_gp($x, $u, $p, $t)
end


@profview begin
    for i in 1:1000
        measurement_gp(x, u, p, t)
    end
end

begin
    b = rand(1)
    x = zeros(size(b0))
end

begin
    k = similar(b0)
end
@benchmark begin
    gp_ocv.kernel(b0, b)
end

@profview_allocs begin
    for i in 1:1000
        measurement_gp(x, u, p, t)
    end
end


