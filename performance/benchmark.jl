using LowLevelParticleFilters
using Distributions
using LinearAlgebra

using StableRNGs

using DataFrames

using AbstractGPs

using CairoMakie

using StaticArrays
import ComponentArrays: ComponentVector, ComponentMatrix, getaxes, @static_unpack

using ForwardDiff

using Statistics

using BenchmarkTools
using JET

include("combined-gp.jl")
include("combined-gp-allocs.jl")



## synthetic dataset
f1(b) = 0.1 + 0.5 * b + 0.1 * sinpi(b * 2) # <- function to infer
f2(b) = exp(b)

df = let n = 100
    rng = StableRNG(123)
    b = 0.1 .+ rand(rng, n) / 1.5
    i = 0.2 .* randn(rng, n)
    y = @. f2(b) + i * f1(b)
    DataFrame(; b, i, y)
end
ys = [SA[y] for y in df.y]
us = [SA[x.b, x.i] for x in eachrow(df)]

## benchmark
function run_kf(kf, us, ys)
    for (u, y) in zip(us, ys)
        kf(u, y)
    end
end

kf1 = make_kf_opt();
kf2 = make_kf();

@benchmark run_kf($kf2, $us, $ys)
@benchmark run_kf($kf1, $us, $ys)

# profile allocations
@profview_allocs begin
    for i in 1:1000
        run_kf(kf1, us, ys)
    end
end


# TODO
# - [ ] fix KernelSum
# - [ ] further explore inplace vs. static approaches
# - [ ] cleanup (including cache)
# - [ ] create dispatchable structs