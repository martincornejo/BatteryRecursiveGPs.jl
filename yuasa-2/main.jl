using DataFrames
using CSV
using Dates
using Intervals
using DataInterpolations
using StatsBase

# using GLMakie
using CairoMakie

using Measurements
using StaticArrays
using RecursiveGPs
using AbstractGPs
using LowLevelParticleFilters
import LowLevelParticleFilters as LLPF
using ForwardDiff
using LinearAlgebra
import ComponentArrays: ComponentVector, ComponentMatrix, getaxes

using LogExpFunctions
using Optimization
using OptimizationOptimJL
using LineSearches

using NonlinearSolve

include("model_yuasa.jl")
include("fit-model.jl")

## dataset
begin # read OCV look-up-table 
    df_ocv = CSV.File("data/ocv-yuasa.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc, extrapolation=ExtrapolationType.Constant)
    focv⁻¹ = LinearInterpolation(df_ocv.soc, df_ocv.ocv, extrapolation=ExtrapolationType.Constant)
end

datadir = "yuasa-2/data-yuasa-cycles-2/"
files = Dict(
    :cell_voltage => datadir * "cell_voltages.csv",
    :cell_soc => datadir * "cell_soc.csv",
    :module_voltage => datadir * "module_voltage.csv",
    :module_current => datadir * "module_current_average.csv",
    :derating_current => datadir * "derating_currents.csv",
    :battery_temperature => datadir * "battery_temperature.csv"
)

dateformat = dateformat"y-m-d H:M:S+00:00"
data = Dict(id => CSV.File(file; dateformat) |> DataFrame for (id, file) in files)


ti = Interval(DateTime("2025-12-10T14:02:00"), DateTime("2025-12-11T02:00:00"))
# ti = Interval(DateTime("2025-12-10T14:02:00"), DateTime("2025-12-11T02:30:00"))
# ti = Interval(DateTime("2025-12-10T13:57:20"), DateTime("2025-12-11T02:37:00"))
zt = fit_zscore()



begin
    θ0 = ComponentVector(; # tunable (hyper)params
        ocv=(; σ=0.5, ℓ=0.5),
        r0=(; σ=0.05, ℓ=0.9),
        vσ=3e-3,
    )
    ϑ = ComponentVector(; # non-tunable params
        Ts=1.0,
        r0μ=2.0e-3,
        # r0μ=1.0e-3,
        rc=(;
            v0=0.0, σ0_v=1e-3, σ1_v=1.0e-4,
            r0=1.0e-3, σ0_r=1.0e-3, σ1_r=0.0,
            # r0=1.0e-3, σ0_r=10e-3, σ1_r=3e-6,
            # τ0=60.0, σ0_τ=5, σ1_τ=2e-6,
            τ0=300.0, σ0_τ=30, σ1_τ=0.0,
            # τ0=15.0, σ0_τ=3.0, σ1_τ=0.0,
        ),
        # rc=(; v0=0.0, r0=0.8e-3, τ0=60.0,)
    )
    θ = ComponentVector(; θ0..., ϑ...)

    # u, y = cell_dataset(data, ti, 1, 1, 1)
    # u, y = cell_dataset(data, ti, 3, 2, 1)
    u, y = cell_dataset(data, ti, 3, 5, 12)
    zt = fit_zscore()
    kf = build_kf(θ, u, zt; n=21)
    sol = run_kf!(kf, u, y)
    # sol2 = run_kf!(kf, u, y; tt=27000)

    mae = mean(abs, StatsBase.reconstruct(zt.σ, sol.et)) .* 1e3
    @info ":" mae # ℓocv ℓr0

    # plot_rc_param_trajectory(kf, sol)
    plot_ecm(kf) |> display
    # plot_sim(kf, sol) |> display
    # plot_sim(kf, sol2) |> display
    # plot_module_dataset(data, 1, 9)
    # plot_sim(kf, sol2) |> display
end


kfs, sols = let # p = 3# , m = 5

    kfs = Dict()
    sols = Dict()

    for p in 1:3, m in 1:9, c in 1:12
        u, y = cell_dataset(data, ti, p, m, c)
        kf = build_kf(θ, u, zt)

        try
            sol = run_kf!(kf, u, y)
            kfs[(; p, m, c)] = kf
            sols[(; p, m, c)] = sol
            @info "Cell p:$(p), m:$(m), c:$(c) complete"
        catch e
            @warn "Cell p:$(p), m:$(m), c:$(c) failed"
        end
    end

    kfs, sols
end


f = extract_ocv(kfs[(; p=3, m=5, c=12)])

vlim = (3.8, 3.95)
params = Dict((; p, m, c) =>
    Dict(
        :Q => calc_Q(kfs[(; p, m, c)], f.ocv⁻¹; v=vlim),
        :soc => calc_soc0(kfs[(; p, m, c)], f.ocv⁻¹; v=vlim),
        :soh => calc_soh(kfs[(; p, m, c)], f.ocv⁻¹, 100; v=vlim),
    ) for m in 1:9, c in 1:12
)


for m in 1:9
    p = 3
    soh = [params[(; p, m, c)][:soh] for c in 1:12]
    soc = [params[(; p, m, c)][:soc] for c in 1:12]
    fig = Figure()
    ax = Axis(fig[1, 1])
    barplot!(ax, 1:12, soh)
    # barplot!(ax, 1:12, soc)
    ylims!(ax, 0.3, 1.0)
    # scatter!(ax, soh, soc)
    ax.yticks = 0.3:0.05:1.0

    ax.title = "Module $m"
    fig |> display
end

for m in 1:9, c in 1:12
    kf = kfs[(; p=3, m, c)]
    # fig = plot_ecm(kf) # |> display
    # ylims!(fig.content[1], 3.4, 4.1)
    # fig.content[1].title = "Cell $(c)"
    # fig |> display
    v = (3.8, 3.95)

    q = calc_deltaq(kf; v)
    soh = calc_soh(kf, focv⁻¹, 100; v)
    soc0 = calc_soc0(kf, focv⁻¹; v)
    @info "$m $c" q soh soc0

end


let p = 3, m = 2, c = 1
    id = (; p, m, c)
    kf = kfs[id]
    sol = sols[id]

    plot_rc_param_trajectory(kf, sol)
end


plot_ecms(kfs)
fig = plot_ecms2(kfs, f.ocv⁻¹, f.ocv; vlim=(3.8, 3.95))
for m in 1:9
    # fig = plot_ecms2(Dict(c => kfs[(; p=3, m, c)] for c in 1:12), f.ocv⁻¹, f.ocv; vlim=(3.8, 3.95))
    fig = plot_ecms(Dict(c => kfs[(; p=3, m, c)] for c in 1:12))
    fig.content[1].title = "Module $(m)"
    fig |> display
end



for m in 1:9, c in 1:12
    kf = kfs[(; p=3, m, c)]
    # fig = plot_ecm(kf) # |> display
    # ylims!(fig.content[1], 3.4, 4.1)
    # fig.content[1].title = "Cell $(c)"
    # fig |> display
    v = (3.8, 3.95)

    q = calc_deltaq(kf; v)
    soh = calc_soh(kf, focv⁻¹, 100; v)
    soc0 = calc_soc0(kf, focv⁻¹; v)
    @info "$m $c" q soh soc0

end

for p in 1:3, m in 1:9, c in 1:12
    kf = kfs[(; p, m, c)]
    plot_ecm(kf) |> display
end

for p in 1:3, m in 1:9
    fig = plot_ecms2(Dict(c => kfs[(; p, m, c)] for c in 1:12), focv⁻¹, focv)
    fig.content[1].title = "Phase $(p), Module $(m)"
    fig |> display
end


fig = plot_ecms(kfs)
fig = plot_ecms2(kfs, focv⁻¹, focv)





function plot_iv_data(zt, u, y)
    idx = map(y_ -> any(y_ .!== missing), y) |> findall
    v = StatsBase.reconstruct(zt.v, first.(y[idx]))
    q = StatsBase.reconstruct(zt.q, [_u.q for _u in u[idx]])
    i = StatsBase.reconstruct(zt.i, [_u.i for _u in u[idx]])

    scatter(q, v; color=abs.(i))
end



function plot_q_filter(kf, sol)
    (; xid, Σid, zt) = kf.p
    xs = ComponentVector.(sol.xt, xid)
    Σs = [ComponentMatrix(R, Σid) for R in sol.Rt]


    q = StatsBase.reconstruct(zt.q, [u.q for u in sol.ut])
    t = 1:length(q)

    qμ = StatsBase.reconstruct(zt.q, [x.cc.q for x in xs])
    qσ = StatsBase.reconstruct(zt.q, sqrt.([Σ[:cc, :cc][:q, :q] for Σ in Σs]))

    fig = Figure()
    ax = Axis(fig[1, 1])

    lines!(ax, q)
    lines!(ax, t, qμ; color=Cycled(2))
    band!(ax, t, qμ - 2qσ, qμ + 2qσ; color=Cycled(2))
    fig
end


let p = 3, m = 3, v1 = 3.7
    for c in 1:12
        kf = kfs[(; p, m, c)]
        Q0 = calc_ΔQ0(kf, v1)
        @info c Q0
    end
end


function calc_ΔQ0(kf, v1)
    zt = kf.p.zt

    q̂min, q̂max = extrema(kf.p.ocv.b0)
    q̂ = q̂min:0.01:q̂max
    q = StatsBase.reconstruct(zt.q, q̂)

    # OCV 
    ocv = predict_gp(kf, q̂, :ocv)
    μ = StatsBase.reconstruct(zt.v, ocv.μ)
    σ = StatsBase.reconstruct(zt.σ, ocv.σ)

    q1μ = q[findfirst(>=(v1), μ)]
    q1σ = q1μ - q[findfirst(>=(v1), μ + σ)]
    q1 = q1μ ± q1σ
    return q1
end

function calc_ocv(kf; vlims=(3.7, 4.0))
    zt = kf.p.zt

    q̂min, q̂max = extrema(kf.p.r0.b0)
    q̂ = q̂min:0.01:q̂max
    q = StatsBase.reconstruct(zt.q, q̂)

    q0 = calc_ΔQ0(kf, vlims[1]) |> Measurements.value
    # Q = calc_Q(kf, f)
    Δq = calc_deltaq(kf; v=vlims) |> Measurements.value

    # OCV 
    ocv = predict_gp(kf, q̂, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)

    LinearInterpolation(
        ocvμ,
        q .- q0,
    )
end

let m = 1
    for c in 1:12
        kf = kfs[(; p=3, m, c)]
        ocv1 = calc_ocv(kf)
        lines(diff(ocv1.u)) |> display
    end
end


soh = [p[:soh] for (id, p) in params] .|> Measurements.value # |> maximum

# bins = 0.38:0.01:0.8
begin
    bins = 38:1:80
    fig, ax = hist(soh * 100; bins, color=:gray)
    ax.xlabel = "Cell SOH / %"
    ax.ylabel = "Cell count"
    ylims!(ax, 0, nothing)
    fig
end

f = let m = 9
    fs = [calc_ocv(kfs[(; p=3, m, c)]; v1=3.69) for c in 1:12]

    s0 = minimum(first(f.t) for f in fs)
    s1 = maximum(last(f.t) for f in fs)
    srange = range(s0, s1, length=1001)

    A = map(fs) do f
        int1 = Interval(first(f.t), last(f.t))
        [s ∈ int1 ? f(s) : missing for s in srange]
    end |> splat(hcat)

    ocv = minimum.(skipmissing.(eachrow(A)))

    soc = range(0.05, 0.95, length=1001)
    focv⁻¹ = LinearInterpolation(soc, ocv)
    focv = LinearInterpolation(ocv, soc)

    (; ocv=focv, ocv⁻¹=focv⁻¹)
end # No!


let
    fig = Figure()
    ax = Axis(fig[1, 1])
    s = 0:0.01:1
    s´ = range(0.15, 1.0, length=length(s))
    s´´ = range(0.0, 0.9, length=length(s))
    lines!(ax, s´´, focv(s) .- 0.1)
    lines!(ax, s´, f.ocv(s))
    fig
end