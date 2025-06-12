using OrdinaryDiffEq
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

using DataFrames
using CSV
using DataInterpolations

using LowLevelParticleFilters
using StaticArrays
using LinearAlgebra
using Distributions

import ComponentArrays as CA
import ComponentArrays: ComponentArray, ComponentVector

using CairoMakie

## create data
@mtkmodel ECM begin
    @parameters begin
        Q = 4.8
        R0 = 25e-3
        R1 = 15e-3
        τ1 = 60.0
        # R2 = 10e-3
        # τ2 = 600.0
    end
    @structural_parameters begin
        focv
        fi
    end
    @variables begin
        i(t), [input = true]
        v(t), [output = true]
        vr(t)
        v1(t) = 0.0
        # v2(t) = 0.0
        ocv(t)
        soc(t)
    end
    @equations begin
        D(soc) ~ i / (Q * 3600.0)
        D(v1) ~ -v1 / τ1 + i * (R1 / τ1)
        # D(v2) ~ -v2 / τ2 + i * (R2 / τ2)
        vr ~ i * R0
        ocv ~ focv(soc)
        i ~ fi(t)
        v ~ ocv + vr + v1 #+ v2
    end
end

begin # read OCV look-up-table and current profile
    df_ocv = CSV.File("data/ocv.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc; extrapolation=ExtrapolationType.Constant)

    df = CSV.File("data/profile.csv") |> DataFrame
    fi = ConstantInterpolation(df.i, df.t)
end

begin # create model
    @mtkcompile ecm = ECM(; focv, fi)
    tspan = (0, 24 * 3600) # one day
    ode = ODEProblem(ecm, [ecm.soc => 0.5], tspan)
end

begin # create synthetic data
    Ts = 1.0 # time sampling
    sol = solve(ode, Tsit5(); saveat=Ts)
    i = sol[ecm.i]
    v = sol[ecm.v]
    s = sol[ecm.soc]

    # plot  
    # lines(sol.t / 3600, v; axis=(; xlabel="Time in h", ylabel="Voltage in V")) |> display
    # lines(sol.t / 3600, s; axis=(; xlabel="Time in h", ylabel="SOC in p.u.")) |> display

    # data
    df = DataFrame(; t=sol.t, i, v, s)
end


## Kalman Filter
function dynamics!(dx, x, u, p, t)
    (; Ts, q, R1, τ1, xid) = p
    soc = x[1]
    v1 = x[2]
    # dx = ComponentArray(dx, xid)
    # x = ComponentArray(x, xid)
    # (;soc, v1) = x
    i = u[1] # control

    dx[1] = soc + i * Ts / (q * 3600)
    dx[2] = v1 * exp(-Ts / τ1) + i * R1 * (1 - exp(-Ts / τ1))
    # dx.soc = soc + i * Ts / (q * 3600)
    # dx.v1 = v1 * exp(-Ts / τ1) + i * R1 * (1 - exp(-Ts / τ1))
end

function measurement(x, u, p, t)
    (; ocv, R0) = p # state
    soc = x[1]
    v1 = x[2] # state
    i = u[1] # control
    SA[ocv(soc)+i*R0+v1]
end


x0 = ComponentArray(soc=0.5, v1=0.0)
xid = CA.getaxes(x0)

# parameters
# Q = 4.8
# R0 = 25e-3
# R1 = 15e-3
# τ1 = 60.0
p = (;
    Ts=1.0,
    q=4.4,
    R0=15e-3,
    R1=15e-3,
    τ1=100,
    ocv=focv,
    xid
)

R1 = diagm([0.01, 0.01])
R2 = @SMatrix [1.0;;]
d0 = MvNormal(x0, I)
kf = ExtendedKalmanFilter(dynamics!, measurement, R1, R2, d0; nx=2, nu=1, ny=1, p)
# kf = UnscentedKalmanFilter(dynamics!, measurement, R1, R2, d0; nx=2, nu=1, ny=1, p)

us = [[i + 0.01randn()] for i in df.i]
ys = [[v + 0.01randn()] for v in df.v]
smoothsol = smooth(kf, us, ys)


s´ = smoothsol.xT .|> first

begin
    fig = Figure()
    ax1 = Axis(fig[1, 1], ylabel="SOC")
    lines!(ax1, df.t, df.s)
    lines!(ax1, df.t, s´)
    ax2 = Axis(fig[2, 1], ylabel="Error")
    lines!(ax2, df.t, df.s - s´)
    fig
end


