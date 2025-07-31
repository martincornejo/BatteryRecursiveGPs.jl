using OrdinaryDiffEq
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

using DataFrames
using CSV
using DataInterpolations

using LinearAlgebra

using CairoMakie

@mtkmodel ECM begin
    @parameters begin
        Q = 4.8
        R1 = 15e-3
        τ1 = 60.0
        R2 = 15e-3
        τ2 = 600.0
    end
    @structural_parameters begin
        focv
        fi
    end
    @variables begin
        i(t), [input = true]
        w(t), [input = true]
        v(t), [output = true]
        vr(t)
        v1(t) = 0.0
        v2(t) = 0.0
        ocv(t)
        soc(t)

    end
    @equations begin
        D(soc) ~ i / (Q * 3600.0) + w
        D(v1) ~ -v1 / τ1 + i * (R1 / τ1)
        D(v2) ~ -v2 / τ2 + i * (R2 / τ2)
        R0(soc) ~ 0.005 + 0.004 * soc^2 - 0.006 * soc
        vr ~ i * R0
        ocv ~ focv(soc)
        i ~ fi(t)
        v ~ ocv + vr + v1
    end
end

begin # read OCV look-up-table and current profile
    df_ocv = CSV.File("data/ocv.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc, extrapolation=ExtrapolationType.Constant)

    df = CSV.File("data/profile.csv") |> DataFrame
    fi = ConstantInterpolation(df.i, df.t)
end

begin # create model
    @mtkbuild ecm = ECM(; focv, fi)
    tspan = (0, 24 * 3600) # one day
    ode = ODEProblem{false}(ecm, [ecm.soc => 0.5], tspan, [])
end

begin # create synthetic data
    Ts = 1.0 # time sampling
    w_fun(t) = 1e-3 * randn()
    sol = solve(ode, Tsit5();
        saveat=Ts,
        inputs=Dict(ecm.w => w_fun)
    )

    v = sol[ecm.v]
    s = sol[ecm.soc]

    # plot
    lines(sol.t / 3600, v; axis=(; xlabel="Time in h", ylabel="Voltage in V")) |> display
    lines(sol.t / 3600, s; axis=(; xlabel="Time in h", ylabel="SOC in p.u.")) |> display
end


begin # save data to CSV
    df_out = DataFrame(
        time_s=sol.t,
        voltage_V=sol[ecm.v],
        soc=sol[ecm.soc],
    )

    CSV.write("output/simulated_data.csv", df_out)
end


