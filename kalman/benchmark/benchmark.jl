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
using StaticArrays
using Optim
using UnPack
using ForwardDiff
using PreallocationTools
includet("../battModel/rc.jl")
includet("../battModel/r0_ocv.jl")
includet("../battModel/batt.jl")
includet("../battModel/rgp.jl")
import ComponentArrays: ComponentVector, getaxes, ComponentMatrix, @static_unpack



#### BENCHMARK RGP ###
begin
    df_data = CSV.read("data/output_data_without_rc.csv", DataFrame)

    df_ocv = CSV.File("data/ocv.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc; extrapolation=ExtrapolationType.Constant)

    df = CSV.File("data/profile.csv") |> DataFrame
    fi = ConstantInterpolation(df.i, df.t)

    # data

    R0(soc) = (0.005 .+ 0.004 .* soc .^ 2 .- 0.006 .* soc) * 7
    df = DataFrame(
        t=df_data.time,
        v=df_data.voltage + fi.(df_data.time) .* (R0(df_data.soc) .- 15e-3),
        i=fi.(df_data.time),  # interpolated current
        soc=df_data.soc,
        q=df_data.soc * 4.8 * 3600
    )


    function fit_zscore(df)
        v = StatsBase.fit(ZScoreTransform, df.v)
        σ = StatsBase.fit(ZScoreTransform, df.v, center=false)
        i = StatsBase.fit(ZScoreTransform, df.i, center=false)
        soc = StatsBase.fit(ZScoreTransform, df.soc)
        q = StatsBase.fit(ZScoreTransform, df.q)
        return (; v, σ, i, soc, q)
    end

    function normalize_data(df, dt)
        v = df.v
        i = df.i
        soc = df.soc
        q = StatsBase.transform(dt.q, df.q)
        return DataFrame(; df.t, v, i, soc, q)
    end


    dt = fit_zscore(df)
    df = normalize_data(df, dt)

    N_points = 5000 #size(df_data, 1)
    data = DataLoader((
            u=(;
                b=df.soc[1:N_points],
                i=df.i[1:N_points]
            ),
            y=df.v[1:N_points]
        ),
        batchsize=1, shuffle=false
    )

end



begin
    l_ocv = 0.2
    σ_ocv = 0.8
    b0 = collect(0:0.01:1)  # basis vector for OCV
    b0 = StatsBase.transform(dt.soc, b0)

    gp = GP(
        ZeroMean(),
        σ_ocv * with_lengthscale(SEKernel(), l_ocv)
    )

    rgp = RGP(
        gp, b0;
        σ2=1e-5,
        tr=dt.v,
        tr_b=dt.soc)


    kf = ExtendedKalmanFilter(
        rgp.dynamics,
        rgp.measurement,
        rgp.R1,
        rgp.R2,
        rgp.d0;
        nx=rgp.nx,
        ny=rgp.ny,
        nu=1,
        rgp.p)



end


begin
    l_ocv = 0.2
    σ_ocv = 0.8
    b0 = collect(0:0.01:1)  # basis vector for OCV
    b0 = StatsBase.transform(dt.soc, b0)

    gp_ocv = GP(
        ZeroMean(),
        LinearKernel() + σ_ocv * with_lengthscale(SEKernel(), l_ocv)
    )

    ocv = OCV(
        gp_ocv, b0;
        σ2=1e-5,
        tr=dt.v,
        tr_b=dt.soc)


    gp_r0 = GP(ZeroMean(), σ_ocv * with_lengthscale(SEKernel(), l_ocv))
    r0 = R0(
        gp_r0, b0;
        σ2=1e-5,
        tr=dt.σ,
        tr_b=dt.soc)



    soc0 = 0.5
    Q_ = 4.8 * 3600
    soc = SOC(;
        Q=Q_,
        soc0=soc0,
        σ1=1e-9,
        Σ_soc=(soc0 - 0.5)^2 + 1e-6
    )

    r0_ocv = R0_OCV(ocv, r0, soc)


    components_batt = (;
        ocv=r0_ocv,
    )


    battModel = BattModel(components_batt)

end


begin
    kf = ExtendedKalmanFilter(
        battModel.dynamics,
        battModel.measurement,
        battModel.R1,
        battModel.R2,
        battModel.d0;
        nx=battModel.nx,
        ny=battModel.ny,
        nu=1,
        battModel.p
    )
end


@benchmark begin
    for (n, batch) in enumerate(data)
        kf(batch.u, batch.y)
    end
end


begin
    kf = ExtendedKalmanFilter(
        r0_ocv.dynamics,
        r0_ocv.measurement,
        r0_ocv.R1,
        r0_ocv.R2,
        r0_ocv.d0;
        nx=r0_ocv.nx,
        ny=r0_ocv.ny,
        nu=1,
        r0_ocv.p
    )
end


begin
    c = ComponentVector(
        c_ocv=[1, 2, 3, 5],
        c_r0=[5, 6, 7],
        c_soc=[8, 9, 10]
    )
    @unpack c_ocv, c_r0, c_soc = c

    c_ocv .= [4, 4, 4, 4]
    println(c)
end




@benchmark begin
    for (n, batch) in enumerate(data)
        kf(batch.u, batch.y)
    end
end





