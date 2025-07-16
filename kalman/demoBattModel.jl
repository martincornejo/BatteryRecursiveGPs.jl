using LowLevelParticleFilters
using Distributions
using LinearAlgebra
using MLUtils: DataLoader
using DataFrames
using CSV
using AbstractGPs
using DataInterpolations
using CairoMakie
using StatsBase
using BenchmarkTools
using Revise
using Optim
includet("battModel/rc.jl")
includet("battModel/r0_ocv.jl")
includet("battModel/batt.jl")
includet("battModel/rgp.jl")
import ComponentArrays: ComponentVector, getaxes, ComponentMatrix




######################### Example of usage ##########################

## Loading data

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
        soc = StatsBase.transform(dt.soc, df.soc)
        q = StatsBase.transform(dt.q, df.q)
        return DataFrame(; df.t, v, i, soc, q)
    end


    dt = fit_zscore(df)
    df = normalize_data(df, dt)

    N_points = size(df_data, 1)
    data = DataLoader((
            u=(;
                i=df.i[1:N_points]
            ),
            y=df.v[1:N_points]
        ),
        batchsize=1, shuffle=false
    )

end


## Generating model

begin

    l_ocv = 0.2
    σ_ocv = 0.8
    b0 = collect(0:0.01:1)  # basis vector for OCV
    b0 = StatsBase.transform(dt.soc, b0)

    gp_ocv = GP(ZeroMean(), LinearKernel() + σ_ocv * with_lengthscale(SEKernel(), l_ocv))
    ocv = OCV(
        gp_ocv, b0;
        σ=1e-5,
        tr=dt.v)


    l_r = 0.2
    σ_r = 0.8
    gp_r0 = GP(ZeroMean(), σ_r * with_lengthscale(SEKernel(), l_r))

    r0 = R0(gp_r0, b0, σ=1e-10, tr=dt.σ)

    soc0 = 0.5
    Q_ = 4.8 * 3600
    soc = SOC(;
        Q=Q_,
        soc0=soc0,
        σ1=1e-9,
        Σ_soc=(soc0 - 0.5)^2 + 1e-6
    )

    r0_ocv = R0_OCV(ocv, r0, soc, dt.soc)


    components_batt = (;
        ocv=r0_ocv,
    )


    battModel = BattModel(components_batt)
end



begin
    if true
        f = Figure(size=(800, 600))

        x = ComponentVector(mean(battModel.d0), battModel.p.xid)[:ocv]
        Σx = ComponentMatrix(cov(battModel.d0), battModel.p.Σid)[:ocv]

        bp = StatsBase.transform(dt.soc, collect(0.0:0.01:1))[1:end-1]
        up = (
            b=bp, i=df.i
        )

        # OCV curve
        ax1 = CairoMakie.Axis(f[1, 1], title="OCV curve", xlabel="SOC", ylabel="V")
        lines!(ax1, StatsBase.reconstruct(dt.soc, bp), StatsBase.reconstruct(dt.v, x[:ocv][1:end-1]), label="OCV aprox")
        band!(
            ax1, StatsBase.reconstruct(dt.soc, bp), StatsBase.reconstruct(dt.v, x[:ocv][1:end-1]) - 2sqrt.(diag(dt.v.scale .^ 2 .* Σx[:ocv][1:end-1, 1:end-1])),
            StatsBase.reconstruct(dt.v, x[:ocv][1:end-1]) + 2sqrt.(diag(dt.v.scale .^ 2 .* Σx[:ocv][1:end-1, 1:end-1])),
            ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))
        lines!(ax1, collect(0:0.01:1), focv(collect(0:0.01:1)), label="OCV real")
        ylims!(ax1, minimum(focv(collect(0:0.01:1))), maximum(focv(collect(0:0.01:1))))
        axislegend(ax1)


        # R0 curve
        ax2 = CairoMakie.Axis(f[2, 1], title="R0 curve", xlabel="SOC", ylabel="mOhm")
        lines!(ax2, StatsBase.reconstruct(dt.soc, bp), StatsBase.reconstruct(dt.σ, x[:r0][1:end-1]), label="R0 aprox")
        band!(
            ax2, StatsBase.reconstruct(dt.soc, bp), StatsBase.reconstruct(dt.σ, x[:r0][1:end-1]) - 2sqrt.(diag(dt.σ.scale .^ 2 .* Σx[:r0][1:end-1, 1:end-1])),
            StatsBase.reconstruct(dt.σ, x[:r0][1:end-1]) + 2sqrt.(diag(dt.σ.scale .^ 2 .* Σx[:r0][1:end-1, 1:end-1])),
            ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))


        soc = collect(0.0:0.01:1)
        lines!(ax2, soc, R0(soc), label="R0 real")
        ylims!(-2e-3 * 7, 6e-3 * 7)
        axislegend(ax2)
        display(f)
    end
end


# Training
# ExtendedKalmanFilter for testing purposes since is Faster
begin

    soc_values = []
    soc_sigmas = []
    rc_values = []
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

    for (n, batch) in enumerate(data)
        kf(batch.u, batch.y)

        x_soc = ComponentVector(kf.x, battModel.p.xid)[:ocv][end]
        Σ_soc = 2sqrt.(ComponentMatrix(kf.R, battModel.p.Σid)[:ocv][end, end])
        push!(soc_values, x_soc)
        push!(soc_sigmas, Σ_soc)
        #x_rc = ComponentVector(kf.x, battModel.p.xid)[:rc1]
        #push!(rc_values, (; t=batch.u, rc1_values=copy(ComponentVector(kf.x, battModel.p.xid)[:rc1])))
    end
end


"""
begin

    x = ComponentVector(kf.x, battModel.p.xid)[:ocv]
    Σx = ComponentMatrix(kf.R, battModel.p.Σid)[:ocv, :ocv]

    maxSoc = maximum(StatsBase.transform(dt.soc, Float64.(soc_values)))
    minSoc = minimum(StatsBase.transform(dt.soc, Float64.(soc_values)))


    maxSocIdx = argmin(abs.(b0 .- maxSoc)) - 1
    minSocIdx = argmin(abs.(b0 .- minSoc)) + 1

    mean_ocv = LinearInterpolation(x[:ocv][minSocIdx:maxSocIdx], b0[minSocIdx:maxSocIdx]; extrapolation=ExtrapolationType.Linear)
    mean_r0 = LinearInterpolation(x[:r0][minSocIdx:maxSocIdx], b0[minSocIdx:maxSocIdx]; extrapolation=ExtrapolationType.Constant)


    ocv = (;
        dynamics=ocv.dynamics,
        measurement=ocv.measurement,
        R1=ocv.R1,
        R2=ocv.R2,
        d0=MvNormal(mean_ocv.(b0), cov(ocv.d0)),
        nx=ocv.nx,
        ny=ocv.ny,
        p=ocv.p
    )

    r0 = (;
        dynamics=r0.dynamics,
        measurement=r0.measurement,
        R1=r0.R1,
        R2=r0.R2,
        d0=MvNormal(mean_r0.(b0), cov(r0.d0)),
        nx=r0.nx,
        ny=r0.ny,
        p=r0.p
    )

    soc0 = soc0
    Q_ = Q_
    soc = SOC(;
        Q=Q_,
        soc0=soc0,
        σ1=0.0,
        Σ_soc=(soc0 - 0.5)^2 + 1e-6
    )


    rc1 = RC(
        1, 30, 50e-3,
        Vrc_σ=1e-3,
        σ1=[1e-3, 2e-3, 3e-11],
        σ2=sqrt(1e-3),
        τh=90,
        Rh=10e-3)
    r0_ocv = R0_OCV(ocv, r0, soc, dt.soc)


    components_batt = (;
        ocv=r0_ocv,
        rc1=rc1
    )


    battModel = BattModel(components_batt)

end




begin
    soc_values = []
    soc_sigmas = []
    rc_values = []
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
    for (n, batch) in enumerate(data)
        kf(batch.u, batch.y)
        x_soc = ComponentVector(kf.x, battModel.p.xid)[:ocv][end]
        Σ_soc = 2sqrt.(ComponentMatrix(kf.R, battModel.p.Σid)[:ocv][end, end])
        push!(soc_values, x_soc)
        push!(soc_sigmas, Σ_soc)

        x_rc = ComponentVector(kf.x, battModel.p.xid)
        push!(rc_values, (; t=batch.u, rc1_values=copy(ComponentVector(kf.x, battModel.p.xid)[:rc1])))

    end
end
"""






begin
    f = Figure(size=(800, 600))
    N_points = size(df.soc, 1)
    ax1 = CairoMakie.Axis(f[1, 1], title="SOC curve", xlabel="time", ylabel="soc")
    lines!(ax1, soc_values, label="SOC aprox")
    lines!(ax1, df_data.soc[1:N_points], label="SOC real")

    band!(
        ax1, collect(1:1:N_points), soc_values - soc_sigmas, soc_values + soc_sigmas,
        ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3)
    )

    axislegend(ax1)
    display(f)
end

begin
    x = ComponentVector(kf.x, battModel.p.xid)
    # Loss function: mean squared error
    function loss(soc_real)
        fix_soc = (;
            soc=0.5,
            ocv=3.73
        )
        u = (;
            b=StatsBase.transform(dt.soc, [soc_real])
        )

        ocv_aprox = measurement_gp(x[:ocv][:ocv], u, ocv.p, 0)[1]

        return abs(ocv_aprox - fix_soc.ocv)
    end

    # Optimization
    result = Optim.optimize(loss, 0.0, 1.0, Brent())
    soc_right = Optim.minimizer(result)
    soc_shift = 0.5 - soc_right

    println("shift: $(soc_shift)")
end


begin
    x = ComponentVector(kf.x, battModel.p.xid)
    # Loss function: mean squared error
    function loss(shift)
        u = (;
            b=StatsBase.transform(
                dt.soc, [soc_values .+ shift])
        )

        ocv_aprox = measurement_gp(x[:ocv][:ocv], u, ocv.p, 0)[1]
        r0_

        return abs(ocv_aprox - fix_soc.ocv)
    end

    # Optimization
    result = Optim.optimize(loss, 0.0, 1.0, Brent())
    soc_right = Optim.minimizer(result)
    soc_shift = 0.5 - soc_right

    println("shift: $(soc_shift)")
end





begin
    soc_shift = 0.0
    f = Figure(size=(800, 600))

    x = ComponentVector(kf.x, battModel.p.xid)[:ocv]
    Σx = ComponentMatrix(kf.R, battModel.p.Σid)[:ocv, :ocv]

    bp = StatsBase.transform(dt.soc, collect(0.0:0.01:1))
    up = (
        b=bp, i=df.i
    )

    name = :ocv

    # OCV curve
    ax1 = CairoMakie.Axis(f[1, 1], title="OCV curve", xlabel="SOC", ylabel="V")
    lines!(ax1, StatsBase.reconstruct(dt.soc, bp), StatsBase.reconstruct(dt.v, x[name]), label="OCV aprox")
    band!(
        ax1, StatsBase.reconstruct(dt.soc, bp), StatsBase.reconstruct(dt.v, x[name]) - 2sqrt.(diag(dt.v.scale .^ 2 .* Σx[name])),
        StatsBase.reconstruct(dt.v, x[name]) + 2sqrt.(diag(dt.v.scale .^ 2 .* Σx[name])),
        ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))
    lines!(ax1, collect(0:0.01:1), focv(collect(0:0.01:1)), label="OCV real")
    #ylims!(ax1, minimum(focv(collect(0:0.01:1))), maximum(focv(collect(0:0.01:1))))
    axislegend(ax1)


    # R0 curve
    name2 = :r0
    ax2 = CairoMakie.Axis(f[2, 1], title="R0 curve", xlabel="SOC", ylabel="mOhm")

    lines!(ax2, StatsBase.reconstruct(dt.soc, bp), StatsBase.reconstruct(dt.σ, x[name2]), label="R0 aprox")
    band!(
        ax2, StatsBase.reconstruct(dt.soc, bp), StatsBase.reconstruct(dt.σ, x[name2]) - 2sqrt.(diag(dt.σ.scale .^ 2 .* Σx[name2])),
        StatsBase.reconstruct(dt.σ, x[name2]) + 2sqrt.(diag(dt.σ.scale .^ 2 .* Σx[name2])),
        ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))


    soc = collect(0.0:0.01:1)
    lines!(ax2, soc, R0(soc), label="R0 real")
    ylims!(ax2, 0.0, 10e-3 * 7)
    axislegend(ax2)

    display(f)

end



begin
    V = df.v[1:N_points]
    x = ComponentVector(kf.x, battModel.p.xid)
    Σx = ComponentMatrix(kf.R, battModel.p.Σid)
    u = (
        b=StatsBase.transform(dt.soc, Float64.(soc_values)),
        i=df.i[1:N_points]
    )


    V_ocv = measurement_gp(x[:ocv][:ocv], u, ocv.p, 0)
    V_r0 = df.i[1:N_points] .* measurement_gp(x[:ocv][:r0], u, r0.p, 0)
    V_rc1 = [entry.rc1_values.Vrc for entry in rc_values]


    V_aprox = V_ocv + V_r0 #+ V_rc1


    fig = Figure(size=(1200, 1200))
    # OCV curve
    ax1 = CairoMakie.Axis(fig[1, 1], title="V", xlabel="t", ylabel="V")
    lines!(ax1, V, label="V real")
    lines!(ax1, V_aprox, label="V aprox", linestyle=:dash)
    axislegend(ax1)


    # Calculate absolute errors
    abs_errors = abs.(V - V_aprox)

    # Group into chunks of 100 and calculate MAE for each chunk
    chunk_size = 100
    n_chunks = div(length(abs_errors), chunk_size)

    # Calculate MAE for each chunk
    mae_values = [mean(abs_errors[(i-1)*chunk_size+1:i*chunk_size]) for i in 1:n_chunks]

    # Create time points for plotting (center of each 10-step interval)
    t_mae = [(i - 1) * chunk_size + chunk_size / 2 for i in 1:n_chunks]

    # Plot MAE every 10 steps
    ax2 = CairoMakie.Axis(fig[2, 1], title="V error", xlabel="t", ylabel="MAE ($chunk_size steps)")
    lines!(ax2, t_mae, mae_values, label="V MAE", color=:red)
    ylims!(ax2, 0.0, 0.01)
    axislegend(ax2)
    display(fig)
end



begin
    df_rc = DataFrame(
        Vrc=[entry.rc1_values.Vrc for entry in rc_values],
        tau=[entry.rc1_values.τ for entry in rc_values],
        R=[entry.rc1_values.R for entry in rc_values]
    )
    f = Figure(size=(800, 600))

    ax1 = Axis(f[1, 1], title="Vrc", xlabel="Time", ylabel="Vrc [V]")
    lines!(ax1, df_rc.Vrc, label="V real", color=:red)

    ax2 = Axis(f[2, 1], title="τ over Time", xlabel="Time", ylabel="τ [s]")
    lines!(ax2, df_rc.tau)
    hlines!(ax2, 60, label="τ real", color=:red)
    ylims!(ax2, 40, 80)

    ax3 = Axis(f[3, 1], title="R over Time", xlabel="Time", ylabel="R [Ω]")
    lines!(ax3, df_rc.R)
    hlines!(ax3, 15e-3, label="R real", color=:red)
    ylims!(ax3, 10e-3, 20e-3)

    display(f)
end












