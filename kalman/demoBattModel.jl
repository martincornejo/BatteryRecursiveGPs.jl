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

import ComponentArrays: ComponentVector, getaxes, ComponentMatrix



### Julia style, only functions will be the user function that return all Kalman filter needed parameters

## Module of RGP
begin
    function RGP(gp, b0, σ=1)
        """
        Main function and only functoins user need to know
        """
        R1 = Diagonal(zero(b0))
        nx = length(b0)
        ny = 1

        p = generate_p_gp(gp, b0, σ)
        d0 = MvNormal(mean(gp, b0), cov(gp, b0) + 1e-6I)

        rgp = (;
            dynamics=dynamics_gp,
            measurement=measurement_gp,
            R1=R1,
            R2=R2fun_gp,
            d0=d0,
            nx=nx,
            ny=ny,
            p=p
        )

        return rgp
    end

    function generate_p_gp(gp, b0, σ)
        μ = mean(gp, b0)
        Σ = cov(gp, b0) + 1e-6I
        μ0 = μ
        Σ0⁻¹ = inv(Σ)

        p = (;
            gp=gp,     # gp (mean + kernel functions)
            b0=b0,      # basis vector
            μ0=μ0,     # ptir of basis vector
            Σ0⁻¹=Σ0⁻¹,
            σ=σ      # inv convariance basis vector
        )
        return p
    end


    function R2fun_gp(x, u, p, t)
        (; gp, b0, Σ0⁻¹, σ) = p
        b = u.b  ## Each submodule is the one of retrieving its control parameter
        H = cov(gp, b, b0) * Σ0⁻¹
        return σ.scale .^ 2 * (cov(gp, b) - H * cov(gp, b0, b))
    end

    function dynamics_gp(x, u, p, t)
        return x # identity
    end

    function measurement_gp(x, u, p, t)
        (; gp, b0, μ0, Σ0⁻¹, σ) = p

        g = x
        b = u.b ## Each submodule is the one of retrieving its control parameter

        H = cov(gp, b, b0) * Σ0⁻¹
        ## Denormalizing the measurement))
        return StatsBase.reconstruct(σ, mean(gp, b) + H * (g - μ0))
    end


end

## Module of RC
## NOTE: FOR SIMPLICITY RC PARAMETERS ARE NOT ADDED TO THE KF; BUT THEY CAN BE EASILY ADDED
begin
    function RC(ts, τ0, R0; Vrc0=0.0, σ1=sqrt(1e-2), σ2=sqrt(1e-3))
        """
        Main function and only functions user need to know
        """


        R1 = [
            1e-2 0 0
            0 2e-6 0
            0 0 3e-11
        ]
        R2(x, u, p, t) = Diagonal(fill(σ2^2, 1))
        nx = 1
        ny = 1

        x0 = ComponentVector(
            Vrc=Vrc0,
            τ=τ0,
            R=R0
        )

        Σ0 = false .* x0 * x0'
        Σ0[:Vrc, :Vrc] = σ1^2
        Σ0[:τ, :τ] = (60 - τ0)^2
        Σ0[:R, :R] = (15e-3 - R0)^2
        Σ0 = Σ0 + 1e-6 * I

        d0 = MvNormal(x0, Σ0)


        xid = getaxes(x0)
        Σid = getaxes(Σ0)
        p = generate_p_rc(ts, xid, Σid)

        rc = (;
            dynamics=dynamics_rc,
            measurement=measurement_rc,
            R1=R1,
            R2=R2,
            d0=d0,
            nx=nx,
            ny=ny,
            p=p
        )

        return rc
    end

    function generate_p_rc(ts, xid, Σid, i=[0.0])
        p = ComponentVector(;
            ts,
            xid,
            Σid,
            i
        )
        return p
    end


    function dynamics_rc(x, u, p, t)
        (; ts, xid) = p
        c = ComponentVector(x, xid)
        i = u.i[1]
        c.Vrc = exp(-ts / c.τ) * c.Vrc + i * c.R * (1 - exp(-ts / c.τ))
        c.τ = c.τ
        c.R = c.R


        return c
    end

    function measurement_rc(x, u, p, t)
        (; xid) = p
        c = ComponentVector(x, xid)
        Vrc = c.Vrc
        return Vrc
    end

end






### Module of BattModel
begin


    function BattModel(components, model)
        """
        Generates the Kalman filter model for the battery
        components_batt: a tuple with the components of the battery
        batt_function: a function that will be used to generate the model-function
        currently not made optimal/a lot of hard coded functions
        """


        component_names = keys(components)
        x0 = ComponentVector(; (name => mean(components[name].d0) for name in component_names)...)

        Σ0 = false .* x0 * x0'
        R1 = false .* x0 * x0'
        for name in component_names
            component = components[name]
            Σ0[name, name] = cov(component.d0)
            R1[name, name] = component.R1
        end

        d0 = MvNormal(x0, Σ0)
        xid = getaxes(x0)
        Σid = getaxes(Σ0)

        p = generate_p_batt(components, model, xid, Σid)


        battModel = (;
            dynamics=dynamics_batt,
            measurement=measurement_batt,
            R1=R1,
            R2=R2_batt_fun,
            d0=d0,
            nx=length(x0),
            ny=1,
            p=p
        )

        return battModel
    end



    function generate_p_batt(components, model, xid, Σid)
        """
        Can be done automatic and easier
        """
        p = (;
            xid=xid,  # axes for state vector
            Σid=Σid,
            model=model,  # axes for covariance matrix
            components=components,  # components of the battery
        )

        return p

    end


    function R2_batt_fun(x, u, p, t)
        (; xid, components, model) = p
        c = ComponentVector(x, xid)
        model_coeff = model(c, u, p, t)

        #### HARD CODED ONE FOR NOW
        R2 = zeros(1)

        for name in keys(components)
            component = components[name]
            R2 += (model_coeff[name] .^ 2) .* component.R2(c[name], u, component.p, t)
        end

        return R2
    end


    function dynamics_batt(x, u, p, t)
        """
        Calling measurement on all components
        This functions should be automatic
        """
        (; xid, components) = p
        c = ComponentVector(x, xid)
        for name in keys(components)
            component = components[name]
            c[name] = component.dynamics(c[name], u, component.p, t)
        end
        return c
    end

    function measurement_batt(x, u, p, t)
        """
        This function should be generated automatically given a model_function
        """
        (; xid, components, model) = p
        c = ComponentVector(x, xid)
        model_coeff = model(c, u, p, t)
        v = [0.0]
        for name in keys(components)
            component = components[name]
            coeff = model_coeff[name]
            v = v .+ coeff .* component.measurement(c[name], u, component.p, t)
        end

        return v
    end


end



######################### Example of usage ##########################

## Loading data


begin
    df_data = CSV.read("data/output_data_with_one_rc.csv", DataFrame)

    df_ocv = CSV.File("data/ocv.csv") |> DataFrame
    focv = LinearInterpolation(df_ocv.ocv, df_ocv.soc; extrapolation=ExtrapolationType.Constant)

    df = CSV.File("data/profile.csv") |> DataFrame
    fi = ConstantInterpolation(df.i, df.t)

    # data
    df = DataFrame(
        t=df_data.time,
        v=df_data.voltage,
        i=fi.(df_data.time),  # interpolated current
        soc=df_data.soc
    )


    function fit_zscore(df)
        v = StatsBase.fit(ZScoreTransform, df.v)
        σ = StatsBase.fit(ZScoreTransform, df.v, center=false)
        i = StatsBase.fit(ZScoreTransform, df.i, center=false)
        soc = StatsBase.fit(ZScoreTransform, df.soc)
        return (; v, σ, i, soc)
    end

    function normalize_data(df, dt)
        v = df.v
        i = df.i
        soc = StatsBase.transform(dt.soc, df.soc)
        return DataFrame(; df.t, v, i, soc)
    end


    dt = fit_zscore(df)
    df = normalize_data(df, dt)

    N_points = size(df, 1)
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



## Generating model
begin
    l_ocv = 0.2
    σ_ocv = 0.8
    b0 = collect(0:0.01:1)  # basis vector for OCV
    b0 = StatsBase.transform(dt.soc, b0)

    focv_mean = LinearInterpolation(
        StatsBase.transform(dt.v, df_ocv.ocv[1:100:end]),
        StatsBase.transform(dt.soc, df_ocv.soc[1:100:end]);
        extrapolation=ExtrapolationType.Linear)

    ocv_mean = x -> focv_mean(x)
    gp_ocv = GP(ZeroMean(), LinearKernel() + σ_ocv * with_lengthscale(SEKernel(), l_ocv))
    ocv = RGP(gp_ocv, b0, dt.v)

    ocv = (;
        dynamics=ocv.dynamics,
        measurement=ocv.measurement,
        R1=ocv.R1,
        R2=ocv.R2,
        d0=MvNormal(ocv_mean.(b0), cov(ocv.d0)),
        nx=ocv.nx,
        ny=ocv.ny,
        p=ocv.p
    )


    l_r = 1.2
    σ_r = 0.5
    r0_mean = x -> StatsBase.transform(dt.σ, [15e-3])[1]
    gp_r0 = GP(ZeroMean(), σ_r * with_lengthscale(SEKernel(), l_r))

    r0 = RGP(gp_r0, b0, dt.σ)

    r0 = (;
        dynamics=r0.dynamics,
        measurement=r0.measurement,
        R1=r0.R1,
        R2=r0.R2,
        d0=MvNormal(r0_mean.(b0), cov(r0.d0)),
        nx=r0.nx,
        ny=r0.ny,
        p=r0.p
    )

    rc1 = RC(1, 40, 15e-3)
    rc2 = RC(1, 120, 30e-3)

    components_batt = (;
        ocv=ocv,
        r0=r0,
        rc1=rc1,
    )

    model(x, u, p, t) = ComponentVector(;
        ocv=1,
        r0=u.i,
        rc1=1,
    )

    battModel = BattModel(components_batt, model)
end




# Training
# ExtendedKalmanFilter for testing purposes since is Faster
@time begin
    rc_values = []
    kf = ExtendedKalmanFilter(
        battModel.dynamics,
        battModel.measurement,
        battModel.R1,
        battModel.R2,
        battModel.d0;
        nx=battModel.nx,
        ny=battModel.ny,
        nu=3,
        battModel.p
    )

    for (n, batch) in enumerate(data)
        kf(batch.u, batch.y)
        #predict!(kf, batch.u)
        #correct!(kf, batch.u, batch.y)



        x_rc1 = ComponentVector(kf.x, battModel.p.xid)[:rc1]
        push!(rc_values, (; t=batch.u, rc1_values=copy(x_rc1)))
    end


end



begin
    V = df.v[1:N_points]
    x = ComponentVector(kf.x, battModel.p.xid)
    Σx = ComponentMatrix(kf.R, battModel.p.Σid)
    u = (b=df.soc, i=df.i)
    V_ocv = measurement_gp(x[:ocv], u, ocv.p, 0)
    V_r0 = df.i .* measurement_gp(x[:r0], u, r0.p, 0)
    V_rc1 = [entry.rc1_values.Vrc for entry in rc_values]


    V_aprox = V_ocv + V_r0 + V_rc1


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
    ylims!(ax2, 0.0, 0.015)
    axislegend(ax2)
    display(fig)
end


## Plotting
begin
    x = ComponentVector(kf.x, battModel.p.xid)
    Σx = ComponentMatrix(kf.R, battModel.p.Σid)
    fig = Figure(size=(1200, 1200))

    bp = StatsBase.transform(dt.soc, collect(0.0:0.015:1))
    up = (
        b=bp, i=df.i
    )

    ocv_values = ocv.measurement(x[:ocv], up, ocv.p, 0)
    # OCV curve
    ax1 = CairoMakie.Axis(fig[1, 1], title="OCV curve", xlabel="SOC", ylabel="V")
    lines!(ax1, StatsBase.reconstruct(dt.soc, b0), StatsBase.reconstruct(dt.v, x[:ocv]), label="OCV aprox")
    band!(ax1, StatsBase.reconstruct(dt.soc, b0), StatsBase.reconstruct(dt.v, x[:ocv]) - 2sqrt.(diag(dt.v.scale .^ 2 .* Σx[:ocv])), StatsBase.reconstruct(dt.v, x[:ocv]) + 2sqrt.(diag(dt.v.scale .^ 2 .* Σx[:ocv])), ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))
    lines!(ax1, collect(0:0.01:1), focv(collect(0:0.01:1)), label="OCV real")
    ylims!(ax1, minimum(focv(collect(0:0.01:1))), maximum(focv(collect(0:0.01:1))))
    axislegend(ax1)


    ax2 = CairoMakie.Axis(fig[2, 1], title="R0 curve", xlabel="SOC", ylabel="V")
    lines!(ax2, StatsBase.reconstruct(dt.soc, b0), StatsBase.reconstruct(dt.σ, x[:r0]), label="R0 aprox")
    band!(ax2, StatsBase.reconstruct(dt.soc, b0), StatsBase.reconstruct(dt.σ, x[:r0]) - 2sqrt.(diag(dt.σ.scale .^ 2 .* Σx[:r0])), StatsBase.reconstruct(dt.σ, x[:r0]) + 2sqrt.(diag(dt.σ.scale .^ 2 .* Σx[:r0])), ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))

    hlines!(ax2, 15e-3, label="R0 real")
    ylims!(ax2, 10e-3, 20e-3)
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

    ax1 = Axis(f[1, 1], title="Vrc over Time", xlabel="Time", ylabel="Vrc [V]")
    lines!(ax1, df_rc.Vrc)

    ax2 = Axis(f[2, 1], title="τ over Time", xlabel="Time", ylabel="τ [s]")
    lines!(ax2, df_rc.tau)

    ax3 = Axis(f[3, 1], title="R over Time", xlabel="Time", ylabel="R [Ω]")
    lines!(ax3, df_rc.R)
    #ylims!(ax3, 10e-3, 20e-3)

    display(f)
end





