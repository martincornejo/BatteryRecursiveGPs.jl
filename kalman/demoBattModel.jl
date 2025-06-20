using LowLevelParticleFilters
using Distributions
using LinearAlgebra
using MLUtils: DataLoader
using DataFrames
using CSV
using AbstractGPs
using DataInterpolations
using CairoMakie

import ComponentArrays: ComponentVector, getaxes, ComponentMatrix



### Julia style, only functions and only generate_X will be the user function that return all Kalman filter needed parameters

## Module of RGP
# NOTE:FOR NORMALIZED RGP THE EQUATIONS CHANGE; FOR SIMPLICITY NOT IMPLEMENTED BUT CAN BE EASILY CHANGED
begin
    function generate_RGP(gp, b0)
        """
        Main function and only functoins user need to know
        """
        R1 = Diagonal(zero(b0))
        nx = length(b0)
        ny = 1

        p = generate_p_gp(gp, b0)
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

    function generate_p_gp(gp, b0)
        μ = mean(gp, b0)
        Σ = cov(gp, b0) + 1e-6I
        μ0 = μ
        Σ0⁻¹ = inv(Σ)

        p = (;
            gp=gp,     # gp (mean + kernel functions)
            b0=b0,      # basis vector
            μ0=μ0,     # ptir of basis vector
            Σ0⁻¹=Σ0⁻¹,   # inv convariance basis vector
        )
        return p
    end


    function R2fun_gp(x, u, p, t)
        (; gp, b0, Σ0⁻¹) = p
        b = u.b  ## Each submodule is the one of retrieving its control parameter
        H = cov(gp, b, b0) * Σ0⁻¹
        return cov(gp, b) - H * cov(gp, b0, b)
    end

    function dynamics_gp(x, u, p, t)
        return x # identity
    end

    function measurement_gp(x, u, p, t)
        (; gp, b0, μ0, Σ0⁻¹) = p

        g = x
        b = u.b ## Each submodule is the one of retrieving its control parameter

        H = cov(gp, b, b0) * Σ0⁻¹
        mean(gp, b) + H * (g - μ0)
    end
end

## Module of RC
## NOTE: FOR SIMPLICITY RC PARAMETERS ARE NOT ADDED TO THE KF; BUT THEY CAN BE EASILY ADDED
begin
    function generate_rc(ts, τ, R; σ1=1e-2, σ2=1e-2, d0=MvNormal([0], 1e-4 * I(1)))
        """
        Main function and only functions user need to know
        """
        R1 = Diagonal(fill(σ1^2, 1))
        R2 = Diagonal(fill(σ2^2, 1))
        nx = 1
        ny = 1

        p = generate_p_rc(ts, τ, R)

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

    function generate_p_rc(ts, τ, R)
        p = (;
            ts,
            τ,
            R
        )
        return p
    end


    function dynamics_rc(x, u, p, t)
        (; ts, τ, R) = p
        i = u.i ## Each submodule is the one of retrieving its control parameter
        return exp(-ts / τ) * x + i * R * (1 - exp(-ts / τ))

    end

    function measurement_rc(x, u, p, t)
        return x
    end

end






### Module of BattModel
begin

    function generate_BattModel(components_batt, batt_function=nothing)
        """
        Generates the Kalman filter model for the battery
        components_batt: a tuple with the components of the battery
        batt_function: a function that will be used to generate the model-function
        currently not made optimal/a lot of hard coded functions
        """
        ## d0
        x0 = ComponentVector(;
            ocv=mean(components_batt.ocv.d0),
            r0=mean(components_batt.r0.d0),
            rc=mean(components_batt.rc.d0)
        )
        Σ0 = false .* x0 * x0'
        Σ0[:ocv, :ocv] = cov(components_batt.ocv.d0)
        Σ0[:r0, :r0] = cov(components_batt.r0.d0)
        Σ0[:rc, :rc] = cov(components_batt.rc.d0)

        d0 = MvNormal(x0, Σ0)
        xid = getaxes(x0)
        Σid = getaxes(Σ0)

        ## R1
        R1 = false .* x0 * x0'
        R1[:ocv, :ocv] = components_batt.ocv.R1
        R1[:r0, :r0] = components_batt.r0.R1
        R1[:rc, :rc] = components_batt.rc.R1

        p = generate_p_batt(components_batt, xid, Σid)


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



    function generate_p_batt(components_batt, xid, Σid)
        """
        Can be done automatic and easier
        """

        p = (;
            ocv=components_batt.ocv,
            r0=components_batt.r0,
            rc=components_batt.rc,
            xid=xid,
            Σid=Σid
        )

        return p

    end


    function R2_batt_fun(x, u, p, t)
        (; xid, ocv, r0, rc) = p
        c = ComponentVector(x, xid)

        ocvR2 = ocv.R2(c.ocv, u, ocv.p, t)
        r0R2 = r0.R2(c.r0, u, r0.p, t)
        rcR2 = rc.R2

        R2 = @. ocvR2 + u[2] * r0R2 + rcR2

        return R2
    end


    function dynamics_batt(x, u, p, t)
        """
        Calling measurement on all components
        This functions should be automatic
        """
        (; xid, ocv, r0, rc) = p
        c = ComponentVector(x, xid)
        c.ocv = ocv.dynamics(c.ocv, u, ocv.p, t)
        c.r0 = r0.dynamics(c.r0, u, r0.p, t)
        c.rc = rc.dynamics(c.rc, u, rc.p, t)
        x = c

    end

    function measurement_batt(x, u, p, t)
        """
        This function should be generated automatically given a model_function
        """
        (; xid, ocv, r0, rc) = p
        c = ComponentVector(x, xid)

        ocv = ocv.measurement(c.ocv, u, ocv.p, t)
        r0 = r0.measurement(c.r0, u, r0.p, t)
        rc = rc.measurement(c.rc, u, rc.p, t)

        x = @. ocv + u.i .* r0 + rc ## This should be a parameter
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
    σ_ocv = 0.1
    b0 = collect(0:0.01:1)  # basis vector for OCV
    gp_ocv = GP(ZeroMean(), LinearKernel() + σ_ocv * with_lengthscale(SEKernel(), l_ocv))

    l_r = 0.2
    σ_r = 0.1
    r0_mean = x -> 15e-3
    r0_kernel = σ_r * with_lengthscale(SEKernel(), l_r)
    gp_r0 = GP(r0_mean, r0_kernel)

    components_batt = (;
        ocv=generate_RGP(gp_ocv, b0),
        r0=generate_RGP(gp_r0, b0),
        rc=generate_rc(1, 60, 15e-3)
    )

    battModel = generate_BattModel(components_batt)
end

# Training
# ExtendedKalmanFilter for testing purposes since is Faster
begin
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
    end


end


## Plotting
begin
    x = ComponentVector(kf.x, battModel.p.xid)
    Σx = ComponentMatrix(kf.R, battModel.p.Σid)
    fig = Figure(size=(1200, 1200))
    # OCV curve
    ax1 = CairoMakie.Axis(fig[1, 1], title="OCV curve", xlabel="SOC", ylabel="V")
    lines!(ax1, b0, x[:ocv], label="OCV aprox")
    band!(ax1, b0, x[:ocv] - 2sqrt.(diag(Σx[:ocv])), x[:ocv] + 2sqrt.(diag(Σx[:ocv])), ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))
    lines!(ax1, b0, focv(b0), label="OCV real")
    ylims!(ax1, minimum(focv(b0)), maximum(focv(b0)))
    axislegend(ax1)


    ax2 = CairoMakie.Axis(fig[2, 1], title="R0 curve", xlabel="SOC", ylabel="V")
    lines!(ax2, b0, x[:r0], label="R0 aprox")
    band!(ax2, b0, x[:r0] - 2sqrt.(diag(Σx[:r0])), x[:r0] + 2sqrt.(diag(Σx[:r0])), ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))
    hlines!(ax2, 15e-3, label="R0 real")
    ylims!(ax2, 0, 30e-3)
    axislegend(ax2)
    display(fig)

end


