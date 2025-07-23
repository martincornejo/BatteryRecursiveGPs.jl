
function fit_module(df, battModel, dt)

    df_train = df


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


    data = DataLoader((
            u=(;
                i=df_train.i
            ),
            y=df_train.v
        ),
        batchsize=1, shuffle=false
    )

    p = Progress(length(data))
    soc_values = []
    soc_sigmas = []
    rc_values = []
    for (n, batch) in enumerate(data)

        kf(batch.u, batch.y)

        x_soc = ComponentVector(kf.x, battModel.p.xid)[:r0_ocv][end]
        Σ_soc = 2sqrt.(ComponentMatrix(kf.R, battModel.p.Σid)[:r0_ocv][end, end])
        push!(soc_values, x_soc)
        push!(soc_sigmas, Σ_soc)
        #x_rc = ComponentVector(kf.x, battModel.p.xid)[:rc1]
        #push!(rc_values, (; t=batch.u, rc1_values=copy(x_rc)))
        next!(p)
    end

    r = (;
        soc=(; soc_values, soc_sigmas),
        kf=kf,
        p=battModel.p,
        rc=rc_values,
        dt=dt)
    return r

end




function fit_zscore(df)
    v = StatsBase.fit(ZScoreTransform, df.v)
    σ = StatsBase.fit(ZScoreTransform, df.v, center=false)
    i = StatsBase.fit(ZScoreTransform, df.i, center=false)
    q = StatsBase.fit(ZScoreTransform, df.q)
    return (; v, σ, i, q)
end

function normalize_data(df, dt)
    v = df.v
    i = df.i
    q = StatsBase.transform(dt.q, df.q)
    return DataFrame(; df.t, v, i, q)
end




function create_batt_no_rc(df, dt; l_ocv=0.4, σ_ocv=0.5, l_r=0.5, σ_r=0.5, q0=0.0, ts_q=1)
    n_basis = 100
    b0 = collect(
        range(minimum(df[!, :q]),
            maximum(df[!, :q]);
            length=n_basis)) # basis vector for OCV

    b0 = StatsBase.transform(dt.q, b0)

    gp_ocv = GP(ZeroMean(), LinearKernel() + σ_ocv * with_lengthscale(SEKernel(), l_ocv))
    ocv = OCV(
        gp_ocv, b0;
        σ2=1e-5,
        tr=dt.v,
        tr_b=dt.q)

    gp_r0 = GP(ZeroMean(), σ_r * with_lengthscale(SEKernel(), l_r))

    r0 = R0(gp_r0, b0;
        σ2=1e-10,
        tr=dt.σ,
        tr_b=dt.q)

    soc = Q(;
        q0=q0,
        σ1=1e-9,
        Σ_q=1e-6,
        ts=ts_q)

    r0_ocv = R0_OCV(ocv, r0, soc)

    components_batt = (;
        r0_ocv=r0_ocv,
    )


    battModel = BattModel(components_batt)
    return battModel
end


function create_batt_rc(battModel_no_rc, mean_ocv, mean_r; ts_rc=1, R0=50e-3, τ0=30)
    r0 = battModel_no_rc.components.r0_ocv.r0
    ocv = battModel_no_rc.components.r0_ocv.ocv
    soc = battModel_no_rc.components.r0_ocv.soc

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
        d0=MvNormal(mean_r, cov(r0.d0)),
        nx=r0.nx,
        ny=r0.ny,
        p=r0.p
    )

    r0_ocv = R0_OCV(ocv, r0, soc)


    rc1 = RC(
        ts_rc, τ0, R0,
        Vrc_σ=1e-3,
        σ1=[1e-3, 2e-3, 3e-11],
        σ2=sqrt(1e-3),
        τh=90,
        Rh=10e-3)

    battModel = BattModel(components_batt)

    components_batt = (;
        ocv=r0_ocv,
        rc1=rc1
    )

    return battModel
end



function plot_soc(cell; f=Figure(size=(800, 600)))
    fig_x = 1
    fig_y = 1
    soc_values = cell.soc.soc_values
    soc_sigmas = cell.soc.soc_sigmas
    N_points = size(soc_values, 1)
    ax1 = CairoMakie.Axis(f[fig_y, fig_x], xlabel="time", ylabel="q")
    lines!(ax1, soc_values, label="q aprox")
    band!(
        ax1, collect(1:1:N_points), soc_values - soc_sigmas, soc_values + soc_sigmas,
        ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3)
    )
    axislegend(ax1)


    display(f)
end


function plot_ocv(cell; f=Figure(size=(800, 600)))
    fig_x = 1
    fig_y = 1
    kf = cell.kf
    p = cell.p
    dt = cell.dt
    soc_values = cell.soc.soc_values

    max_soc = maximum(soc_values)
    min_soc = minimum(soc_values)

    x = ComponentVector(kf.x, p.xid)[:r0_ocv]
    Σx = ComponentMatrix(kf.R, p.Σid)[:r0_ocv, :r0_ocv]

    bp = p.components.r0_ocv.p.ocv.p.b0
    println(x[:c_ocv])
    name = :c_ocv

    # OCV curve
    ax1 = CairoMakie.Axis(f[fig_y, fig_x], xlabel="SOC", ylabel="V")
    lines!(ax1, StatsBase.reconstruct(dt.q, bp), StatsBase.reconstruct(dt.v, x[name]), label="OCV aprox")
    band!(
        ax1, StatsBase.reconstruct(dt.q, bp), StatsBase.reconstruct(dt.v, x[name]) - 2sqrt.(diag(dt.v.scale .^ 2 .* Σx[name])),
        StatsBase.reconstruct(dt.v, x[name]) + 2sqrt.(diag(dt.v.scale .^ 2 .* Σx[name])),
        ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))
    axislegend(ax1)
    vlines!(ax1, min_soc, color=:red)
    vlines!(ax1, max_soc, color=:red)

    display(f)
end

function plot_r0(cell; f=Figure(size=(800, 600)))
    fig_x = 1
    fig_y = 1
    kf = cell.kf
    p = cell.p
    dt = cell.dt
    soc_values = cell.soc.soc_values

    max_soc = maximum(soc_values)
    min_soc = minimum(soc_values)

    x = ComponentVector(kf.x, p.xid)[:r0_ocv]
    Σx = ComponentMatrix(kf.R, p.Σid)[:r0_ocv, :r0_ocv]

    bp = p.components.r0_ocv.p.ocv.p.b0

    # R0 curve
    name2 = :c_r0
    ax2 = CairoMakie.Axis(f[fig_y, fig_x], xlabel="SOC", ylabel="mOhm")

    lines!(ax2, StatsBase.reconstruct(dt.q, bp), StatsBase.reconstruct(dt.σ, x[name2]), label="R0 aprox")
    band!(
        ax2, StatsBase.reconstruct(dt.q, bp), StatsBase.reconstruct(dt.σ, x[name2]) - 2sqrt.(diag(dt.σ.scale .^ 2 .* Σx[name2])),
        StatsBase.reconstruct(dt.σ, x[name2]) + 2sqrt.(diag(dt.σ.scale .^ 2 .* Σx[name2])),
        ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))
    vlines!(ax2, min_soc, color=:red)
    vlines!(ax2, max_soc, color=:red)
    axislegend(ax2)

    display(f)
end
function plot_rc(cell; f=Figure(size=(800, 600)))

    fig_x = 1
    fig_y = 1
    rc_values = cell.rc

    df_rc = DataFrame(
        Vrc=[entry.rc1_values.Vrc for entry in rc_values],
        tau=[entry.rc1_values.τ for entry in rc_values],
        R=[entry.rc1_values.R for entry in rc_values]
    )
    ax1 = Axis(f[fig_y, fig_x], title="Vrc", xlabel="Time", ylabel="Vrc [V]")
    lines!(ax1, df_rc.Vrc, label="V real", color=:red)

    ax2 = Axis(f[fig_y+1, fig_x], title="τ over Time", xlabel="Time", ylabel="τ [s]")
    lines!(ax2, df_rc.tau)
    hlines!(ax2, df_rc.tau[end], label="τ end", color=:red)
    axislegend(ax2)


    ax3 = Axis(f[fig_y+2, fig_x], title="R over Time", xlabel="Time", ylabel="R [Ω]")
    lines!(ax3, df_rc.R)
    hlines!(ax3, df_rc.R[end], label="R end", color=:red)
    ylims!(ax3, 0.0, 0.005)
    axislegend(ax3)

    display(f)
end

