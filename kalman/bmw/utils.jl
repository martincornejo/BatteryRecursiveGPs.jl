function fit_modules(data, ti, t0, ids; N_points=60_000, cells=1:8)
    m = Dict()
    for id in ids
        df = sample_dataset(data, ti, t0, id; cells=1:8)
        m[id] = fit_cells(df, cells=cells, N_points=N_points)
        @info "Module $(id): fit complete"
    end
    return m
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


function fit_cells(df; cells=1:8, N_points=60_000)
    r = Dict()
    for cell in cells
        df_cell = rename(df, [:module_current => :i, :module_temperature => :T, Symbol("cell_voltage_$cell") => :v])
        select!(df_cell, [:t, :i, :v, :T])
        df_cell[!, :i] = -df_cell.i
        df_cell[!, :q] = cumsum(df_cell.i)

        i = ConstantInterpolation(df_cell.i, df_cell.t)
        v = ConstantInterpolation(df_cell.v, df_cell.t)

        df_train = df_cell[1:N_points, :]

        dt = fit_zscore(df_train)

        battModel = generate_model(dt, df_train)

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
        soc_values = []
        soc_sigmas = []
        rc_values = []
        p = Progress(length(data))
        for (n, batch) in enumerate(data)

            kf(batch.u, batch.y)

            x_soc = ComponentVector(kf.x, battModel.p.xid)[:ocv][end]
            Σ_soc = 2sqrt.(ComponentMatrix(kf.R, battModel.p.Σid)[:ocv][end, end])
            push!(soc_values, x_soc)
            push!(soc_sigmas, Σ_soc)
            x_rc = ComponentVector(kf.x, battModel.p.xid)[:rc1]
            push!(rc_values, (; t=batch.u, rc1_values=copy(x_rc)))
            next!(p)
        end

        r[cell] = (;
            soc=(; soc_values, soc_sigmas),
            kf=kf,
            p=battModel.p,
            rc=rc_values,
            dt=dt)
    end

    return r

end


function generate_model(dt, df_train)

    l_ocv = 0.4
    σ_ocv = 0.5
    l_r = 0.5
    σ_r = 0.5


    n_basis = 100
    b0 = collect(
        range(minimum(df_train[!, :q]),
            maximum(df_train[!, :q]);
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


    prior_ocv = [-1.6644776074370395, -2.1836393286382876, -1.3553789570074029, -1.3702869055983842, -1.7648821247879702, -1.7315555472574704, -1.255678911191118, -0.8832691174920763, -0.9629222082959771, -1.2992969597781494, -1.4613187618103705, -1.2801223264712007, -0.9814479247644765, -0.8770584298751375, -1.0125275817156345, -1.1569359574002582, -1.1026413264093538, -0.9083831728614248, -0.8156569683671129, -0.9535560246877967, -1.1789540324174006, -1.2321630314773706, -1.0277868083237403, -0.7528927343134321, -0.6647524810829583, -0.8136787334640351, -0.9938683350796478, -0.9702171360330093, -0.7298160027885519, -0.4887458192931018, -0.45670216596990626, -0.6198692094965514, -0.7707145355537229, -0.7396968962476124, -0.5722555916942473, -0.46676290033394646, -0.5527628766779213, -0.7531282330452228, -0.8707626413035959, -0.7955610835875943, -0.6078867142512239, -0.47827633405012104, -0.483200328493298, -0.5357233238108765, -0.4949387932031017, -0.3257570140509931, -0.13656333234296533, -0.062185940327126724, -0.12425646826942909, -0.21634451563991747, -0.22118123056828, -0.12879725833367284, -0.03261438817718923, -0.01749624939598707, -0.06702596208244838, -0.08811323729031877, -0.018489268373144352, 0.10543886452454063, 0.193096262863236, 0.20177879156814743, 0.17965078835517378, 0.20409792777017344, 0.2905230079621033, 0.3788428276119491, 0.41209210021947545, 0.4114023376161228, 0.4537041863988938, 0.5736205340857832, 0.7063208464951826, 0.7482866105744578, 0.677010059325833, 0.5935946511433582, 0.6289661563561093, 0.8059327167755506, 1.0060366491227666, 1.0829790026521513, 1.007003946362146, 0.8897345915797488, 0.8660541040749802, 0.9604132305393175, 1.0764972842487819, 1.1126701990688415, 1.0724030820918355, 1.0516690669897326, 1.124149549517762, 1.2553742241502495, 1.3408504893486293, 1.3245887900413, 1.2655938025310027, 1.276100709948883, 1.3955892558567358, 1.538879651159547, 1.5832107993290894, 1.5093663419899277, 1.4399706481033463, 1.5094452770255038, 1.6837630358887268, 1.7538701350561317, 1.6344390015783057, 1.8505767307263796]
    mean_ocv = LinearInterpolation(StatsBase.transform(dt.v, prior_ocv), b0; extrapolation=ExtrapolationType.Linear)
    mean_r = fill(15e-3, length(b0))


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

    soc0 = 0.5
    Q_ = 0.0
    q0 = Q_ * soc0

    soc = Q(;
        q0=q0,
        σ1=1e-9,
        Σ_q=(soc0 - 0.5)^2 + 1e-6,
        ts=1)

    r0_ocv = R0_OCV(ocv, r0, soc)

    rc1 = RC(
        1, 30, 50e-3,
        Vrc_σ=1e-3,
        σ1=[1e-3, 2e-3, 3e-11],
        σ2=sqrt(1e-3),
        τh=90,
        Rh=10e-3)


    components_batt = (;
        ocv=r0_ocv,
        rc1=rc1
    )


    battModel = BattModel(components_batt)
    return battModel

end


function plot_soc(cell, n_mod, n_cell; f=Figure(size=(800, 600)), fig_x=1, fig_y=1)
    soc_values = cell.soc.soc_values
    soc_sigmas = cell.soc.soc_sigmas
    N_points = size(soc_values, 1)
    ax1 = CairoMakie.Axis(f[fig_y, fig_x], title="module = $(n_mod), cell = $(n_cell)", xlabel="time", ylabel="q")
    lines!(ax1, soc_values, label="q aprox")
    band!(
        ax1, collect(1:1:N_points), soc_values - soc_sigmas, soc_values + soc_sigmas,
        ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3)
    )
    axislegend(ax1)
    return f
end


function plot_ocv(cell, n_mod, n_cell; f=Figure(size=(800, 600)), fig_x=1, fig_y=1)
    kf = cell.kf
    p = cell.p
    dt = cell.dt
    soc_values = cell.soc.soc_values

    max_soc = maximum(soc_values)
    min_soc = minimum(soc_values)

    x = ComponentVector(kf.x, p.xid)[:ocv]
    Σx = ComponentMatrix(kf.R, p.Σid)[:ocv, :ocv]

    bp = p.components.ocv.p.ocv.p.b0
    name = :c_ocv

    # OCV curve
    ax1 = CairoMakie.Axis(f[fig_y, fig_x], title="OCV module = $(n_mod), cell = $(n_cell)", xlabel="SOC", ylabel="V")
    lines!(ax1, StatsBase.reconstruct(dt.q, bp), StatsBase.reconstruct(dt.v, x[name]), label="OCV aprox")
    band!(
        ax1, StatsBase.reconstruct(dt.q, bp), StatsBase.reconstruct(dt.v, x[name]) - 2sqrt.(diag(dt.v.scale .^ 2 .* Σx[name])),
        StatsBase.reconstruct(dt.v, x[name]) + 2sqrt.(diag(dt.v.scale .^ 2 .* Σx[name])),
        ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))
    axislegend(ax1)
    vlines!(ax1, min_soc, color=:red)
    vlines!(ax1, max_soc, color=:red)
    return f
end

function plot_r0(cell, n_mod, n_cell; f=Figure(size=(800, 600)), fig_x=1, fig_y=1)
    kf = cell.kf
    p = cell.p
    dt = cell.dt
    soc_values = cell.soc.soc_values

    max_soc = maximum(soc_values)
    min_soc = minimum(soc_values)

    x = ComponentVector(kf.x, p.xid)[:ocv]
    Σx = ComponentMatrix(kf.R, p.Σid)[:ocv, :ocv]

    bp = p.components.ocv.p.ocv.p.b0

    # R0 curve
    name2 = :c_r0
    ax2 = CairoMakie.Axis(f[fig_y, fig_x], title="R0 module = $(n_mod), cell = $(n_cell)", xlabel="SOC", ylabel="mOhm")

    lines!(ax2, StatsBase.reconstruct(dt.q, bp), StatsBase.reconstruct(dt.σ, x[name2]), label="R0 aprox")
    band!(
        ax2, StatsBase.reconstruct(dt.q, bp), StatsBase.reconstruct(dt.σ, x[name2]) - 2sqrt.(diag(dt.σ.scale .^ 2 .* Σx[name2])),
        StatsBase.reconstruct(dt.σ, x[name2]) + 2sqrt.(diag(dt.σ.scale .^ 2 .* Σx[name2])),
        ; label="uncertainty band", color=(Makie.wong_colors()[2], 0.3))
    vlines!(ax2, min_soc, color=:red)
    vlines!(ax2, max_soc, color=:red)
    axislegend(ax2)

    return f

end

function plot_rc(cell;)
    rc_values = cell.rc

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
    hlines!(ax2, df_rc.tau[end], label="τ end", color=:red)
    axislegend(ax2)


    ax3 = Axis(f[3, 1], title="R over Time", xlabel="Time", ylabel="R [Ω]")
    lines!(ax3, df_rc.R)
    hlines!(ax3, df_rc.R[end], label="R end", color=:red)
    ylims!(ax3, 0.0, 0.005)
    axislegend(ax3)

    display(f)
end



