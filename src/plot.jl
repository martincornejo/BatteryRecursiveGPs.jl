function plot_sim(kf, sol; Ts=1.0, plot_Δv=true)
    zt = kf.p.zt
    (; idx, u, xt, Rt, ut, yt) = sol

    μ = zeros(length(idx))
    σ = zeros(length(idx))
    t = (0:length(u)-1) * Ts / 3600 |> collect

    for i in eachindex(idx, ut, xt, Rt)
        v = predict_kf(kf, ut[i], xt[i], Rt[i])
        μ[i] = StatsBase.reconstruct(zt.v, v.μ) |> first
        σ[i] = StatsBase.reconstruct(zt.σ, sqrt.(v.Σ)) |> first
    end

    # plot
    fig = Figure()
    # ax = [Axis(fig[i, 1]) for i in 1:3]
    ax = [Axis(fig[i, 1]) for i in 1:2]
    colors = Makie.wong_colors()

    # terminal voltage 
    v = StatsBase.reconstruct(zt.v, first.(yt))
    lines!(ax[1], t[idx], v)
    lines!(ax[1], t[idx], μ, color=colors[2])
    band!(ax[1], t[idx], μ - 2σ, μ + 2σ, color=(colors[2], 0.5))

    # voltage error
    e = v - μ
    lines!(ax[2], t[idx], e * 1e3, color=colors[2])
    band!(ax[2], t[idx], (e - 2σ) * 1e3, (e + 2σ) * 1e3, color=(colors[2], 0.5))

    # input current
    # i = StatsBase.reconstruct(zt.i, [x.i for x in u])
    # scatterlines!(ax[3], t, i, color=(:red, 0.5))

    # if plot_Δv
    #     Δv = abs.(diff(v))
    #     Δv_idx = idx[findall(>(0.01), Δv).+1]  # +1 because diff() reduces length by 1
    #     vlines!(ax[1], t[Δv_idx], color=(:gray, 0.5), linestyle=:dash)
    #     vlines!(ax[2], t[Δv_idx], color=(:gray, 0.5), linestyle=:dash)
    #     vlines!(ax[3], t[Δv_idx], color=(:gray, 0.5), linestyle=:dash)
    # end

    if sol.tt != length(sol.u)
        vlines!(ax[1], t[sol.tt]; color=:red)
        vlines!(ax[2], t[sol.tt]; color=:red)
    end

    xlims!.(ax, t[begin], t[end])

    ax[1].ylabel = "Voltage / V"
    ax[2].ylabel = "Voltage error / mV"
    ax[2].xlabel = "Time / h"

    linkxaxes!(ax...)

    fig
end


function plot_ecm(kf, sol=nothing; n=1)
    fig = Figure(size=(600, 600))
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / mΩ"
    ax[2].xlabel = "ΔQ / Ah"
    hidexdecorations!(ax[1], ticks=false, grid=false)

    zt = kf.p.zt

    if sol === nothing
        q̂min, q̂max = extrema(kf.p.r0.b0)
    else
        x = ComponentVector.(sol.xt, kf.p.xid)
        q̂min, q̂max = extrema([_x.cc.q for _x in x])
    end
    q̂ = q̂min:0.01:q̂max
    q = StatsBase.reconstruct(zt.q, q̂)

    # OCV 
    ocv = predict_gp(kf, q̂, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
    ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)

    lines!(ax[1], q, ocvμ)
    band!(ax[1], q, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha=0.8)

    # R0
    r0 = predict_gp(kf, q̂, :r0)
    rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1e3
    rσ = StatsBase.reconstruct(zt.r, r0.σ) * 1e3

    lines!(ax[2], q, rμ)
    band!(ax[2], q, rμ + 2rσ, rμ - 2rσ, alpha=0.8)

    ylims!(ax[1], 3.4 * n, 4.2 * n)
    ylims!(ax[2], 0.0 * n, 3.0 * n)

    linkxaxes!(ax...)
    fig
end


function plot_ecms(kfs, sols=nothing; n=1)
    fig = Figure(size=(450, 400))
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV (V)"
    ax[2].ylabel = "R₀ (mΩ)"
    # ax[2].xlabel = "SOC / p.u."
    ax[2].xlabel = "Charge (Ah)"
    hidexdecorations!(ax[1], ticks=false, grid=false)

    for (id, kf) in kfs

        zt = kf.p.zt

        if sols === nothing
            q̂min, q̂max = extrema(kf.p.r0.b0)
        else
            x = ComponentVector.(sols[id].xt, kf.p.xid)
            q̂min, q̂max = extrema([_x.cc.q for _x in x])
        end

        q̂ = collect(q̂min:0.01:q̂max)
        q = StatsBase.reconstruct(zt.q, q̂)

        # OCV 
        ocv = predict_gp(kf, q̂, :ocv)
        ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
        ocvσ = StatsBase.reconstruct(zt.σ, sqrt.(diag(ocv.Σ)))

        lines!(ax[1], q, ocvμ)
        band!(ax[1], q, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha=0.8)

        # R0
        r0 = predict_gp(kf, q̂, :r0)
        rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1e3
        rσ = StatsBase.reconstruct(zt.r, sqrt.(diag(r0.Σ))) * 1e3

        lines!(ax[2], q, rμ)
        band!(ax[2], q, rμ + 2rσ, rμ - 2rσ, alpha=0.8)

    end

    # xlims!(ax[1], 0, 1)
    # xlims!(ax[2], 0, 1)
    ylims!(ax[1], n * 3.2, n * 4.2)
    ylims!(ax[2], n * 0.0, n * 3.0)
    ax[1].yticks = (n*3.2):(n*0.3):(n*4.2)
    ax[2].yticks = (0):(n):(n*3)
    linkxaxes!(ax...)
    fig
end


function plot_ecms_norm(kfs, sols, fsoc, focv, fR0=nothing; vlim=(3.5, 3.95))
    fig = Figure(size=(600, 600))
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / mΩ"
    ax[2].xlabel = "SOC / p.u."
    hidexdecorations!(ax[1], ticks=false, grid=false)

    s = 0.0:0.01:1.0
    # s = 0.15:0.01:0.9
    lines!(ax[1], s, focv(s); color=:black, linestyle=:dash)
    if fR0 !== nothing
        lines!(ax[2], s, fR0.(s) * 1e3; color=:black, linestyle=:dash)
    end

    for (id, kf) in kfs

        zt = kf.p.zt

        soc0 = calc_soc0(kf, sols[id], fsoc; v=vlim) |> Measurements.value
        Q = calc_Q(kf, sols[id], fsoc; v=vlim) |> Measurements.value

        # q̂min, q̂max = extrema(kf.p.r0.b0)
        xs = ComponentVector.(sols[id].xt, kf.p.xid)
        q̂min, q̂max = extrema([x.cc.q for x in xs])
        q̂ = q̂min:0.01:q̂max
        q = StatsBase.reconstruct(zt.q, q̂)

        soc = soc0 .+ q ./ Q

        # OCV 
        ocv = predict_gp(kf, q̂, :ocv)
        ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
        ocvσ = StatsBase.reconstruct(zt.σ, sqrt.(diag(ocv.Σ)))

        lines!(ax[1], soc, ocvμ)
        band!(ax[1], soc, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha=0.8)

        # R0
        r0 = predict_gp(kf, q̂, :r0)
        rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1e3
        rσ = StatsBase.reconstruct(zt.r, sqrt.(diag(r0.Σ))) * 1e3

        lines!(ax[2], soc, rμ)
        band!(ax[2], soc, rμ + 2rσ, rμ - 2rσ, alpha=0.8)

    end

    # xlims!(ax[1], 0, 1)
    # xlims!(ax[2], 0, 1)
    ylims!(ax[1], 3.4, 4.15)
    ylims!(ax[2], 0.2, 3.0)
    linkxaxes!(ax...)
    fig
end


function plot_module_soh(param_cells, param_modules)
    # cell soh
    cell_soh = [param[:soh] for (id, param) in param_cells] .|> Measurements.value # |> maximum

    # module soh
    module_ids = [(; p, m) for p in 1:3, m in 1:9] |> vec |> sort
    df_soh = map(module_ids) do id
        (; p, m) = id
        # module state from cell state
        ids_ = [(; p, m, c) for c in 1:12]
        param_cells_module = Dict(id => param_cells[id] for id in ids_)
        soh = calc_soh_pack(param_cells_module, 100)
        u = calc_Q_utilization(param_cells_module)

        # module state from module state
        soh_module = param_modules[id][:soh]

        (; p, m, soh, u, soh_module)
    end |> DataFrame


    # figure
    # Cell-SOH
    fig = Figure(size=(950, 360))
    gl1 = GridLayout(fig[1, 1])
    ax1 = Axis(gl1[1, 1])

    bins = 40:1:90
    hist!(ax1, cell_soh * 100; bins, color=Cycled(5), strokecolor=:gray, strokewidth=1)

    ax1.xlabel = "Cell SOH (%)"
    ax1.ylabel = "Cell count"
    ylims!(ax1, 0, nothing)
    xlims!(ax1, first(bins), last(bins))

    ylims!(ax1, 0, 60)

    ax1.xgridvisible = false
    ax1.ygridvisible = false
    ax1.rightspinevisible = false
    ax1.topspinevisible = false
    ax1.ytrimspine = (false, true)

    # Module-SOH
    gl2 = GridLayout(fig[1, 2])
    ax2 = [Axis(gl2[j, i]) for i in 1:3, j in 1:2]

    for p in 1:3
        df_ = subset(df_soh, :p => ByRow(==(p)))
        soh_μ = Measurements.value.(df_.soh) * 100
        soh_σ = Measurements.uncertainty.(df_.soh) * 100

        m_soh_μ = Measurements.value.(df_.soh_module) * 100

        u_μ = Measurements.value.(df_.u) * 100
        u_σ = Measurements.uncertainty.(df_.u) * 100

        ids = 1:9

        barplot!(ax2[p, 1], ids, m_soh_μ; color=Cycled(3))
        barplot!(ax2[p, 2], ids, soh_μ; color=Cycled(1))
    end

    for p in 1:3, j in 1:2
        if j == 1
            ax2[p, j].title = "Phase $p"
        end
        ylims!(ax2[p, 1], 40, 90)
        ylims!(ax2[p, 2], 40, 90)
        # ylims!(ax[p, 2], 50, 100)

        ax2[p, j].xticks = 1:9

        ax2[p, j].xgridvisible = false
        ax2[p, j].ygridvisible = false
        ax2[p, j].rightspinevisible = false
        ax2[p, j].topspinevisible = false
    end

    # ax[2].leftspinevisible = false
    # ax[3].leftspinevisible = false
    for j in 1:2
        hideydecorations!(ax2[2, j]; ticks=false)
        hideydecorations!(ax2[3, j]; ticks=false)
    end
    ax2[1, 1].ylabel = "Module-based\nmodule SOH (%)"
    ax2[1, 2].ylabel = "Cell-based\nmodule SOH (%)"

    for p in 1:3
        ax2[p, end].xlabel = "Module"
    end


    Label(fig[0, 1], "Cell SOH estimation",
        fontsize=16,
        font=:bold,
        # padding=(0, 15, 15, 0),
        tellwidth=false,
        # halign=:left
    )
    Label(fig[0, 2], "Module SOH estimation",
        fontsize=16,
        font=:bold,
        # padding=(0, 0, 15, 0),
        tellwidth=false,
        # halign=:left
    )
    colsize!(fig.layout, 1, Relative(0.3))
    fig
end


function plot_cell_soh_hist(params)
    # cell soh
    cell_soh = [param[:soh] for (id, param) in params] .|> Measurements.value # |> maximum

    # figure
    # Cell-SOH
    fig = Figure(size=(450, 400))
    gl1 = GridLayout(fig[1, 1])
    ax1 = Axis(gl1[1, 1])

    bins = 40:1:90
    hist!(ax1, cell_soh * 100; bins, color=Cycled(5), strokecolor=:gray, strokewidth=1)

    ax1.xlabel = "Cell SOH (%)"
    ax1.ylabel = "Cell count"
    ylims!(ax1, 0, nothing)
    xlims!(ax1, first(bins), last(bins))

    ylims!(ax1, 0, 60)

    ax1.xgridvisible = false
    ax1.ygridvisible = false
    ax1.rightspinevisible = false
    ax1.topspinevisible = false
    ax1.ytrimspine = (false, true)

    fig
end


function plot_module_inhomogenity(params)
    # Module-SOH
    fig = Figure(size=(800, 400))
    ax2 = [Axis(fig[j, i]) for i in 1:3, j in 1:3]

    # module soh
    module_ids = [(; p, m) for p in 1:3, m in 1:9] |> vec |> sort
    df_soh = map(module_ids) do id
        (; p, m) = id
        ids_ = [(; p, m, c) for c in 1:12]
        params_module = Dict(id => params[id] for id in ids_)
        soh = calc_soh_pack(params_module, 100)
        u1 = calc_Q_utilization(params_module)
        u2 = calc_Q_utilization(params_module; delta_soc=false)

        socs = [val[:soc] for val in values(params_module)]
        sohs = [val[:soh] for val in values(params_module)]

        Δsoc_max = maximum(socs) - minimum(socs)
        Δsoh_max = maximum(sohs) - minimum(sohs)

        soh_module = params2[id][:soh]

        (; p, m, soh, u1, u2, soh_module, Δsoc_max, Δsoh_max)
    end |> DataFrame

    for p in 1:3
        df_ = subset(df_soh, :p => ByRow(==(p)))
        soh_μ = Measurements.value.(df_.soh) * 100
        soh_σ = Measurements.uncertainty.(df_.soh) * 100

        m_soh_μ = Measurements.value.(df_.soh_module) * 100

        u1_μ = Measurements.value.(df_.u1) * 100
        u_σ = Measurements.uncertainty.(df_.u1) * 100

        u2_μ = Measurements.value.(df_.u2) * 100
        u_σ = Measurements.uncertainty.(df_.u1) * 100

        Δsoc = Measurements.value.(df_.Δsoc_max) * 100
        Δsoh = Measurements.value.(df_.Δsoh_max) * 100

        ids = 1:9

        barplot!(ax2[p, 1], ids, Δsoc; color=Cycled(8))
        barplot!(ax2[p, 2], ids, Δsoh; color=Cycled(3))
        barplot!(ax2[p, 3], ids, u2_μ; color=Cycled(2), label="ΔSOH only")
        barplot!(ax2[p, 3], ids, u1_μ; color=Cycled(6), label="ΔSOH and ΔSOC")
    end

    for p in 1:3, j in 1:3
        if j == 1
            ax2[p, j].title = "Phase $p"
        end
        ylims!(ax2[p, 1], 0, 10)
        ylims!(ax2[p, 2], 0, 40)
        ylims!(ax2[p, 3], 50, 100)
        # ylims!(ax[p, 2], 50, 100)

        ax2[p, j].xticks = 1:9

        ax2[p, j].xgridvisible = false
        ax2[p, j].ygridvisible = false
        ax2[p, j].rightspinevisible = false
        ax2[p, j].topspinevisible = false
    end

    # ax[2].leftspinevisible = false
    # ax[3].leftspinevisible = false
    for j in 1:2
        hideydecorations!(ax2[2, j]; ticks=false)
        hideydecorations!(ax2[3, j]; ticks=false)
    end
    ax2[1, 1].ylabel = "Max.\nΔSOC (%)"
    ax2[1, 2].ylabel = "Max.\nΔSOH (%)"
    ax2[1, 3].ylabel = "Module Q\nutilization (%)"

    for p in 1:3
        ax2[p, end].xlabel = "Module"
    end

    Legend(fig[4, :], ax2[1, 3]; merge=true, orientation=:horizontal, framevisible=false)

    colsize!(fig.layout, 1, Relative(0.3))
    fig
end


function plot_rc_param_trajectory(kf, sol; r1=nothing, τ1=nothing)
    (; xid, Σid, zt) = kf.p
    xs = ComponentVector.(sol.xt, xid)
    Σs = [ComponentMatrix(R, Σid) for R in sol.Rt]

    rμ = StatsBase.reconstruct(zt.r, abs.([x.rc.r for x in xs])) * 1e3
    rσ = StatsBase.reconstruct(zt.r, sqrt.([Σ[:rc, :rc][:r, :r] for Σ in Σs])) * 1e3

    τμ = [x.rc.τ for x in xs]
    τσ = sqrt.([Σ[:rc, :rc][:τ, :τ] for Σ in Σs])

    vμ = StatsBase.reconstruct(zt.σ, [x.rc.v for x in xs]) * 1e3
    vσ = StatsBase.reconstruct(zt.r, sqrt.([Σ[:rc, :rc][:v, :v] for Σ in Σs])) * 1e3

    t = (1:length(rμ)) / 3600 * 10
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:3]
    lines!(ax[1], t, rμ)
    band!(ax[1], t, rμ - 2rσ, rμ + 2rσ; alpha=0.5)
    lines!(ax[2], t, τμ)
    band!(ax[2], t, τμ - 2τσ, τμ + 2τσ; alpha=0.5)

    lines!(ax[3], t, vμ)
    band!(ax[3], t, vμ - 2vσ, vμ + 2vσ; alpha=0.5)

    if r1 !== nothing
        hlines!(ax[1], r1 * 1e3; color=:black, linestyle=:dash)
    end
    if τ1 !== nothing
        hlines!(ax[2], τ1; color=:black, linestyle=:dash)
    end

    ax[1].ylabel = "R / mΩ"
    ax[2].ylabel = "τ / s"
    ax[3].ylabel = "RC voltage / mV"
    ax[3].xlabel = "Time / h"

    for _ax in ax
        xlims!(_ax, t[1], t[end])
    end

    linkxaxes!(ax...)
    fig
end

function plot_q_trajectory(kf, sol)
    (; xid, Σid, zt) = kf.p
    xs = ComponentVector.(sol.xt, xid)
    Σs = [ComponentMatrix(R, Σid) for R in sol.Rt]


    q = StatsBase.reconstruct(zt.q, [u.q for u in sol.ut])
    t = 1:length(q)

    qμ = StatsBase.reconstruct(zt.q, [x.cc.q for x in xs])
    qσ = StatsBase.reconstruct(zt.q, sqrt.([Σ[:cc, :cc][:q, :q] for Σ in Σs]))

    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]

    lines!(ax[1], t / 3600, q; color=Cycled(1), label="Coulomb counting")
    lines!(ax[1], t / 3600, qμ; color=Cycled(2), label="Estimated Q")
    band!(ax[1], t / 3600, qμ - 2qσ, qμ + 2qσ; color=Cycled(2), alpha=0.5, label="Estimated Q")

    lines!(ax[2], t / 3600, q - qμ; color=Cycled(2), label="Estimated Q")
    band!(ax[2], t / 3600, (q - qμ) - 2qσ, (q - qμ) + 2qσ; color=Cycled(2), alpha=0.5, label="Estimated Q")

    xlims!(ax[1], t[begin] / 3600, t[end] / 3600)

    ax[1].ylabel = "Charge / Ah"
    ax[2].ylabel = "Error / Ah"
    ax[2].xlabel = "Time / h"

    Legend(fig[3, 1], ax[1]; merge=true, orientation=:horizontal)
    fig
end


function plot_q_estimation(data, sol, kf)
    (; xid, Σid, zt) = kf.p
    xs = ComponentVector.(sol.xt, xid)
    Σs = [ComponentMatrix(R, Σid) for R in sol.Rt]


    q = StatsBase.reconstruct(zt.q, [u.q for u in sol.ut])
    t = 1:length(q)

    qμ = StatsBase.reconstruct(zt.q, [x.cc.q for x in xs])
    qσ = StatsBase.reconstruct(zt.q, sqrt.([Σ[:cc, :cc][:q, :q] for Σ in Σs]))

    q_real = data.q
    t_real = data.t

    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]

    lines!(ax[1], t_real / 3600, q_real; color=:black, label="Real Q")
    lines!(ax[1], t / 3600, q; color=:red, linestyle=:dash, label="Coulomb counting")
    lines!(ax[1], t / 3600, qμ; color=Cycled(2), label="Estimated Q")
    band!(ax[1], t / 3600, qμ - 2qσ, qμ + 2qσ; color=Cycled(2), alpha=0.5, label="Estimated Q")

    lines!(ax[2], t / 3600, qμ - q_real; color=Cycled(2), label="Estimated Q")
    band!(ax[2], t / 3600, (qμ - q_real) - 2qσ, (qμ - q_real) + 2qσ; color=Cycled(2), alpha=0.5, label="Estimated Q")
    lines!(ax[2], t_real / 3600, q - q_real; color=:red, linestyle=:dash, label="Coulomb counting")

    xlims!(ax[1], t[begin] / 3600, t[end] / 3600)

    ax[1].ylabel = "Charge / Ah"
    ax[2].ylabel = "Error / Ah"
    ax[2].xlabel = "Time / h"

    Legend(fig[3, 1], ax[1]; merge=true, orientation=:horizontal)
    fig
end


function plot_soc_trajectories(kfs, sols, kfs2, sols2, id_module)
    fig = Figure(size=(540, 400))
    ax = Axis(fig[1, 1])

    colors = Makie.wong_colors()

    (; p, m) = id_module
    Ts = 1.0
    t = sols[(; p, m, c=1)].idx * Ts
    df_soc = DataFrame(; t)

    for c in 1:12
        cell_id = (; p, m, c)
        sol = sols[cell_id]
        kf = kfs[cell_id]
        (; xid, Σid, zt) = kf.p
        x = ComponentVector.(sol.xt, xid)
        R = ComponentMatrix.(sol.Rt, Ref(Σid))
        qμ = StatsBase.reconstruct(zt.q, [_x.cc.q for _x in x])
        qσ = StatsBase.reconstruct(zt.q, [sqrt(_R[:cc, :cc][:q, :q]) for _R in R])

        Q = params[cell_id][:Q] |> Measurements.value
        soc0 = params[cell_id][:soc] |> Measurements.value

        socμ = qμ / Q .+ soc0
        socσ = qσ / Q

        df_soc[!, "soc_cell_$c"] = socμ

        lines!(ax, t / 3600, socμ; color=(:gray, 0.2), label="Cell SOC")
        band!(ax, t / 3600, socμ - socσ, socμ + socσ; color=(:gray, 0.1), label="Cell SOC")

        # lines!(ax, t / 3600, qμ; color=(:gray, 0.5))
        # band!(ax, t / 3600, qμ - 2qσ, qμ + 2qσ; color=(:gray, 0.1))
    end

    # soc
    soc_pack = calc_module_soc(df_soc, params, id_module)
    soc_pack_μ = soc_pack .|> Measurements.value
    soc_pack_σ = soc_pack .|> Measurements.uncertainty
    lines!(ax, t / 3600, soc_pack_μ; color=colors[1], label="Module SOC (cell)")
    band!(ax, t / 3600, soc_pack_μ - 2soc_pack_σ, soc_pack_μ + 2soc_pack_σ; color=(colors[1], 0.5), label="Module SOC (cell)")

    #
    sol2 = sols2[id_module]
    kf2 = kfs2[id_module]
    t2 = sol2.idx * Ts
    (; xid, Σid, zt) = kf2.p
    x = ComponentVector.(sol2.xt, xid)
    R = ComponentMatrix.(sol2.Rt, Ref(Σid))
    qμ = StatsBase.reconstruct(zt.q, [_x.cc.q for _x in x])
    qσ = StatsBase.reconstruct(zt.q, [sqrt(_R[:cc, :cc][:q, :q]) for _R in R])

    Q = params2[id_module][:Q] |> Measurements.value
    soc0 = params2[id_module][:soc] |> Measurements.value

    soc_moudle_μ = qμ / Q .+ soc0
    soc_moudle_σ = qσ / Q

    lines!(ax, t2 / 3600, soc_moudle_μ; color=colors[3], label="Module SOC (module)")
    band!(ax, t2 / 3600, soc_moudle_μ - 2soc_moudle_σ, soc_moudle_μ + 2soc_moudle_σ; color=(colors[3], 0.5), label="Module SOC (module)")

    ##
    ylims!(ax, 0, 1)
    xlims!(ax, first(t) / 3600, last(t) / 3600)

    ax.ylabel = "SOC (p.u.)"
    ax.xlabel = "Time (h)"

    ax.title = "Phase $p, Module $m"

    Legend(fig[2, 1], ax; merge=true, orientation=:horizontal)

    fig
end


function animate_ecm_evolution(file, kf, sol)
    fig = Figure(size=(600, 600))
    ax = [Makie.Axis(fig[i, 1]) for i in 1:3]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / mΩ"
    ax[2].xlabel = "ΔQ / Ah"
    hidexdecorations!(ax[1], ticks=false, grid=false)

    zt = kf.p.zt


    x = ComponentVector.(sol.xt, kf.p.xid)
    q̂min, q̂max = extrema([_x.cc.q for _x in x])

    # q̂min, q̂max = extrema(kf.p.r0.b0)


    q̂ = q̂min:0.01:q̂max
    q = StatsBase.reconstruct(zt.q, q̂)

    (; xt, Rt) = sol

    # OCV 
    ocv = predict_gp(kf, q̂, xt[1], Rt[1], :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
    ocvσ = StatsBase.reconstruct(zt.σ, sqrt.(diag(ocv.Σ)))

    l1 = lines!(ax[1], q, ocvμ)
    b1 = band!(ax[1], q, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha=0.8)

    # R0
    r0 = predict_gp(kf, q̂, xt[1], Rt[1], :r0)
    rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1e3
    rσ = StatsBase.reconstruct(zt.r, sqrt.(diag(r0.Σ))) * 1e3

    l2 = lines!(ax[2], q, rμ)
    b2 = band!(ax[2], q, rμ + 2rσ, rμ - 2rσ, alpha=0.8)


    # 
    v = StatsBase.reconstruct(zt.v, first.(sol.yt))
    # v = StatsBase.reconstruct(zt.i, first.(sol.ut))
    lines!(ax[3], v)
    v1 = vlines!(ax[3], 1; color=:red)

    ylims!(ax[1], 3.4, 4.2)
    ylims!(ax[2], 0.0, 3.0)
    # for i in eachindex(xt, Rt)
    # linkxaxes!(ax...)
    linkxaxes!(ax[1], ax[2])


    record(fig, file, 1:10:length(xt)) do i
        # OCV
        ocv = predict_gp(kf, q̂, xt[i], Rt[i], :ocv)
        ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
        ocvσ = StatsBase.reconstruct(zt.σ, sqrt.(diag(ocv.Σ)))

        # R0
        r0 = predict_gp(kf, q̂, xt[i], Rt[i], :r0)
        rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1e3
        rσ = StatsBase.reconstruct(zt.r, sqrt.(diag(r0.Σ))) * 1e3

        Makie.update!(l1, arg2=ocvμ)
        Makie.update!(b1, arg2=ocvμ + 2ocvσ, arg3=ocvμ - 2ocvσ)

        Makie.update!(l2, arg2=rμ)
        Makie.update!(b2, arg2=rμ + 2rσ, arg3=rμ - 2rσ)

        Makie.update!(v1, arg1=i)
    end

end


function animate_model(file, sol)
    fig = Figure(size=(900, 400))
    gl1 = GridLayout(fig[1, 2])
    ax1 = [Axis(gl1[i, 1]) for i in 1:2]
    rowsize!(gl1, 1, Relative(0.65))
    # rowsize!(gl1, 1, Auto(2))

    gl2 = GridLayout(fig[1, 1])
    ax2 = [Axis(gl2[i, 1]) for i in 1:2]

    colors = Makie.wong_colors()

    θ0 = ComponentVector(; # tunable (hyper)params
        ocv=(; σ=0.5, ℓ=0.7),
        r0=(; σ=0.05, ℓ=0.5),
        vσ=3e-3,
    )
    ϑ = ComponentVector(; # non-tunable params
        Ts=1.0,
        r0μ=1.0e-3,
        rc=(;
            v0=0.0, σ0_v=1e-3, σ1_v=1.0e-4,
            r0=1.0e-3, σ0_r=0.5e-3, σ1_r=0.0,
            τ0=300.0, σ0_τ=30.0, σ1_τ=0.0,
        ),
    )
    θ = ComponentVector(; θ0..., ϑ...)

    Ts = 1.0
    (; u, y) = sol
    # tt = length(y)
    tt = 0.0
    zt = fit_zscore()
    kf = build_kf(θ, u, zt)
    sol = run_kf!(kf, u, y; tt)

    t = sol.idx * Ts

    ## output prediction
    vμ = StatsBase.reconstruct(zt.v, first.(sol.yμ))
    vσ = StatsBase.reconstruct(zt.σ, sqrt.(first.(sol.yΣ)))
    ve = StatsBase.reconstruct(zt.σ, sol.et)
    v̂ = StatsBase.reconstruct(zt.v, first.(sol.yt))

    lines!(ax1[1], t / 3600, v̂, color=:gray, label="Measurement")
    l1 = lines!(ax1[1], t / 3600, vμ; color=colors[6], label="Model")
    b1 = band!(ax1[1], t / 3600, vμ - 2vσ, vμ + 2vσ; color=(colors[2], 0.5), label="Model")
    v1 = vlines!(ax1[1], tt; color=:red, linestyle=:dash)
    l2 = lines!(ax1[2], t / 3600, ve * 1e3, color=colors[6])
    b2 = band!(ax1[2], t / 3600, (ve - 2vσ) * 1e3, (ve + 2vσ) * 1e3; color=(colors[2], 0.5))

    xlims!(ax1[1], 0, length(sol.u) / 3600)
    xlims!(ax1[2], 0, length(sol.u) / 3600)
    ylims!(ax1[1], 3.45, 4.2)
    ylims!(ax1[2], -50, 50)

    ax1[1].ylabel = "Terminal voltage (V)"
    ax1[2].ylabel = "Error (mV)"
    ax1[2].xlabel = "Time (h)"

    ax1[1].xminorticks = IntervalsBetween(5)
    ax1[1].xminorticksvisible = true
    ax1[2].xminorticks = IntervalsBetween(5)
    ax1[2].xminorticksvisible = true

    ax1[1].yminorticks = IntervalsBetween(2)
    ax1[1].yminorticksvisible = true
    ax1[2].yminorticks = IntervalsBetween(5)
    ax1[2].yminorticksvisible = true

    axislegend(ax1[1]; merge=true, position=:ct, orientation=:horizontal, framevisible=false)
    hidexdecorations!(ax1[1], ticks=false, minorticks=false, grid=false, minorgrid=false)

    ## ecm
    x = ComponentVector.(sol.xt, kf.p.xid)
    q̂min, q̂max = extrema([_x.cc.q for _x in x])

    q̂ = q̂min:0.01:q̂max
    q = StatsBase.reconstruct(zt.q, q̂)

    # OCV 
    ocv = predict_gp(kf, q̂, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
    ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)

    l3 = lines!(ax2[1], q, ocvμ)
    b3 = band!(ax2[1], q, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha=0.8)

    # R0
    r0 = predict_gp(kf, q̂, :r0)
    rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1e3
    rσ = StatsBase.reconstruct(zt.r, r0.σ) * 1e3

    l4 = lines!(ax2[2], q, rμ)
    b4 = band!(ax2[2], q, rμ + 2rσ, rμ - 2rσ, alpha=0.8)

    ylims!(ax2[1], 3.2, 4.2)
    ylims!(ax2[2], 0.0, 3.0)
    ax2[1].yticks = 3.2:0.3:4.2
    ax2[2].yticks = 0:3

    ax2[1].ylabel = "OCV (V)"
    ax2[2].ylabel = "R₀ (mΩ)"
    ax2[2].xlabel = "Charge (Ah)"
    hidexdecorations!(ax2[1], ticks=false, grid=false)

    ax2[1].title = "ECM reconstruction"
    ax1[1].title = "Model prediction"
    colgap!(fig.layout, 30)

    record(fig, file, 1:60:length(sol.y)) do i
        # simulate
        kf = build_kf(θ, u, zt)
        sol = run_kf!(kf, u, y; tt=i)

        vμ = StatsBase.reconstruct(zt.v, first.(sol.yμ))
        vσ = StatsBase.reconstruct(zt.σ, sqrt.(first.(sol.yΣ)))
        ve = StatsBase.reconstruct(zt.σ, sol.et)

        Makie.update!(l1, arg2=vμ)
        Makie.update!(b1, arg2=vμ + 2vσ, arg3=vμ - vσ)

        Makie.update!(l2, arg2=ve * 1e3)
        Makie.update!(b2, arg2=(ve + 2vσ) * 1e3, arg3=(ve - 2vσ) * 1e3)

        Makie.update!(v1, arg1=i / 3600)

        # OCV
        ocv = predict_gp(kf, q̂, :ocv)
        ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
        ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)

        # R0
        r0 = predict_gp(kf, q̂, :r0)
        rμ = StatsBase.reconstruct(zt.r, r0.μ) * 1e3
        rσ = StatsBase.reconstruct(zt.r, r0.σ) * 1e3

        Makie.update!(l3, arg2=ocvμ)
        Makie.update!(b3, arg2=ocvμ + 2ocvσ, arg3=ocvμ - 2ocvσ)

        Makie.update!(l4, arg2=rμ)
        Makie.update!(b4, arg2=rμ + 2rσ, arg3=rμ - 2rσ)
    end

    fig
end

