function plot_ecm(kf, zt, focv, fR0; closeup=true)
    fig = Figure(size=(600, 600))
    colors = Makie.wong_colors()
    ax = [Makie.Axis(fig[i, 1]) for i in 1:2]
    ax[1].ylabel = "OCV / V"
    ax[2].ylabel = "R0 / Ω"

    # for n in (11, 21, 51, 101)
    n = 21
    # kf = build_kf(n)
    # t = @timed run_kf!(kf, us, ys)
    # (; time, bytes) = t
    # memory = 1e-6 * bytes
    # @info n time memory

    # predict new points -> mean and std
    # smin, smax = df.s |> extrema
    # bgp = StatsBase.transform(zt.s, smin:0.01:smax)
    soc = 0:0.01:1
    # bgp = StatsBase.transform(zt.s, 0:0.01:1)
    # bgp = 0:0.01:1
    ocv = predict_gp(kf, soc, :ocv)
    ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
    ocvσ = StatsBase.reconstruct(zt.σ, ocv.σ)


    # plot results 
    # lines!(ax[1], 0:0.01:1, f1.(0:0.01:1), color=colors[1], label="f1(x)")
    lines!(ax[1], soc, ocvμ)
    band!(ax[1], soc, ocvμ + 2ocvσ, ocvμ - 2ocvσ, alpha=0.8)
    # scatter!(ax[1], df_train.s, df.y, color=(:red, 0.5), label="Data")
    # axislegend(ax[1]; merge=true, position=:lt)
    lines!(ax[1], soc, focv(soc), color=:black, linestyle=:dot)

    if closeup
        ylims!(ax[1], 3.45, 4.2)
    end


    # predict new points -> mean and std
    r0 = predict_gp(kf, soc, :r0)
    rμ = StatsBase.reconstruct(zt.r, r0.μ)
    rσ = StatsBase.reconstruct(zt.r, r0.σ)

    # plot results 
    # lines!(ax[2], 0:0.01:1, f2.(0:0.01:1), color=colors[1], label="f2(x)")
    lines!(ax[2], soc, rμ)
    band!(ax[2], soc, rμ + 2rσ, rμ - 2rσ, alpha=0.8)

    lines!(ax[2], soc, fR0.(soc), color=:black, linestyle=:dot)
    # axislegend(ax[2]; merge=true, position=:lt)
    # end
    fig
end

function plot_soc_estimation(smoothsol, df)
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]
    s´ = smoothsol.xT .|> first
    σ = smoothsol.Rt .|> first

    lines!(ax[1], df.t / 3600, s´, label="Estimate")
    band!(ax[1], df.t / 3600, s´ - 2σ, s´ + 2σ, label="Estimate")
    # lines!(ax, df_.t, 0.4 .+ df_.s ./ 4.8)
    lines!(ax[1], df.t / 3600, df.s, label="True")
    axislegend(ax[1], location=:lt, merge=true)

    lines!(ax[2], df.t / 3600, df.s - s´)

    linkxaxes!(ax...)
    fig
end
