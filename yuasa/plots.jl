function plot_dataset(df)
    fig = Figure()
    colors = Makie.wong_colors()
    ax = [Axis(fig[i, 1]) for i in 1:4]
    for i in 1:12
        lines!(ax[1], df.t / 3600, df[:, "v_cell_$i"])
    end
    lines!(ax[2], df.t / 3600, df.v, color=colors[1])
    lines!(ax[3], df.t / 3600, df.i, color=colors[2])
    lines!(ax[4], df.t / 3600, df.q, color=colors[3])
    fig
end

