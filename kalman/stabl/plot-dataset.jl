
get_phases(dataset) = getfield.(keys(dataset), :p) |> unique |> sort
get_modules(dataset) = getfield.(keys(dataset), :m) |> unique |> sort
function get_cells(dataset)
    df = dataset[(p=1, m=1)]
    cell_voltage_cols = filter(col -> startswith(col, "cell_voltage_"), names(df))
    ids = map(cell_voltage_cols) do name
        m = match(r"cell_voltage_(\d+)", name)
        parse(Int, m.captures[1])
    end
    sort(ids)
end

function plot_module_data(dfs, id)
    colors = Makie.wong_colors()
    fig = Figure(size=(600, 700))
    ax = [Axis(fig[i, 1]) for i in 1:5]
    df = dfs[id]
    for c in 1:12
        lines!(ax[1], df.time / 3600, df[!, "cell_voltage_$c"])
    end
    lines!(ax[2], df.time / 3600, df.module_voltage, color=colors[1])
    lines!(ax[3], df.time / 3600, df.module_current, color=colors[2])
    lines!(ax[4], df.time / 3600, df.q, color=colors[3])
    lines!(ax[5], df.time / 3600, df.module_temperature, color=colors[4])

    for i in 1:3
        hidexdecorations!(ax[i], ticks=false, grid=false)
    end
    ax[end].xlabel = "Time / h"

    ax[1].ylabel = "Cell U / V"
    ax[2].ylabel = "Module U / V"
    ax[3].ylabel = "Module I / A"
    ax[4].ylabel = "Module Q / Ah"
    ax[5].ylabel = "Module T / °C"

    fig
end

function plot_data_alignment(data, id, t0)
    df_cells = data[:cell_voltage][id]
    df_module_avg = data[:module_current_avg][id]
    df_module_rms = data[:module_current_rms][id]

    fig = Figure()
    ax1 = Axis(fig[1, 1], ylabel="Module avg. current / A")
    ax2 = Axis(fig[2, 1], ylabel="Cell voltage / V", xlabel="Time / s")
    t_cell = Dates.value.(df_cells.time - t0) * 1e-3
    t_module_avg = Dates.value.(df_module_avg.time - t0) * 1e-3
    t_module_rms = Dates.value.(df_module_rms.time - t0) * 1e-3
    scatterlines!(ax1, t_module_avg, -df_module_avg.value, label="Avg.")
    scatterlines!(ax1, t_module_rms, -df_module_rms.value, label="RMS")
    axislegend(ax1, position=:rt)

    scatterlines!(ax2, t_cell, df_cells.cell_voltage_1)
    td = t_cell[[false; abs.(diff(df_cells.cell_voltage_1)) .> 0.01]]
    vlines!(ax1, td, color=(:black, 0.5))
    vlines!(ax2, td, color=(:black, 0.5))

    linkxaxes!(ax1, ax2)
    fig
end



# == from previous analysis


function plot_module_vi(dfs)
    fig = Figure(size=(1000, 600))
    gl = GridLayout(fig[1, 1])
    ax = [Axis(gl[i, j]) for i in 1:2, j in 1:3]

    for p in get_phases(dfs), m in get_modules(dfs)
        df = dfs[(; p, m)]
        # df = sample_dataset(data, ti, t0, (; p, m))
        lines!(ax[1, p], df.time / 3600, df.module_voltage)
        lines!(ax[2, p], df.time / 3600, df.module_current)
    end
    for i in 1:2, j in 2:3
        hideydecorations!(ax[i, j], grid=false, ticks=false)
    end
    for j in 1:3
        hidexdecorations!(ax[1, j], grid=false, ticks=false)
    end
    ax[1, 1].ylabel = "Module voltage in V"
    ax[2, 1].ylabel = "Module current in A"
    for i in 1:3
        ax[2, i].xlabel = "Time in h"
    end
    linkxaxes!(ax...)
    rowgap!(gl, 10)
    colgap!(gl, 10)
    fig
end

function plot_phase_vi(dfs, p)
    fig = Figure(size=(1000, 600))
    ax = [Axis(fig[i, 1]) for i in 1:2]
    for m in get_modules(dfs)
        # df = sample_dataset(data, ti, t0, (; p, m))
        df = dfs[(; p, m)]
        lines!(ax[1], df.time / 3600, df.module_voltage)
        lines!(ax[2], df.time / 3600, df.module_current)
    end

    hidexdecorations!(ax[1])
    ax[1].ylabel = "Module voltage in V"
    ax[2].ylabel = "Module current in A"
    ax[2].xlabel = "Time in h"
    linkxaxes!(ax...)
    fig
end

function plot_system_cell_v(dfs)
    phases = get_phases(dfs)
    modules = get_modules(dfs)
    cells = get_cells(dfs)

    fig = Figure(size=(600, 500))
    gl = GridLayout(fig[1, 1])
    ax = [Axis(gl[i, j], limits=(nothing, (3.3, 4.2))) for i in modules, j in phases]

    for p in phases, m in modules
        df = dfs[(; p, m)]
        for i in cells
            lines!(ax[m, p], df.time / 3600, df[:, "cell_voltage_$i"])
        end
    end

    for i in modules, j in phases[2:end]
        hideydecorations!(ax[i, j], grid=false, ticks=false)
    end
    for i in modules[1:end-1], j in phases
        hidexdecorations!(ax[i, j], grid=false, ticks=false)
    end
    linkaxes!(ax...)

    ax[5, 1].ylabel = "Cell voltage / V"

    for i in modules
        ax[i, 1].yticks = [3.5, 4.0]
    end
    for j in phases
        # ax[1, j].title = "Phase $j"
        ax[end, j].xlabel = "Time / h"
    end

    colgap!(gl, 10)
    rowgap!(gl, 10)

    fig
end

function plot_system_cell_Δv(dfs)
    phases = get_phases(dfs)
    modules = get_modules(dfs)
    cells = get_cells(dfs)


    fig = Figure(size=(600, 500))
    gl = GridLayout(fig[1, 1])
    ax = [Axis(gl[i, j], limits=(nothing, (0, 400)), xlabel="Time / h") for i in modules, j in phases]

    for p in phases, m in modules
        # df = sample_dataset(data, ti, t0, (; p, m), cells=1:8)
        df = dfs[(; p, m)]
        v = Array(df[:, ["cell_voltage_$i" for i in cells]])
        v_min = minimum(v; dims=2) |> vec
        v_max = maximum(v; dims=2) |> vec
        lines!(ax[m, p], df.time / 3600, 1e3 * (v_max - v_min))
        ax[m, p].yticks = [0.0, 200, 400]
    end

    ax[5, 1].ylabel = "Cell ΔV / mV"
    for i in modules, j in phases[2:end]
        hideydecorations!(ax[i, j], grid=false, ticks=false)
    end
    for i in modules[1:end-1], j in phases
        hidexdecorations!(ax[i, j], grid=false, ticks=false)
    end
    linkaxes!(ax...)

    colgap!(gl, 10)
    rowgap!(gl, 10)

    fig
end


# function plot_module_temperature(dfs)
#     fig = Figure(size=(800, 400))
#     ax = [Axis(fig[i, 1]) for i in 1:2]
#     df_temp = DataFrame()
#     df_temp[!, :datetime] = ti
#     tt = ti - ti[begin] .|> Second .|> Dates.value .|> Float64
#     df_temp[!, :t] = tt
#     for key in keys(data[:module_temperature])
#         p, m = key
#         df = sample_dataset(data, ti, t0, (; p, m))
#         df_temp[!, Symbol("p$(p)_m$(m)")] = df[:, :module_temperature]
#         lines!(ax[1], df.t / 3600, df.module_temperature)
#     end

#     # 
#     T_max = maximum(Array(df_temp[:, 3:end]); dims=2) |> vec
#     T_min = minimum(Array(df_temp[:, 3:end]); dims=2) |> vec
#     ΔT = T_max - T_min
#     lines!(ax[2], df_temp.t / 3600, ΔT)

#     #
#     ax[1].ylabel = "Module T in °C"
#     ax[2].ylabel = "Max ΔT in K"
#     ax[2].xlabel = "Time in h"

#     linkxaxes!(ax...)
#     fig
# end

function plot_cell_vq(dfs, cell)
    fig = Figure()
    ax = Axis(fig[1, 1]; ylabel="Voltage in V", xlabel="ΔQ in Ah")

    # df = sample_dataset(data, ti, t0, (; cell.m, cell.p), cells=cell.c)
    df = dfs[(; p=cell.p, cell.m)]
    q = df.q
    i = df.module_current
    v = df[:, "cell_voltage_$(cell.c)"]

    sc = scatter!(ax, q, v, color=abs.(i))
    Colorbar(fig[1, 2], sc, label="Current magnitude in A")
    fig
end