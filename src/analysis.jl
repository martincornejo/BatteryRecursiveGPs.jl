
# TODO: rename functions

function calc_deltaq(model::AbstractBatteryModel, sol; v=(3.85, 4.0), n=1)
    kf = model.kf
    v1 = v[1] * n
    v2 = v[2] * n

    zt = kf.p.zt

    xs = ComponentVector.(sol.xt, kf.p.xid)
    q̂min, q̂max = extrema([x.cc.q for x in xs])
    q̂ = range(q̂min, q̂max, 50) |> collect
    q = StatsBase.reconstruct(zt.q, q̂)

    # OCV
    ocv = predict_gp(kf, q̂, :ocv)
    μ = StatsBase.reconstruct(zt.v, ocv.μ)
    σ = StatsBase.reconstruct(zt.σ, sqrt.(diag(ocv.Σ)))

    q1μ = q[findfirst(>=(v1), μ)]
    q1σ = q1μ - q[findfirst(>=(v1), μ + σ)]
    q1 = q1μ ± q1σ

    q2μ = q[findfirst(>=(v2), μ)]
    q2σ = q[findfirst(>=(v2), μ - σ)] - q2μ
    q2 = q2μ ± q2σ

    q2 - q1
end

function calc_Q(model::AbstractBatteryModel, sol, fsoc; v=(3.85, 4.0), n=1)
    v1, v2 = v

    Δsoc = fsoc(v2) - fsoc(v1)
    Δq = calc_deltaq(model, sol; v, n)
    Δq / (Δsoc)
end

function calc_soh(model::AbstractBatteryModel, sol, fsoc, Q; v=(3.85, 4.0), n=1)
    Q´ = calc_Q(model, sol, fsoc; v, n)
    return Q´ / Q
end

function calc_soc0(model::AbstractBatteryModel, sol, fsoc; v=(3.85, 4.0), n=1)
    kf = model.kf
    v1 = v[1] * n

    zt = kf.p.zt

    xs = ComponentVector.(sol.xt, kf.p.xid)
    q̂min, q̂max = extrema([x.cc.q for x in xs])
    q̂ = range(q̂min, q̂max, 200) |> collect
    q = StatsBase.reconstruct(zt.q, q̂)

    # OCV
    ocv = predict_gp(kf, q̂, :ocv)
    μ = StatsBase.reconstruct(zt.v, ocv.μ)
    σ = StatsBase.reconstruct(zt.σ, sqrt.(diag(ocv.Σ)))

    q1μ = q[findfirst(>=(v1), μ)]
    q1σ = q1μ - q[findfirst(>=(v1), μ + σ)]
    q1 = q1μ ± q1σ

    Q´ = calc_Q(model, sol, fsoc; v, n)
    Δs = q1 / Q´

    s0 = fsoc(v1 / n)
    s0 - Δs
end


function calc_Q_pack(params)
    Qdch = map(collect(keys(params))) do cell_id
        cell = params[cell_id]
        Qdch = cell[:soc] * cell[:Q]
    end |> minimum

    Qch = map(collect(keys(params))) do cell_id
        cell = params[cell_id]
        Qch = (1 - cell[:soc]) * cell[:Q]
    end |> minimum

    Qch + Qdch
end

function calc_soh_pack(params, Q; delta_soc=true)
    if delta_soc
        # Qloss due to degradation + Δsoc
        Q_pack = calc_Q_pack(params)
    else
        # Qloss only from degradation
        Q_pack = minimum(params[cell_id][:Q] for cell_id in keys(params))
    end

    Q_pack / Q
end

function calc_Q_utilization(params; delta_soc=true)
    n_cells = length(params)
    Q_cells_total = sum(params[cell_id][:Q] for cell_id in keys(params))

    if delta_soc
        # Qloss due to degradation + Δsoc
        Q_pack = calc_Q_pack(params)
    else
        # Qloss only from degradation
        Q_pack = minimum(params[cell_id][:Q] for cell_id in keys(params))
    end
    (Q_pack * n_cells) / Q_cells_total
end

function calc_soc_pack(df, params)
    # TODO: improve

    df2 = DataFrame(
        "t" => df.t,
        ["Qdch$i" => df[:, "soc_cell_$i"] * params[Symbol("cell_$i")][:Q] for i in 1:12]...,
        ["Qch$i" => (1 .- df[:, "soc_cell_$i"]) * params[Symbol("cell_$i")][:Q] for i in 1:12]...,
    )

    Q_pack_dch = minimum.(eachrow(df2[:, ["Qdch$i" for i in 1:12]]))
    Q_pack_ch = minimum.(eachrow(df2[:, ["Qch$i" for i in 1:12]]))

    Q_pack = Q_pack_ch + Q_pack_dch
    Q_pack_dch ./ Q_pack
end

function plot_module_soc(df, params)
    # TODO: improve

    fig = Figure()
    ax = Axis(fig[1, 1])

    for i in 1:12
        lines!(ax, df.t / 3600, df[:, "soc_cell_$i"], color=(:blue, 0.2), label="Cell")
    end

    S_pack = calc_soc_pack(df, params)
    lines!(ax, df.t / 3600, S_pack, color=:black, label="Module")


    axislegend(ax, position=:lb, merge=true)
    xlims!(ax, df[begin, :t] / 3600, df[end, :t] / 3600)
    ax.ylabel = "SOC / p.u."
    ax.xlabel = "Time / h"

    fig
end
