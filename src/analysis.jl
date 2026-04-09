
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


"""
    gls_fit(y, X, Σ)

Generalized least squares: fit `y = X β` with observation covariance `Σ`.
Returns `(; β, Σβ)`.
"""
function gls_fit(y, X, Σ)
    # Eigendecompose Σ and keep only eigenvalues above numerical noise.
    # This avoids forming Σ⁻¹ explicitly (numerically unstable when Σ is
    # near-singular due to correlated GP observations).
    F   = eigen(Symmetric(Σ))
    τ   = sqrt(eps(eltype(Σ))) * maximum(F.values)
    keep = F.values .> τ
    # Whitening transform: maps to uncorrelated unit-variance observations
    W  = Diagonal(1 ./ sqrt.(F.values[keep])) * F.vectors[:, keep]'
    ỹ  = W * y
    X̃  = W * X
    # OLS on the whitened system — X̃'X̃ is 2×2 and well-conditioned
    β  = (X̃' * X̃) \ (X̃' * ỹ)
    Σβ = inv(X̃' * X̃)
    return (; β, Σβ)
end

"""
    calc_wls(model, sol, fsoc; n_grid=100)

Estimate cell capacity `Q` and initial SOC `s0` using a forward GP approach:

1. Evaluate the GP OCV over a dense charge grid of `n_grid` points.
2. Map predicted voltages through the reference `fsoc` to get SOC values.
3. Propagate the full GP posterior covariance through `fsoc` via the Jacobian:
       Σ_soc = J · Σ_ẑᵥ · Jᵀ,  J = Diagonal(∂fsoc/∂v / scale_v)
4. Fit `soc = β[1] + β[2]·q` by GLS using `Σ_soc` as the observation covariance.

The voltage range is derived from the GP mean predictions (extrema of μ_v),
clipped to the `fsoc` interpolation domain. This uses each cell's full observed
voltage range rather than a fixed conservative window.

Returns `(; Q, s0)` as `Measurements.jl` objects.
"""
function calc_wls(model::AbstractBatteryModel, sol, fsoc; n_grid=100)
    kf = model.kf
    zt = kf.p.zt

    # Charge range from the actual KF state trajectory — the GP is data-informed
    # here; outside this window it reverts to its prior and carries no real signal.
    xs = ComponentVector.(sol.xt, kf.p.xid)
    q̂min, q̂max = extrema([x.cc.q for x in xs])

    q̂ = collect(range(q̂min, q̂max, n_grid))
    q  = StatsBase.reconstruct(zt.q, q̂)

    # GP prediction: mean and full covariance over the observed charge range
    ocv  = predict_gp(kf, q̂, :ocv)
    μ_v  = StatsBase.reconstruct(zt.v, ocv.μ)

    # Use the full observed voltage range for this cell, clipped to the fsoc domain
    fsoc_lims    = extrema(fsoc.t)
    v_low, v_up  = extrema(μ_v)
    v_low        = max(fsoc_lims[1], v_low)
    v_up         = min(fsoc_lims[2], v_up)

    idxs = findall(v_low .<= μ_v .<= v_up)

    q_filt  = q[idxs]
    μ_filt  = μ_v[idxs]

    # Reference SOC at each predicted voltage
    soc = fsoc.(μ_filt)

    # Diagonal Jacobian: ∂soc_i/∂ẑᵥ_i = (∂fsoc/∂v_phys) · (∂v_phys/∂ẑᵥ) = dfsoc · scale_v
    # reconstruct(zt.v, v̂) = v̂·scale_v + mean_v  →  ∂v_phys/∂v̂ = scale_v (not 1/scale_v).
    # DataInterpolations.derivative gives the exact analytical derivative of the interpolation,
    # avoiding the extrapolation errors that ForwardDiff.jacobian causes near domain boundaries.
    scale_v  = zt.v.scale[1]
    dfsoc_dv = DataInterpolations.derivative.(Ref(fsoc), μ_filt)
    J        = Diagonal(dfsoc_dv .* scale_v)

    # Propagate GP covariance: Σ_soc = J · Σ_ẑᵥ[idxs, idxs] · Jᵀ
    Σ_soc = J * ocv.Σ[idxs, idxs] * J'

    # GLS fit: soc = β[1] + β[2]·q
    X   = [ones(length(q_filt)) q_filt]
    fit = gls_fit(soc, X, Σ_soc)

    β₂ = fit.β[2] ± sqrt(fit.Σβ[2, 2])
    β₁ = fit.β[1] ± sqrt(fit.Σβ[1, 1])

    Q  = 1 / β₂   # Measurements.jl propagates the uncertainty through 1/β₂
    s0 = β₁

    return (; Q, s0)
end

"""
    gp_ocv(model, sol)

Return the GP OCV posterior over the observed charge range as `(; q, μ, σ)`
in physical units (Ah, V, V).
"""
function gp_ocv(model::AbstractBatteryModel, sol)
    kf = model.kf
    zt = kf.p.zt

    xs   = ComponentVector.(sol.xt, kf.p.xid)
    q̂min, q̂max = extrema([x.cc.q for x in xs])
    q̂   = collect(q̂min:0.01:q̂max)
    q    = StatsBase.reconstruct(zt.q, q̂)

    ocv = predict_gp(kf, q̂, :ocv)
    μ   = StatsBase.reconstruct(zt.v, ocv.μ)
    σ   = StatsBase.reconstruct(zt.σ, sqrt.(diag(ocv.Σ)))

    return (; q, μ, σ)
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
