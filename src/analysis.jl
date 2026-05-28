# TODO: rename functions

# === Private helpers for multi-model dispatch ===

# Returns the KF that holds the GP posterior (OCV/R0 hyperparameters).
# For state-only models the GP lives in the inner full-model KF, not the 2-state EKF.
_gp_kf(model::AbstractBatteryModel) = model.kf
_gp_kf(model::YuasaStateModel) = model.kf.p.kf
_gp_kf(model::FeneconStateModel) = model.kf.p.kf
_gp_kf(model::Fenecon2RCStateModel) = model.kf.p.kf
_gp_kf(model::RCGPStateModel) = model.kf.p.kf
_gp_kf(model::RCGP2RCStateModel) = model.kf.p.kf

# Returns the normalised charge range (q̂min, q̂max) from a KF solution.
_charge_range(::AbstractBatteryModel, sol) = extrema(sol.qμ)

# State-only models store SA[q̂, vrc, ...] — charge is the first element.
_charge_range(::YuasaStateModel, sol) = extrema(first.(sol.xt))
_charge_range(::FeneconStateModel, sol) = extrema(first.(sol.xt))
_charge_range(::Fenecon2RCStateModel, sol) = extrema(first.(sol.xt))
_charge_range(::RCGPStateModel, sol) = extrema(first.(sol.xt))
_charge_range(::RCGP2RCStateModel, sol) = extrema(first.(sol.xt))


function calc_deltaq(model::AbstractBatteryModel, sol; v = (3.85, 4.0), n = 1)
    kf = _gp_kf(model)
    v1 = v[1] * n
    v2 = v[2] * n

    zt = kf.p.zt

    q̂min, q̂max = _charge_range(model, sol)
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

    return q2 - q1
end

function calc_Q(model::AbstractBatteryModel, sol, fsoc; v = (3.85, 4.0), n = 1)
    v1, v2 = v

    Δsoc = fsoc(v2) - fsoc(v1)
    Δq = calc_deltaq(model, sol; v, n)
    return Δq / (Δsoc)
end

function calc_soh(model::AbstractBatteryModel, sol, fsoc, Q; v = (3.85, 4.0), n = 1)
    Q´ = calc_Q(model, sol, fsoc; v, n)
    return Q´ / Q
end

function calc_soc0(model::AbstractBatteryModel, sol, fsoc; v = (3.85, 4.0), n = 1)
    kf = _gp_kf(model)
    v1 = v[1] * n

    zt = kf.p.zt

    q̂min, q̂max = _charge_range(model, sol)
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
    return s0 - Δs
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
    F = eigen(Symmetric(Σ))
    τ = sqrt(eps(eltype(Σ))) * maximum(F.values)
    keep = F.values .> τ
    # Whitening transform: maps to uncorrelated unit-variance observations
    W = Diagonal(1 ./ sqrt.(F.values[keep])) * F.vectors[:, keep]'
    ỹ = W * y
    X̃ = W * X
    # OLS on the whitened system — X̃'X̃ is 2×2 and well-conditioned
    β = (X̃' * X̃) \ (X̃' * ỹ)
    Σβ = inv(X̃' * X̃)
    return (; β, Σβ)
end

"""
    calc_wls(model, sol, fsoc, focv; n_grid=100, Q_range=40:0.5:160)

Estimate cell capacity `Q` and initial SOC `s0` via **profile search**.

**Why profile search**: the joint (Q, s0) objective `‖μ_v(q) − focv(s0+q/Q)‖²`
is ill-conditioned because the Jacobian columns (s0 direction and 1/Q direction)
are nearly collinear when the observed charge range is small relative to Q.
Newton-type methods starting away from the global minimum diverge or find wrong
local minima. The 1D *profile* objective — minimising over s0 analytically for
each fixed Q — is unimodal and well-behaved.

**Algorithm**:
1. Coarse sweep over `Q_range` (step 0.5 Ah default): for each Q compute the
   closed-form optimal `s0 = mean(fsoc(μ_v) − q/Q)` and the RMS voltage residual.
2. Fine sweep ±2 Ah around the coarse minimum at 0.05 Ah step.
3. At Q*, compute uncertainties from one GLS pass (Σβ of the linearised problem).

Returns `(; Q, s0)` as `Measurements.jl` objects.
"""
function calc_wls(model::AbstractBatteryModel, sol, fsoc, focv; n_grid = 100, Q_range = 40:0.5:160)
    kf = _gp_kf(model)
    zt = kf.p.zt

    # Charge range from the actual KF state trajectory.
    q̂min, q̂max = _charge_range(model, sol)

    q̂ = collect(range(q̂min, q̂max, n_grid))
    q = StatsBase.reconstruct(zt.q, q̂)

    # GP prediction: mean and full covariance in normalised voltage units.
    ocv = predict_gp(kf, q̂, :ocv)
    μ_v = StatsBase.reconstruct(zt.v, ocv.μ)

    # Filter to the fsoc interpolation domain (avoid extrapolation).
    fsoc_lims = extrema(fsoc.t)
    focv_lims = extrema(focv.t)
    v_low, v_up = extrema(μ_v)
    v_low = max(fsoc_lims[1], v_low)
    v_up = min(fsoc_lims[2], v_up)
    idxs = findall(v_low .<= μ_v .<= v_up)

    q_filt = q[idxs]
    μ_filt = μ_v[idxs]
    Σ_v = ocv.Σ[idxs, idxs]
    scale_v = zt.v.scale[1]
    soc_gp = fsoc.(μ_filt)           # reference SOC at each GP voltage

    # Profile residual for a fixed Q: optimal s0 in closed form, RMSE in mV.
    function _profile(Q)
        s0 = mean(soc_gp .- q_filt ./ Q)
        soc_m = s0 .+ q_filt ./ Q
        valid = findall(focv_lims[1] .<= soc_m .<= focv_lims[2])
        isempty(valid) && return Inf
        return sqrt(mean(abs2, μ_filt[valid] .- focv.(soc_m[valid]))) * 1000
    end

    # Coarse search
    Q_coarse = first(Q_range)
    best_coarse = Inf
    for Q in Q_range
        r = _profile(Q)
        if r < best_coarse
            best_coarse = r
            Q_coarse = Q
        end
    end

    # Fine search ±2 Ah around coarse minimum
    Q_fine = (Q_coarse - 2.0):0.05:(Q_coarse + 2.0)
    Q_star = Q_coarse
    best_fine = best_coarse
    for Q in Q_fine
        r = _profile(Q)
        if r < best_fine
            best_fine = r
            Q_star = Q
        end
    end
    s0_star = mean(soc_gp .- q_filt ./ Q_star)

    # --- Uncertainties via one GLS pass at Q_star ---
    soc_model = s0_star .+ q_filt ./ Q_star
    uidxs = findall(focv_lims[1] .<= soc_model .<= focv_lims[2])
    soc_u = soc_model[uidxs]
    q_u = q_filt[uidxs]
    v̂_norm = StatsBase.transform(zt.v, focv.(soc_u))
    r = ocv.μ[idxs][uidxs] .- v̂_norm
    dfocv_ds = DataInterpolations.derivative.(Ref(focv), soc_u)
    A = (dfocv_ds ./ scale_v) .* [ones(length(q_u)) q_u]
    fit = gls_fit(r, A, Σ_v[uidxs, uidxs])

    inv_Q = (1 / Q_star) ± sqrt(fit.Σβ[2, 2])
    s0_m = s0_star ± sqrt(fit.Σβ[1, 1])

    return (; Q = 1 / inv_Q, s0 = s0_m)
end

"""
    gp_ocv(model, sol)

Return the GP OCV posterior over the observed charge range as `(; q, μ, σ)`
in physical units (Ah, V, V).
"""
function gp_ocv(model::AbstractBatteryModel, sol)
    kf = _gp_kf(model)
    zt = kf.p.zt

    q̂min, q̂max = _charge_range(model, sol)
    q̂ = collect(q̂min:0.01:q̂max)
    q = StatsBase.reconstruct(zt.q, q̂)

    ocv = predict_gp(kf, q̂, :ocv)
    μ = StatsBase.reconstruct(zt.v, ocv.μ)
    σ = StatsBase.reconstruct(zt.σ, sqrt.(diag(ocv.Σ)))

    return (; q, μ, σ)
end

"""
    gp_r0(model, sol)

Return the GP R0 posterior over the observed charge range as `(; q, μ, σ)`
in physical units (Ah, Ω, Ω).
"""
function gp_r0(model::AbstractBatteryModel, sol)
    kf = _gp_kf(model)
    zt = kf.p.zt

    q̂min, q̂max = _charge_range(model, sol)
    q̂ = collect(q̂min:0.01:q̂max)
    q = StatsBase.reconstruct(zt.q, q̂)

    r0 = predict_gp(kf, q̂, :r0)
    μ = StatsBase.reconstruct(zt.r, r0.μ)
    σ = StatsBase.reconstruct(zt.r, sqrt.(diag(r0.Σ)))

    return (; q, μ, σ)
end

"""
    gp_r1(model, sol)

Return the GP R1 posterior over the observed charge range as `(; q, μ, σ)`
in physical units (Ah, Ω, Ω). For models with a 2-RC layout, `gp_r2` returns
the second branch's posterior.
"""
function gp_r1(model::AbstractBatteryModel, sol)
    kf = _gp_kf(model)
    zt = kf.p.zt

    q̂min, q̂max = _charge_range(model, sol)
    q̂ = collect(q̂min:0.01:q̂max)
    q = StatsBase.reconstruct(zt.q, q̂)

    r1 = predict_gp(kf, q̂, :r1)
    μ = StatsBase.reconstruct(zt.r, r1.μ)
    σ = StatsBase.reconstruct(zt.r, sqrt.(diag(r1.Σ)))

    return (; q, μ, σ)
end

function gp_r2(model::AbstractBatteryModel, sol)
    kf = _gp_kf(model)
    zt = kf.p.zt

    q̂min, q̂max = _charge_range(model, sol)
    q̂ = collect(q̂min:0.01:q̂max)
    q = StatsBase.reconstruct(zt.q, q̂)

    r2 = predict_gp(kf, q̂, :r2)
    μ = StatsBase.reconstruct(zt.r, r2.μ)
    σ = StatsBase.reconstruct(zt.r, sqrt.(diag(r2.Σ)))

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

    return Qch + Qdch
end

function calc_soh_pack(params, Q; delta_soc = true)
    if delta_soc
        # Qloss due to degradation + Δsoc
        Q_pack = calc_Q_pack(params)
    else
        # Qloss only from degradation
        Q_pack = minimum(params[cell_id][:Q] for cell_id in keys(params))
    end

    return Q_pack / Q
end

function calc_Q_utilization(params; delta_soc = true)
    n_cells = length(params)
    Q_cells_total = sum(params[cell_id][:Q] for cell_id in keys(params))

    if delta_soc
        # Qloss due to degradation + Δsoc
        Q_pack = calc_Q_pack(params)
    else
        # Qloss only from degradation
        Q_pack = minimum(params[cell_id][:Q] for cell_id in keys(params))
    end
    return (Q_pack * n_cells) / Q_cells_total
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
    return Q_pack_dch ./ Q_pack
end

function plot_module_soc(df, params)
    # TODO: improve

    fig = Figure()
    ax = Axis(fig[1, 1])

    for i in 1:12
        lines!(ax, df.t / 3600, df[:, "soc_cell_$i"], color = (:blue, 0.2), label = "Cell")
    end

    S_pack = calc_soc_pack(df, params)
    lines!(ax, df.t / 3600, S_pack, color = :black, label = "Module")


    axislegend(ax, position = :lb, merge = true)
    xlims!(ax, df[begin, :t] / 3600, df[end, :t] / 3600)
    ax.ylabel = "SOC / p.u."
    ax.xlabel = "Time / h"

    return fig
end
