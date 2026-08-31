# TODO: rename functions

# === Private helpers for multi-model dispatch ===

# Returns the KF that holds the GP posterior (OCV/R0 hyperparameters).
# For state-only models the GP lives in the inner full-model KF, not the 2-state EKF.
_gp_kf(model::AbstractBatteryModel) = model.kf
_gp_kf(model::AbstractBatteryStateModel) = model.kf.p.kf

# Returns the normalised charge trajectory from a KF solution.
# Reduced full-model sols carry it as `qμ`; raw state-only sols store
# SA[q̂, vrc, ...] — charge is the first element.
_charge(::AbstractBatteryModel, sol) = sol.qμ
_charge(::AbstractBatteryStateModel, sol) = first.(sol.xt)

# Returns the normalised charge range (q̂min, q̂max) from a KF solution.
_charge_range(model::AbstractBatteryModel, sol) = extrema(_charge(model, sol))


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

"""
    charge_trajectory(model, sol)

Filtered charge estimate over the fit window as `(; t, q)` in physical
units (s, Ah). Charge is relative to the window start.
"""
function charge_trajectory(model::AbstractBatteryModel, sol)
    zt = _gp_kf(model).p.zt
    t = (sol.idx .- 1) .* model.kf.p.Ts
    q = StatsBase.reconstruct(zt.q, _charge(model, sol))
    qσ = sqrt.(sol.qσ) .* zt.q.scale[1]
    return (; t, q, qσ)
end

"""
    voltage_error(model, sol)

Innovation (one-step voltage prediction error) series as `(; t, e)` in
physical units (s, V).
"""
function voltage_error(model::AbstractBatteryModel, sol)
    zt = _gp_kf(model).p.zt
    t = (sol.idx .- 1) .* model.kf.p.Ts
    e = StatsBase.reconstruct(zt.σ, sol.et)
    return (; t, e)
end

calc_Q_pack(Q, soc) = minimum(soc .* Q) + minimum((1 .- soc) .* Q)

function calc_Q_pack(params::AbstractDict)
    Q = [cell[:Q] for cell in values(params)]
    soc = [cell[:soc] for cell in values(params)]
    return calc_Q_pack(Q, soc)
end

function calc_soh_pack(Q, soc, Q_nom; delta_soc = true)
    # delta_soc: Qloss due to degradation + Δsoc; otherwise degradation only
    Q_pack = delta_soc ? calc_Q_pack(Q, soc) : minimum(Q)
    return Q_pack / Q_nom
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

"""
    calc_soc_pack(Q, soc)

Pack SOC from per-cell capacities `Q` and SOC trajectories `soc`
(one row per time step, one column per cell).
"""
function calc_soc_pack(Q, soc::AbstractMatrix)
    Q_dch = vec(minimum(soc .* Q'; dims = 2))
    Q_ch = vec(minimum((1 .- soc) .* Q'; dims = 2))
    return Q_dch ./ (Q_dch .+ Q_ch)
end
