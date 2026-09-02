# Two-stage GP hyperparameter selection. Stage 1 fits every unit (cell or module) at a
# generic init ϑ and builds a composite-OCV reference from the units that already match it;
# stage 2 re-fits the misfits over the ℓ grids and keeps, per unit, the ℓ minimising the
# composite residual. Fits are farmed out to `pool` as `fit_ocv_curve` — θ is built here on
# the master, so workers need nothing but BatteryRecursiveGPs.

# closure scoring an OCV curve against a fixed reference composite → RMSE in mV
function make_scorer(comp_ref)
    order_soc = sortperm(comp_ref.soc_grid)
    order_v = sortperm(comp_ref.v_grid)
    flat = ExtrapolationType.Constant
    v_of_soc = LinearInterpolation(comp_ref.v_grid[order_soc], comp_ref.soc_grid[order_soc]; extrapolation = flat)
    soc_of_v = LinearInterpolation(comp_ref.soc_grid[order_v], comp_ref.v_grid[order_v]; extrapolation = flat)
    v_range = extrema(comp_ref.v_grid)
    soc_lo, soc_hi = first(comp_ref.soc_grid[order_soc]), last(comp_ref.soc_grid[order_soc])
    function score(ocv)
        aligned = fit_cells_to_reference([(; ocv.q, ocv.μ)], soc_of_v, v_range)
        Q = Measurements.value(aligned.Q_cell[1])
        s0 = Measurements.value(aligned.s0[1])
        soc = ocv.q ./ Q .+ s0
        mask = (soc .>= soc_lo) .& (soc .<= soc_hi)
        return sqrt(mean(abs2, (ocv.μ[mask] .- v_of_soc.(soc[mask])) .* 1000))
    end
    return score
end

# ϑ with the two selected length scales replaced
_with_ℓ(ϑ, ℓ_ocv, ℓ_r1) =
    merge(ϑ, (; ocv = merge(ϑ.ocv, (; ℓ = ℓ_ocv)), r1 = merge(ϑ.r1, (; ℓ = ℓ_r1))))

# Fit one unit's OCV curve on a worker. θ is built here on the master, so the worker needs
# nothing but BatteryRecursiveGPs.
_fit_ocv(pool, d, ϑ, zt, n) =
    remotecall(fit_ocv_curve, pool, YuasaModel, d.u, d.y, scale_θ(d.u, d.y, ϑ; n), zt)

# Stage 1: fit every unit at ϑ. Any worker failure crashes — the init is a known-good
# config, so a failure here is a real bug.
function fit_units_init(pool, ids, unit_data, ϑ, zt; n = 1)
    tasks = Dict(id => _fit_ocv(pool, unit_data[id], ϑ, zt, n) for id in ids)
    return Dict(id => fetch(task) for (id, task) in tasks)
end

# Stage 2 + pick: per-unit argmin RMSE over `(ℓ_ocv_init, ℓ_ocv_grid...)`. Starts from each
# unit's init curve; misfit units (rmse > thresh_mV against `comp_ref`) additionally fit each
# grid ℓ in parallel and upgrade the pick when one beats the current best. Individual
# escalation-ℓ failures are tolerated (logged + skipped).
# Returns Dict{id, (; ℓ_ocv, ℓ_r1, rmse, rmse_init, curve)} for every unit.
function escalate_units(
        curves_init, comp_ref, pool, unit_data, zt;
        ϑ, ℓ_ocv_grid, ℓ_r1_grid = Float64[], thresh_mV, n = 1
    )
    score = make_scorer(comp_ref)

    picks = Dict()
    for (id, curve) in curves_init
        rmse = score(curve)
        picks[id] = (; ℓ_ocv = ϑ.ocv.ℓ, ℓ_r1 = ϑ.r1.ℓ, rmse, rmse_init = rmse, curve)
    end

    misfits = sort([id for (id, pick) in picks if pick.rmse > thresh_mV])
    challengers = [
        (ℓ_ocv, ℓ_r1)
            for ℓ_ocv in [ϑ.ocv.ℓ; ℓ_ocv_grid], ℓ_r1 in [ϑ.r1.ℓ; ℓ_r1_grid]
            if (ℓ_ocv, ℓ_r1) != (ϑ.ocv.ℓ, ϑ.r1.ℓ)
    ]

    tasks = Dict(
        (id, ℓ_ocv, ℓ_r1) => _fit_ocv(pool, unit_data[id], _with_ℓ(ϑ, ℓ_ocv, ℓ_r1), zt, n)
            for id in misfits, (ℓ_ocv, ℓ_r1) in challengers
    )

    for id in misfits, (ℓ_ocv, ℓ_r1) in challengers
        best = picks[id]
        try
            curve = fetch(tasks[(id, ℓ_ocv, ℓ_r1)])
            rmse = score(curve)
            rmse < best.rmse && (picks[id] = (; ℓ_ocv, ℓ_r1, rmse, best.rmse_init, curve))
        catch e
            @warn "escalation fit failed: id=$id ℓ_ocv=$ℓ_ocv ℓ_r1=$ℓ_r1" exception = e
        end
    end

    return picks
end

"""
    select_hyperparams(pool, ids, unit_data, zt; ϑ, ℓ_ocv_grid, ℓ_r1_grid = Float64[],
                       thresh_mV, n = 1, n_v_pair = 20) -> (; curves_init, comp_ref, picks, ids, ϑ, thresh_mV, n)

Choose a GP length scale per unit for one level, cells or modules. Every unit is first fitted
at the shared init `ϑ`; units whose OCV then sits further than `thresh_mV` from the reference
composite are refitted over `ℓ_ocv_grid` × `ℓ_r1_grid` on `pool`, keeping whichever ℓ lands
closest. `n` is the number of series cells.

The reference composite is built coarse, filtered to the units within `thresh_mV`, then
refitted, so badly misfit units cannot bias the target they are scored against.
"""
function select_hyperparams(
        pool, ids, unit_data, zt;
        ϑ, ℓ_ocv_grid, ℓ_r1_grid = Float64[], thresh_mV, n = 1, n_v_pair = 20
    )
    curves_init = fit_units_init(pool, ids, unit_data, ϑ, zt; n)

    comp_coarse = fit_composite_ocv(values(curves_init); uq = false, n_v_pair)
    coarse_score = make_scorer(comp_coarse)
    inliers = [c for c in values(curves_init) if coarse_score(c) <= thresh_mV]
    comp_ref = fit_composite_ocv(inliers; uq = false, n_v_pair)

    picks = escalate_units(
        curves_init, comp_ref, pool, unit_data, zt;
        ϑ, ℓ_ocv_grid, ℓ_r1_grid, thresh_mV, n
    )

    return (; curves_init, comp_ref, picks, ids, ϑ, thresh_mV, n)
end

# unit id → compact label, e.g. "P1M9C2" (cell) or "P1M9" (module)
unit_name(id) = hasproperty(id, :c) ? "P$(id.p)M$(id.m)C$(id.c)" : "P$(id.p)M$(id.m)"

# unit id → JSON key, e.g. "1_9_2" (cell) or "1_9" (module)
id_key(id) = hasproperty(id, :c) ? "$(id.p)_$(id.m)_$(id.c)" : "$(id.p)_$(id.m)"

"""
    build_hyperparam_export(sel) -> Dict

The selected dimensionless ℓ/σ per unit, keyed for JSON. Together with `default_θ`, these four
numbers are all `scale_θ` needs to rebuild the full θ.
"""
function build_hyperparam_export(sel)
    return Dict(
        id_key(id) => Dict(
                "ocv_ell" => pick.ℓ_ocv, "ocv_sigma" => sel.ϑ.ocv.σ,
                "r1_ell" => pick.ℓ_r1, "r1_sigma" => sel.ϑ.r1.σ,
            )
            for (id, pick) in sel.picks
    )
end

"""
    load_hyperparams(file, ids) -> Dict{id, ϑ}

Read back what [`build_hyperparam_export`](@ref) wrote, as `(; ocv = (; ℓ, σ),
r1 = (; ℓ, σ))` per unit — the `ϑ` that [`scale_θ`](@ref) expects.
"""
function load_hyperparams(file, ids)
    json = JSON.parsefile(file, Dict{String, Dict{String, Any}})
    ϑ = map(ids) do id
        entry = json[id_key(id)]
        return (;
            ocv = (; ℓ = entry["ocv_ell"], σ = entry["ocv_sigma"]),
            r1 = (; ℓ = entry["r1_ell"], σ = entry["r1_sigma"]),
        )
    end
    return Dict(ids .=> ϑ)
end

"""
    selection_counts(cells, modules) -> DataFrame

How many units ended on each length scale, one row per ℓ and one column per level and curve
(`cells_ocv`, `modules_ocv`, `cells_r1`, `modules_r1`).
"""
function selection_counts(cells, modules)
    cnt(f, ps) = countmap([f(p) for p in values(ps)])
    cols = [
        "cells_ocv" => cnt(p -> p.ℓ_ocv, cells.picks), "modules_ocv" => cnt(p -> p.ℓ_ocv, modules.picks),
        "cells_r1" => cnt(p -> p.ℓ_r1, cells.picks), "modules_r1" => cnt(p -> p.ℓ_r1, modules.picks),
    ]
    ls = sort(unique(reduce(vcat, collect.(keys.(last.(cols))))))
    return DataFrame("ℓ" => ls, (name => [get(c, l, 0) for l in ls] for (name, c) in cols)...)
end

"""
    calc_scaled_hyperparams(sel, unit_data, zt) -> DataFrame

The selected hyperparameters of every unit in physical units: `ℓ_ocv`/`ℓ_r1` in Ah, and
`σ_ocv`/`σ_r1` as the prior standard deviation per cell in mV and mΩ. `ℓ_ocv_rel`/`ℓ_r1_rel`
carry the dimensionless values as selected, before scaling.

Each unit's ℓ is scaled by its own observed charge span, so units that ended on the same
`ℓ_*_rel` still differ in Ah.
"""
function calc_scaled_hyperparams(sel, unit_data, zt)
    (; picks, ids, ϑ, n) = sel
    return map(ids) do id
        pick = picks[id]
        unit = unit_data[id]
        θ = scale_θ(unit.u, unit.y, _with_ℓ(ϑ, pick.ℓ_ocv, pick.ℓ_r1); n)
        (;
            name = unit_name(id),
            ℓ_ocv = θ.ocv.ℓ * zt.q.scale[1], ℓ_r1 = θ.r1.ℓ * zt.q.scale[1],
            σ_ocv = sqrt(θ.ocv.σ) * zt.σ.scale[1] / n * 1000, σ_r1 = sqrt(θ.r1.σ) * zt.r.scale[1] / n * 1000,
            ℓ_ocv_rel = pick.ℓ_ocv, ℓ_r1_rel = pick.ℓ_r1,
        )
    end |> DataFrame
end

# --- selection-figure data ---

function compcurve(comp)
    order_soc = sortperm(comp.soc_grid)
    return collect(comp.soc_grid)[order_soc], collect(comp.v_grid)[order_soc]
end

# SOC in %, derivative in mV per % — the units used for dV/dSOC elsewhere in the paper.
# 1 V per unit SOC fraction = 1000 mV / 100 % = 10 mV/%.
function dvdsoc(soc, v)
    order_soc = sortperm(soc)
    soc_pct = soc[order_soc] .* 100
    v_sorted = v[order_soc]
    midpoints = (soc_pct[1:(end - 1)] .+ soc_pct[2:end]) ./ 2
    return midpoints, diff(v_sorted) ./ diff(soc_pct) .* 1000
end

# align each unit's (q, μ) to a composite's SOC axis → vector of (; soc, μ) in `ids` order
function align_units(curves_by_id, ids, comp)
    order_v = sortperm(comp.v_grid)
    soc_of_v = LinearInterpolation(comp.soc_grid[order_v], comp.v_grid[order_v]; extrapolation = ExtrapolationType.Constant)
    ocvs = [curves_by_id[id] for id in ids]
    aligned = fit_cells_to_reference([(; o.q, o.μ) for o in ocvs], soc_of_v, extrema(comp.v_grid))
    return map(eachindex(ocvs)) do k
        Q = Measurements.value(aligned.Q_cell[k])
        s0 = Measurements.value(aligned.s0[k])
        return (; soc = ocvs[k].q ./ Q .+ s0, μ = ocvs[k].μ)
    end
end

"""
    calc_hyperparam_selection(sel) -> (; fans, comp, rmse, thresh)

Plot-ready view of one [`select_hyperparams`](@ref) run, scaled per cell so cells and modules
are comparable. `fans` holds every unit's dV/dSOC curve aligned to the reference composite,
once at `:init` and once `:adapted`, with `flagged` marking the units that missed
`thresh_mV`. `comp` is the reference composite, trimmed at the steep edges where the
numerical derivative blows up, and `rmse` the composite-OCV residual per unit in both stages.
"""
function calc_hyperparam_selection(sel)
    (; curves_init, picks, comp_ref, ids, thresh_mV, n) = sel
    curves_final = Dict(id => picks[id].curve for id in ids)

    rows = NamedTuple[]
    for (stage, curves) in ((:init, curves_init), (:adapted, curves_final))
        aligned = align_units(curves, ids, comp_ref)
        for (k, id) in enumerate(ids)
            soc, dv = dvdsoc(aligned[k].soc, aligned[k].μ ./ n)
            push!(rows, (; name = unit_name(id), stage, soc, dvdsoc = dv, flagged = picks[id].rmse_init > thresh_mV))
        end
    end

    sc, vc = compcurve(comp_ref)
    s, dv = dvdsoc(sc, vc ./ n)
    edge = (s .> 0.01) .& (s .< 0.995)

    rmse = DataFrame(
        name = [(hasproperty(id, :c) ? "Cell " : "Module ") * unit_name(id) for id in ids],
        init = [picks[id].rmse_init / n for id in ids],
        adapted = [picks[id].rmse / n for id in ids],
    )

    return (; fans = DataFrame(rows), comp = DataFrame(soc = s[edge], dvdsoc = dv[edge]), rmse, thresh = thresh_mV / n)
end
