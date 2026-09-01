# Two-stage GP hyperparameter selection. Stage 1 fits every unit (cell or module) at a
# generic init ϑ and builds a composite-OCV reference from the units that already match it;
# stage 2 re-fits the misfits over the ℓ grids and keeps, per unit, the ℓ minimising the
# composite residual. Fits are farmed out to `pool` as `fit_ocv_curve` — θ is built here on
# the master, so workers need nothing but BatteryRecursiveGPs.

# closure scoring an OCV curve against a fixed reference composite → RMSE in mV
function make_scorer(comp_ref)
    o_s = sortperm(comp_ref.soc_grid)
    o_v = sortperm(comp_ref.v_grid)
    v_of_soc = LinearInterpolation(comp_ref.v_grid[o_s], comp_ref.soc_grid[o_s]; extrapolation = ExtrapolationType.Constant)
    soc_of_v = LinearInterpolation(comp_ref.soc_grid[o_v], comp_ref.v_grid[o_v]; extrapolation = ExtrapolationType.Constant)
    v_range = extrema(comp_ref.v_grid)
    soc_lo, soc_hi = first(comp_ref.soc_grid[o_s]), last(comp_ref.soc_grid[o_s])
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

# Stage 1: fit every unit at ϑ. Any worker failure crashes — the init is a known-good
# config, so a failure here is a real bug.
function fit_units_init(pool, ids, unit_data, ϑ, zt; n = 1)
    tasks = Dict(
        id => remotecall(
                fit_ocv_curve, pool, YuasaModel,
                unit_data[id].u, unit_data[id].y,
                scale_θ(unit_data[id].u, unit_data[id].y, ϑ; n), zt
            )
            for id in ids
    )
    return Dict(id => fetch(t) for (id, t) in tasks)
end

# Stage 2 + pick: per-unit argmin RMSE over `(ℓ_ocv_init, ℓ_ocv_grid...)`. Starts from each
# unit's init curve; misfit units (rmse > thresh_mV against `comp_ref`) additionally fit each
# grid ℓ in parallel and upgrade the pick when one beats the current best. Individual
# escalation-ℓ failures are tolerated (logged + skipped).
# Returns Dict{id, (; ℓ_ocv, ℓ_r1, rmse_mV, rmse_init, curve, escalated)} for every unit.
function escalate_units(
        curves_init, comp_ref, pool, unit_data, zt;
        ϑ, ℓ_ocv_grid, ℓ_r1_grid = Float64[], thresh_mV, n = 1
    )
    score = make_scorer(comp_ref)

    # initial pick = init curve for every unit
    picks = Dict(
        id => begin
                r = score(curve)
                (; ℓ_ocv = ϑ.ocv.ℓ, ℓ_r1 = ϑ.r1.ℓ, rmse_mV = r, rmse_init = r, curve, escalated = false)
            end for (id, curve) in curves_init
    )

    # misfit units: joint (ocv.ℓ, r1.ℓ) grid, keep argmin vs current pick
    misfits = [id for (id, p) in picks if p.rmse_mV > thresh_mV]
    ℓ_ocv_all = [ϑ.ocv.ℓ; ℓ_ocv_grid]
    ℓ_r1_all = [ϑ.r1.ℓ;  ℓ_r1_grid]
    challengers = [
        (lo, lr) for lo in ℓ_ocv_all, lr in ℓ_r1_all
            if !(lo == ϑ.ocv.ℓ && lr == ϑ.r1.ℓ)
    ]
    tasks = Dict(
        (id, ℓ_ocv, ℓ_r1) =>
            remotecall(
                fit_ocv_curve, pool, YuasaModel, unit_data[id].u, unit_data[id].y,
                scale_θ(
                    unit_data[id].u, unit_data[id].y,
                    merge(
                        ϑ, (;
                            ocv = merge(ϑ.ocv, (; ℓ = ℓ_ocv)),
                            r1 = merge(ϑ.r1, (; ℓ = ℓ_r1)),
                        )
                    ); n
                ), zt
            )
            for id in misfits, (ℓ_ocv, ℓ_r1) in challengers
    )
    for id in misfits
        current = picks[id]
        for (ℓ_ocv, ℓ_r1) in challengers
            try
                curve = fetch(tasks[(id, ℓ_ocv, ℓ_r1)])
                rmse = score(curve)
                if rmse < current.rmse_mV
                    current = (;
                        ℓ_ocv, ℓ_r1, rmse_mV = rmse,
                        current.rmse_init, curve, escalated = true,
                    )
                end
            catch e
                @warn "escalation fit failed: id=$id ℓ_ocv=$ℓ_ocv ℓ_r1=$ℓ_r1" exception = e
            end
        end
        picks[id] = current
    end

    return picks
end

# One full selection run for a level (cells or modules). The reference composite is built
# coarse → outlier-filtered → refit, so a few badly misfit units cannot bias the target the
# escalation is scored against. `comp_final` is the composite after adaptation, from the
# units that end up within the threshold.
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

    keep = [picks[id].curve for id in ids if picks[id].rmse_mV <= thresh_mV]
    comp_final = fit_composite_ocv(keep; uq = false, n_v_pair)

    return (; curves_init, comp_ref, comp_final, picks, ids, ϑ, thresh_mV, n)
end

# unit id → compact label, e.g. "P1M9C2" (cell) or "P1M9" (module)
unit_name(id) = hasproperty(id, :c) ? "P$(id.p)M$(id.m)C$(id.c)" : "P$(id.p)M$(id.m)"

# unit id → JSON key, e.g. "1_9_2" (cell) or "1_9" (module)
id_key(id) = hasproperty(id, :c) ? "$(id.p)_$(id.m)_$(id.c)" : "$(id.p)_$(id.m)"

"""
    build_hyperparam_export(sel) -> Dict

The selected nominal ℓ/σ per unit, keyed for JSON. Together with `default_θ`, these four
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

# marginal counts of the selected nominal length scales — the paper's selection table
function selection_counts(cells, modules)
    cnt(f, ps) = countmap([f(p) for p in values(ps)])
    cols = [
        "cells_ocv" => cnt(p -> p.ℓ_ocv, cells.picks), "modules_ocv" => cnt(p -> p.ℓ_ocv, modules.picks),
        "cells_r1" => cnt(p -> p.ℓ_r1, cells.picks), "modules_r1" => cnt(p -> p.ℓ_r1, modules.picks),
    ]
    ls = sort(unique(reduce(vcat, collect.(keys.(last.(cols))))))
    return DataFrame("ℓ" => ls, (name => [get(c, l, 0) for l in ls] for (name, c) in cols)...)
end

# Data-scaled GP hyperparameters per unit in physical units: length scales in Ah (ℓ from
# scale_θ is in z-scored charge → × zt.q scale), σ as the PRIOR STD per cell in mV (OCV) /
# mΩ (R1) — θ.σ multiplies the kernel, i.e. it is a variance, so the std is √θσ (yuasa.jl).
# The selected nominal values are carried along for the figure's color coding.
function calc_scaled_hyperparams(sel, unit_data, zt)
    (; picks, ids, ϑ, n) = sel
    return map(ids) do id
        p = picks[id]
        d = unit_data[id]
        θ = scale_θ(
            d.u, d.y,
            merge(ϑ, (; ocv = merge(ϑ.ocv, (; ℓ = p.ℓ_ocv)), r1 = merge(ϑ.r1, (; ℓ = p.ℓ_r1)))); n
        )
        (;
            name = unit_name(id),
            ℓ_ocv = θ.ocv.ℓ * zt.q.scale[1], ℓ_r1 = θ.r1.ℓ * zt.q.scale[1],
            σ_ocv = sqrt(θ.ocv.σ) * zt.σ.scale[1] / n * 1000, σ_r1 = sqrt(θ.r1.σ) * zt.r.scale[1] / n * 1000,
            nom_ocv = p.ℓ_ocv, nom_r1 = p.ℓ_r1,
        )
    end |> DataFrame
end

# --- selection-figure data ---

compcurve(comp) = (o = sortperm(comp.soc_grid); (collect(comp.soc_grid)[o], collect(comp.v_grid)[o]))

# SOC in %, derivative in mV per % — the units used for dV/dSOC elsewhere in the paper.
# 1 V per unit SOC fraction = 1000 mV / 100 % = 10 mV/%.
function dvdsoc(soc, v)
    o = sortperm(soc); s = soc[o] .* 100; vv = v[o]
    return ((s[1:(end - 1)] .+ s[2:end]) ./ 2, diff(vv) ./ diff(s) .* 1000)
end

# align each unit's (q, μ) to a composite's SOC axis → vector of (; soc, μ) in `ids` order
function align_units(curves_by_id, ids, comp)
    o_v = sortperm(comp.v_grid)
    soc_of_v = LinearInterpolation(comp.soc_grid[o_v], comp.v_grid[o_v]; extrapolation = ExtrapolationType.Constant)
    ocvs = [curves_by_id[id] for id in ids]
    al = fit_cells_to_reference([(; o.q, o.μ) for o in ocvs], soc_of_v, extrema(comp.v_grid))
    return [(; soc = ocvs[k].q ./ Measurements.value(al.Q_cell[k]) .+ Measurements.value(al.s0[k]), μ = ocvs[k].μ) for k in eachindex(ocvs)]
end

# Plot-ready view of one selection run, per-cell scaled (module curves and residuals ÷ n):
# `fans` are the dV/dSOC curves of every unit aligned to the reference composite, before
# (`:init`) and after (`:adapted`) adaptation, with `flagged` marking the stage-1 misfits;
# `comp` is the reference composite itself, trimmed at the steep edges where the numerical
# derivative blows up; `rmse` is the composite-OCV residual per unit in both stages.
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
        adapted = [picks[id].rmse_mV / n for id in ids],
    )

    return (; fans = DataFrame(rows), comp = DataFrame(soc = s[edge], dvdsoc = dv[edge]), rmse, thresh = thresh_mV / n)
end
