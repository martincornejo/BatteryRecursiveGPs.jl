# === model accuracy ===

function calc_v_summary(models, sols, ids)
    df = map(ids) do id
        (; e) = voltage_error(models[id], sols[id])
        (; id..., rmse = sqrt(mean(abs2, e)) * 1000)
    end |> DataFrame
    return df
end

# module-level comparison on equal footing: the module model is scored against the
# measured module voltage it was fit on; the cell models as a virtual module,
# Σ predicted vs Σ measured cell voltage. Σ cells vs measured module voltage is NOT
# comparable — wiring drops between the two measurements reach several 100 mV.
function calc_module_v_summary(cell_run, module_run, module_ids)
    df = map(module_ids) do id
        (; p, m) = id
        errs = map(1:12) do c
            cid = (; p, m, c)
            sol = cell_run.sols[cid]
            Dict(zip(sol.idx, voltage_error(cell_run.models[cid], sol).e))
        end
        common = sort(collect(intersect([Set(keys(d)) for d in errs]...)))
        e_cells = [sum(d[i] for d in errs) for i in common]
        e_mod = voltage_error(module_run.models[id], module_run.sols[id]).e
        (;
            p, m,
            rmse_module = sqrt(mean(abs2, e_mod)) * 1000,
            rmse_cells = sqrt(mean(abs2, e_cells)) * 1000,
        )
    end |> DataFrame
    return df
end

function calc_v_run_summary(v_runs, cell_ids, module_ids)
    df = map(collect(pairs(v_runs))) do (run, r)
        df_cell = calc_v_summary(r.cell.models, r.cell.sols, cell_ids)
        df_mod = calc_module_v_summary(r.cell, r.mod, module_ids)
        (;
            run,
            cell_med = median(df_cell.rmse),
            cell_q95 = quantile(df_cell.rmse, 0.95), # or 99?
            cell_max = maximum(df_cell.rmse),
            module_med = median(df_mod.rmse_module),
            cells_agg_med = median(df_mod.rmse_cells),
        )
    end |> DataFrame
    return df
end

# === SOH estimation ===

function calc_module_soh_summary(cell_ids, cell_fit, module_ids, module_fit; Q_nom = 100)
    cell_idx = Dict(cell_ids .=> eachindex(cell_ids))
    df_soh = map(enumerate(module_ids)) do (k, id)
        (; p, m) = id
        idx = [cell_idx[(; p, m, c)] for c in 1:12]
        Q = cell_fit.Q_cell[idx]
        soc = cell_fit.s0[idx]
        sohs = Q ./ Q_nom

        # cell-to-cell spread within the module
        Δsoc_max = maximum(soc) - minimum(soc)
        Δsoh_max = maximum(sohs) - minimum(sohs)
        σ_soc = std(soc)
        σ_soh = std(sohs)

        soh = calc_soh_pack(Q, soc, Q_nom)
        soh_module = module_fit.Q_cell[k] / Q_nom

        # unusable capacity (fraction of nominal):
        # SOH spread (irreversible) vs SOC imbalance (recoverable by balancing)
        soh_min = calc_soh_pack(Q, soc, Q_nom; delta_soc = false)
        loss_soh = mean(sohs) - soh_min
        loss_soc = soh_min - soh

        (; p, m, soh, soh_module, Δsoc_max, Δsoh_max, σ_soh, σ_soc, loss_soh, loss_soc)
    end |> DataFrame
    return df_soh
end

# === SOC estimation ===

function calc_soc_trajectories(models, sols, fit, ids; tg)
    socs = map(enumerate(ids)) do (k, id)
        (; t, q) = charge_trajectory(models[id], sols[id])
        fq = LinearInterpolation(q, t; extrapolation = ExtrapolationType.Constant)
        Measurements.value(fit.s0[k]) .+ fq.(tg) ./ Measurements.value(fit.Q_cell[k])
    end
    return stack(socs)  # length(tg) × length(ids)
end

function calc_module_soc(id, tg, soc_cell, soc_module, cell_fit, cell_ids, module_ids)
    (; p, m) = id
    cell_idx = Dict(cell_ids .=> eachindex(cell_ids))
    idx = [cell_idx[(; p, m, c)] for c in 1:12]
    k = findfirst(==(id), module_ids)
    Q = Measurements.value.(cell_fit.Q_cell[idx])
    return DataFrame(
        "t" => collect(tg),
        ["soc_cell_$c" => soc_cell[:, i] for (c, i) in enumerate(idx)]...,
        "soc_pack" => calc_soc_pack(Q, soc_cell[:, idx]),
        "soc_module" => soc_module[:, k],
    )
end

# module SOC error trajectories: e(t) = soc_module − aggregated cell soc, length(tg) × n_modules
function calc_soc_error(soc_cell, soc_module, cell_fit, cell_ids, module_ids)
    cell_idx = Dict(cell_ids .=> eachindex(cell_ids))
    E = map(enumerate(module_ids)) do (k, id)
        (; p, m) = id
        idx = [cell_idx[(; p, m, c)] for c in 1:12]
        Q = Measurements.value.(cell_fit.Q_cell[idx])
        soc_module[:, k] .- calc_soc_pack(Q, soc_cell[:, idx])
    end
    return stack(E)
end

function calc_module_soc_summary(soc_err, module_ids)
    df_soc = map(enumerate(module_ids)) do (k, id)
        e = soc_err[:, k]
        (; id.p, id.m, bias = mean(e), rmse = sqrt(mean(abs2, e)), max = maximum(abs, e))
    end |> DataFrame
    return df_soc
end
