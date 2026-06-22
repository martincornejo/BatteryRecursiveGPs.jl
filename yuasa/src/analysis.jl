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
    trajs = map(enumerate(ids)) do (k, id)
        (; t, q, qσ) = charge_trajectory(models[id], sols[id])
        fq = LinearInterpolation(Measurements.measurement.(q, qσ), t; extrapolation = ExtrapolationType.Constant)
        fit.s0[k] .+ fq.(tg) ./ fit.Q_cell[k]
    end
    return stack(trajs)  # length(tg) × length(ids), eltype Measurement{Float64}
end

function calc_module_soc(id, tg, soc_cell, soc_module, cell_fit, cell_ids, module_ids)
    (; p, m) = id
    cell_idx = Dict(cell_ids .=> eachindex(cell_ids))
    idx = [cell_idx[(; p, m, c)] for c in 1:12]
    k = findfirst(==(id), module_ids)
    Q = Measurements.value.(cell_fit.Q_cell[idx])
    soc_c = soc_cell[:, idx]
    return DataFrame(
        "t" => collect(tg),
        ["soc_cell_$c" => soc_c[:, i] for (c, i) in enumerate(1:12)]...,
        "soc_pack" => calc_soc_pack(Q, soc_c),
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

# charge-estimation accuracy on the reference module (only P1M9 carries an oscilloscope current
# probe → accurate ground-truth charge). The probe measures one shared string current, so q_ref is
# common to all 12 cells and only the per-cell voltage differs: 12 estimates against one reference.
# `ids` are the reference-module cells; `Qs` their capacities (aligned to `ids`). Error is reported
# as SOC % (charge / capacity, the downstream quantity and field-standard metric); absolute Ah and
# the module-current Coulomb-counting baseline are kept for reference.
function calc_charge_accuracy(soc_models, soc_sols, data, ti, ids, Qs; zt = fit_zscore())
    map(ids, Qs) do id, Q
        sol = soc_sols[id]
        (; q) = charge_trajectory(soc_models[id], sol)                                   # EKF estimate [Ah]
        qref = StatsBase.reconstruct(zt.q, [ui.q for ui in cell_dataset_osci(data, ti, id.c).u])[sol.idx]
        q_cc = StatsBase.reconstruct(zt.q, [u.q for u in sol.ut])                         # module-current CC [Ah]
        e, e_cc = q .- qref, q_cc .- qref
        rmse = sqrt(mean(abs2, e))
        (; id.p, id.m, id.c, Q,
           rmse_soc = 100rmse / Q, max_soc = 100maximum(abs, e) / Q,                     # %SOC
           rmse, max = maximum(abs, e), rmse_cc = sqrt(mean(abs2, e_cc)))                 # Ah
    end |> DataFrame
end

# time-resolved charge error as SOC %, interpolated onto the common grid `tg` (one column per cell).
function calc_charge_error(soc_models, soc_sols, data, ti, ids, Qs; tg, zt = fit_zscore())
    df = DataFrame(t = collect(tg) ./ 3600)
    for (id, Q) in zip(ids, Qs)
        (; t, q) = charge_trajectory(soc_models[id], soc_sols[id])
        qref = StatsBase.reconstruct(zt.q, [ui.q for ui in cell_dataset_osci(data, ti, id.c).u])[soc_sols[id].idx]
        f = LinearInterpolation(100 .* (q .- qref) ./ Q, t; extrapolation = ExtrapolationType.Constant)
        df[!, "c$(id.c)"] = f.(tg)
    end
    return df
end

# fault-rejection diagnostic for one reference cell: run the SOC EKF under injected faults — an
# initial-SOC error (`offset`, Ah) and a constant current-sensor bias (`bias`, A) — and return,
# per scenario, the charge trajectories vs the oscilloscope reference and module-current Coulomb
# counting. Faults are in physical units so the Coulomb-counting error reads directly off the axis.
function calc_soc_diagnostic(model, sol, data, ti, c, scenarios; θ, Ts = 1.0, zt = model.kf.p.zt)
    scale = only(zt.q.scale)  # shared i/q z-score scale → integrate current in z-scored units
    q_ref = StatsBase.reconstruct(zt.q, [ui.q for ui in cell_dataset_osci(data, ti, c).u])
    map(scenarios) do (; offset, bias)
        bias_z = bias / scale
        # inject the bias into the input current and accumulate it into the charge channel so the
        # module-current Coulomb-counting reference (sol.ut.q) drifts with it
        u = [(; i = ui.i + bias_z, q = ui.q + bias_z * (k / 3600), ui.T) for (k, ui) in enumerate(sol.u)]
        sm = RCGPStateModel(model; q0 = offset / scale, Ts, θ)
        s = reduce_sol(sm, run_kf!(sm, u, sol.y))
        t = s.idx
        (; offset, bias, t = t ./ 3600, q_ref = q_ref[t],
           q_cc = StatsBase.reconstruct(zt.q, [v.q for v in s.ut]) .+ offset,  # CC inherits the offset
           q = StatsBase.reconstruct(zt.q, s.qμ),
           qσ = sqrt.(s.qσ) .* scale)
    end
end


# total charge throughput (Ah) over a module's current trace
function calc_throughput(data, p, m)
    df_i = select(data[:module_current], "_time" => "time", "module_average_current_$(p)_$(m)" => "value")
    dt = [Dates.value.(diff(df_i.time)) * 1.0e-3; 0]
    return sum(abs, df_i.value .* dt) / 3600
end


# === data availability ===

# how much data is actually available: per signal table, the fraction of expected samples
# present over `ti`. Each table shares one `_time` column but its channels carry missings,
# so we count non-missing values across all channels against the count expected at the
# nominal sampling rate. Complements the median-timestep view in `plot_data_resolution`.
function calc_data_completeness(data, ti)
    nominal_dt = Dict(  # nominal sampling period [s] per signal table
        :cell_voltage => 10.0,
        :module_voltage => 1.0,
        :module_current => 1.0,
        :battery_temperature => 15.0,
    )
    dur_s = (last(ti) - first(ti)) / Millisecond(1000)
    map(collect(keys(nominal_dt))) do sig
        df = filter(:_time => ∈(ti), data[sig])
        chans = names(df, Not(:_time))
        expected = (floor(Int, dur_s / nominal_dt[sig]) + 1) * length(chans)
        actual = sum(c -> count(!ismissing, df[!, c]), chans)
        (; signal = sig, n_channels = length(chans), actual, expected, completeness = actual / expected)
    end |> DataFrame
end
