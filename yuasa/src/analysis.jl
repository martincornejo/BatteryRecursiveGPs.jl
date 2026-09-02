# === model accuracy ===

"""
    calc_v_summary(models, sols, ids) -> DataFrame

Voltage-fit accuracy per unit: the RMSE of the innovation series in mV, one row per id.
"""
function calc_v_summary(models, sols, ids)
    df = map(ids) do id
        (; e) = voltage_error(models[id], sols[id])
        (; id..., rmse = sqrt(mean(abs2, e)) * 1000)
    end |> DataFrame
    return df
end

"""
    calc_module_v_summary(cell_models, cell_sols, module_models, module_sols, module_ids) -> DataFrame

Module-level voltage accuracy from both directions, in mV: `rmse_module` scores the module
model against the module voltage it was fitted on, and `rmse_cells` scores the 12 cell models
as a virtual module, summed prediction against summed cell voltage. Both are sums of cell
voltages, so they compare directly.

Do not score summed cell voltages against the measured module voltage instead: the wiring
drops between those two measurements reach several hundred mV.
"""
function calc_module_v_summary(cell_models, cell_sols, module_models, module_sols, module_ids)
    df = map(module_ids) do id
        (; p, m) = id
        errs = map(1:12) do c
            cid = (; p, m, c)
            sol = cell_sols[cid]
            Dict(zip(sol.idx, voltage_error(cell_models[cid], sol).e))
        end
        common = sort(collect(intersect([Set(keys(d)) for d in errs]...)))
        e_cells = [sum(d[i] for d in errs) for i in common]
        e_mod = voltage_error(module_models[id], module_sols[id]).e
        (;
            p, m,
            rmse_module = sqrt(mean(abs2, e_mod)) * 1000,
            rmse_cells = sqrt(mean(abs2, e_cells)) * 1000,
        )
    end |> DataFrame
    return df
end

# === SOH estimation ===

"""
    calc_module_soh_summary(cell_ids, cell_fit, module_ids, module_fit; Q_nom = 100) -> DataFrame

Per-module SOH from the cell fits and from the module fit, with the cell-to-cell spread that
separates them. `soh` is the string's usable capacity over `Q_nom` and `soh_module` the module
model's own estimate; `Δsoc_max`, `Δsoh_max`, `σ_soc` and `σ_soh` give the within-module
spread.

The unusable capacity splits into `loss_soh`, the irreversible part from capacity spread, and
`loss_soc`, the part a balancing cycle would recover. Both as a fraction of `Q_nom`.
"""
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

"""
    calc_cell_spread(cell_fit, cell_ids) -> DataFrame

Per-module spread of the per-cell composite-OCV fit in physical units — capacity in Ah, initial
SOC in % — as mean, standard deviation, min and max. [`calc_module_soh_summary`](@ref) reports
the same quantities normalised by `Q_nom` and only as σ/Δ.

Fit uncertainties propagate through, so the columns hold `Measurement`s.
"""
function calc_cell_spread(cell_fit, cell_ids)
    df = map(cell_fit.Q_cell, cell_fit.s0, cell_ids) do Q, s0, id
        (; p, m, c) = id
        (; p, m, c, soh = Q, soc = s0 * 100)
    end |> DataFrame
    # min/max MUST stay wrapped in lambdas: a bare `minimum`/`maximum` hits DataFrames'
    # fast-aggregation path, which initialises via `typemin` — undefined for `Measurement`.
    return combine(
        groupby(df, [:p, :m]),
        :soh => mean => :soh_mean,
        :soh => std => :soh_std,
        :soh => (x -> minimum(x)) => :soh_min,
        :soh => (x -> maximum(x)) => :soh_max,
        :soc => mean => :soc_mean,
        :soc => std => :soc_std,
        :soc => (x -> minimum(x)) => :soc_min,
        :soc => (x -> maximum(x)) => :soc_max,
        :soc => (x -> maximum(x) - minimum(x)) => :soc_delta,
        nrow => :n,
    )
end

"""
    calc_composite_rmse(comp_fit, cells, ids; n_soc = 200) -> DataFrame

Per-cell misfit to the consensus OCV from [`fit_composite_ocv`](@ref): each cell is placed on
the composite's SOC gauge through its own `(Q_cell, s0)`, then compared to the composite over
the SOC range the two share. `ocv_rmse` and `ocv_max` in mV, one row per cell.

The reference is the fit's own consensus, so this measures the voltage error the alignment
leaves behind, not agreement with a measurement.
"""
function calc_composite_rmse(comp_fit, cells, ids; n_soc = 200)
    (; soc_grid, v_grid, Q_cell, s0) = comp_fit
    Q = Measurements.value.(Q_cell)
    s = Measurements.value.(s0)

    ord = sortperm(soc_grid)
    v_comp = LinearInterpolation(v_grid[ord], soc_grid[ord]; extrapolation = ExtrapolationType.Constant)
    soc_comp_lo, soc_comp_hi = extrema(soc_grid)

    df = map(enumerate(ids)) do (i, id)
        soc = s[i] .+ collect(cells[i].q) ./ Q[i]
        v_cell = LinearInterpolation(collect(cells[i].μ), soc; extrapolation = ExtrapolationType.Constant)
        lo = max(minimum(soc), soc_comp_lo) + 0.002
        hi = min(maximum(soc), soc_comp_hi) - 0.002
        sg = collect(range(lo, hi; length = n_soc))
        r = (v_cell.(sg) .- v_comp.(sg)) .* 1000
        (; id..., ocv_rmse = sqrt(mean(abs2, r)), ocv_max = maximum(abs, r))
    end |> DataFrame
    return df
end

# === SOC estimation ===

"""
    calc_soc_trajectories(models, sols, fit, ids; tg) -> Matrix{Measurement}

SOC of every unit on the shared time grid `tg`, from each one's filtered charge trajectory
placed on its composite-fit capacity and initial SOC. `length(tg)` × `length(ids)`, carrying
the filter and fit uncertainties.
"""
function calc_soc_trajectories(models, sols, fit, ids; tg)
    trajs = map(enumerate(ids)) do (k, id)
        (; t, q, qσ) = charge_trajectory(models[id], sols[id])
        fq = LinearInterpolation(Measurements.measurement.(q, qσ), t; extrapolation = ExtrapolationType.Constant)
        fit.s0[k] .+ fq.(tg) ./ fit.Q_cell[k]
    end
    return stack(trajs)  # length(tg) × length(ids), eltype Measurement{Float64}
end

"""
    calc_module_soc(id, tg, soc_cell, soc_module, cell_fit, cell_ids, module_ids) -> DataFrame

The SOC trajectories of one module on the grid `tg`: its 12 cells as `soc_cell_1` …
`soc_cell_12`, those 12 aggregated into a string SOC as `soc_pack`, and the module model's own
estimate as `soc_module`.
"""
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

"""
    calc_soc_error(soc_cell, soc_module, cell_fit, cell_ids, module_ids) -> Matrix

What the module model gets wrong about SOC: `soc_module − aggregated cell SOC` per time step,
`length(tg)` × `length(module_ids)`.
"""
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

"""
    calc_module_soc_summary(soc_err, module_ids) -> DataFrame

Collapse the [`calc_soc_error`](@ref) trajectories to one row per module: `bias`, `rmse` and
worst-case `max`.
"""
function calc_module_soc_summary(soc_err, module_ids)
    df_soc = map(enumerate(module_ids)) do (k, id)
        e = soc_err[:, k]
        (; id.p, id.m, bias = mean(e), rmse = sqrt(mean(abs2, e)), max = maximum(abs, e))
    end |> DataFrame
    return df_soc
end

"""
    cell_capacities(fit, cell_ids, ids) -> Vector{Float64}

Capacities in Ah for the subset `ids`, as point estimates, looked up in a composite `fit` whose
`Q_cell` is indexed parallel to `cell_ids`. This is the `Qs` argument of
[`calc_charge_accuracy`](@ref) and [`calc_charge_error`](@ref).
"""
function cell_capacities(fit, cell_ids, ids)
    idx = Dict(cell_ids .=> eachindex(cell_ids))
    return [Measurements.value(fit.Q_cell[idx[id]]) for id in ids]
end

"""
    calc_charge_accuracy(soc_models, soc_sols, data, ti, ids, Qs; zt = fit_zscore()) -> DataFrame

Charge-estimation accuracy against the oscilloscope reference, one row per cell. `ids` are
cells of the reference module P1M9 — the only module carrying a current probe — and `Qs` their
capacities, aligned to `ids`. The probe measures the one shared string current, so all 12 cells
are scored against the same reference and only their voltages differ.

`rmse_soc` and `max_soc` are in % SOC, `rmse` and `max` the same errors in Ah, and `rmse_cc`
the module-current Coulomb-counting baseline.
"""
function calc_charge_accuracy(soc_models, soc_sols, data, ti, ids, Qs; zt = fit_zscore())
    q_probe = StatsBase.reconstruct(zt.q, [ui.q for ui in cell_dataset_osci(data, ti, first(ids).c).u])
    return map(ids, Qs) do id, Q
        sol = soc_sols[id]
        (; q) = charge_trajectory(soc_models[id], sol)                                   # EKF estimate [Ah]
        qref = q_probe[sol.idx]
        q_cc = StatsBase.reconstruct(zt.q, [u.q for u in sol.ut])                         # module-current CC [Ah]
        e, e_cc = q .- qref, q_cc .- qref
        rmse = sqrt(mean(abs2, e))
        (;
            id.p, id.m, id.c, Q,
            rmse_soc = 100rmse / Q, max_soc = 100maximum(abs, e) / Q,                     # %SOC
            rmse, max = maximum(abs, e), rmse_cc = sqrt(mean(abs2, e_cc)),
        )                 # Ah
    end |> DataFrame
end

"""
    calc_charge_error(soc_models, soc_sols, data, ti, ids, Qs; tg, zt = fit_zscore()) -> DataFrame

The [`calc_charge_accuracy`](@ref) error over time rather than collapsed to an RMSE: charge
error in % SOC on the grid `tg`, with `t` in hours and one column `c<n>` per cell.
"""
function calc_charge_error(soc_models, soc_sols, data, ti, ids, Qs; tg, zt = fit_zscore())
    df = DataFrame(t = collect(tg) ./ 3600)
    q_probe = StatsBase.reconstruct(zt.q, [ui.q for ui in cell_dataset_osci(data, ti, first(ids).c).u])
    for (id, Q) in zip(ids, Qs)
        (; t, q) = charge_trajectory(soc_models[id], soc_sols[id])
        qref = q_probe[soc_sols[id].idx]
        f = LinearInterpolation(100 .* (q .- qref) ./ Q, t; extrapolation = ExtrapolationType.Constant)
        df[!, "c$(id.c)"] = f.(tg)
    end
    return df
end

"""
    calc_soc_diagnostic(model, sol, data, ti, c, scenarios; θ, Ts = 1.0, zt = model.kf.p.zt)

Fault-rejection diagnostic for one reference cell. Each entry of `scenarios` gives an
`offset` in Ah, a wrong initial charge, and a `bias` in A, a constant current-sensor error;
the SOC filter is rerun under each and the charge trajectory returned alongside the
oscilloscope reference `q_ref` and the Coulomb-counting baseline `q_cc`.

The faults are injected in physical units, so the Coulomb-counting error reads directly off a
charge axis.
"""
function calc_soc_diagnostic(model, sol, data, ti, c, scenarios; θ, Ts = 1.0, zt = model.kf.p.zt)
    scale = only(zt.q.scale)  # shared i/q z-score scale → integrate current in z-scored units
    q_ref = StatsBase.reconstruct(zt.q, [ui.q for ui in cell_dataset_osci(data, ti, c).u])
    return map(scenarios) do (; offset, bias)
        bias_z = bias / scale
        # inject the bias into the input current and accumulate it into the charge channel so the
        # module-current Coulomb-counting reference (sol.ut.q) drifts with it
        u = [(; i = ui.i + bias_z, q = ui.q + bias_z * (k / 3600), ui.T) for (k, ui) in enumerate(sol.u)]
        sm = YuasaStateModel(model; q0 = offset, Ts, θ)
        s = reduce_sol(sm, run_kf!(sm, u, sol.y))
        t = s.idx
        (;
            offset, bias, t = t ./ 3600, q_ref = q_ref[t],
            q_cc = StatsBase.reconstruct(zt.q, [v.q for v in s.ut]) .+ offset,  # CC inherits the offset
            q = StatsBase.reconstruct(zt.q, s.qμ),
            qσ = sqrt.(s.qσ) .* scale,
        )
    end
end


"""
    calc_throughput(data, p, m) -> Float64

Total charge throughput in Ah over module `m` of phase `p`, integrating the absolute current
across its whole trace.
"""
function calc_throughput(data, p, m)
    df_i = select(data[:module_current], "_time" => "time", "module_average_current_$(p)_$(m)" => "value")
    dt = [Dates.value.(diff(df_i.time)) * 1.0e-3; 0]
    return sum(abs, df_i.value .* dt) / 3600
end


# === data availability ===

"""
    calc_data_completeness(data, ti) -> DataFrame

How much data each signal table actually carries over the window `ti`: non-missing values
across all its channels against the count expected at that signal's nominal sampling rate, as
`actual`, `expected` and their ratio `completeness`.
"""
function calc_data_completeness(data, ti)
    nominal_dt = Dict(  # nominal sampling period [s] per signal table
        :cell_voltage => 10.0,
        :module_voltage => 1.0,
        :module_current => 1.0,
        :battery_temperature => 15.0,
    )
    dur_s = (last(ti) - first(ti)) / Millisecond(1000)
    return map(collect(keys(nominal_dt))) do sig
        df = filter(:_time => ∈(ti), data[sig])
        chans = names(df, Not(:_time))
        expected = (floor(Int, dur_s / nominal_dt[sig]) + 1) * length(chans)
        actual = sum(c -> count(!ismissing, df[!, c]), chans)
        (; signal = sig, n_channels = length(chans), actual, expected, completeness = actual / expected)
    end |> DataFrame
end


# === fitted ECM parameters, per cell ===

"""
    calc_ecm_parameters(models, sols, ids; v_ref = 3.9, n = 1) -> DataFrame

Per-unit ECM parameters, alongside the OCV: `R0`, `R1` and `R_DC = R0 + R1` at `v_ref` (mΩ), RC
time constant `τ` (s) and Arrhenius coefficient `k` (K), each with its posterior standard
deviation.

`n` is the number of series cells the unit represents (1 cell, 12 module). Series quantities are
divided by it so the two levels compare directly; `τ` and `k` are intensive and are not.
"""
function calc_ecm_parameters(models, sols, ids; v_ref = 3.9, n = 1)
    return map(ids) do id
        model = models[id]
        sol = sols[id]
        zt = model.kf.p.zt
        ocv = gp_ocv(model, sol)
        r1 = gp_r1(model, sol)
        j = argmin(abs.(ocv.μ ./ n .- v_ref)) # compare at a voltage reference, since SOC not yet known

        # the `*_σ` fields are VARIANCES, hence the square roots; and a variance is rescaled by
        # `var·scale²`, not by `reconstruct`, which would also add the mean. τ and k are raw.
        R0 = StatsBase.reconstruct(zt.r, [sol.r0_μ[end]])[1] * 1000 / n
        (;
            id,
            R0, R0_sd = sqrt(max(sol.r0_σ[end], 0)) * zt.r.scale[1] * 1000 / n,
            R1 = r1.μ[j] * 1000 / n, R1_sd = r1.σ[j] * 1000 / n,
            R_DC = R0 + r1.μ[j] * 1000 / n, R_DC_sd = r1.σ[j] * 1000 / n,
            τ = sol.rc_τμ[end], τ_sd = sqrt(max(sol.rc_τσ[end], 0)),
            k = sol.arr_kμ[end], k_sd = sqrt(max(sol.arr_kσ[end], 0)),
            v_in_range = minimum(ocv.μ) / n <= v_ref <= maximum(ocv.μ) / n,
        )
    end |> DataFrame
end

"""
    calc_parameter_summary(df) -> DataFrame

Fleet spread of each ECM parameter against its median posterior uncertainty, from a
[`calc_ecm_parameters`](@ref) table: `median`, `post_sd`, `fleet_sd` and their `ratio`.
"""
function calc_parameter_summary(df)
    rows = map(
        (
            (:R1, :R1_sd, "R1 / mΩ"), (:R0, :R0_sd, "R0 / mΩ"), (:R_DC, :R_DC_sd, "R_DC / mΩ"),
            (:τ, :τ_sd, "τ / s"), (:k, :k_sd, "Arrhenius k / K"),
        )
    ) do (mc, sc, name)
        post = median(df[!, sc])
        fleet = std(df[!, mc])
        (; name, median = median(df[!, mc]), post_sd = post, fleet_sd = fleet, ratio = fleet / post)
    end
    return DataFrame(rows)
end
