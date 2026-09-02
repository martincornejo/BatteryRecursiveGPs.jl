"""
    load_dataset(datadir; signals = nothing, oscilloscope = false)

Read the cycling dataset into a `Dict{Symbol, DataFrame}`. `signals` selects which signals to load
— any of `:cell_voltage`, `:cell_soc`, `:module_voltage`, `:module_current`, `:derating_current`, `:battery_temperature` 
— defaulting to all of them.
With `oscilloscope = true` the P1M9 current reference is also read, under `:oscilloscope_current`.
"""
function load_dataset(datadir; signals = nothing, oscilloscope = false)
    files = (
        cell_voltage = "cell_voltages.csv",
        cell_soc = "cell_soc.csv",
        module_voltage = "module_voltage.csv",
        module_current = "module_current_average.csv",
        derating_current = "derating_currents.csv",
        battery_temperature = "battery_temperature.csv",
    )
    signals = something(signals, keys(files))
    data = Dict{Symbol, DataFrame}(
        sig => read_csv(DataFrame, datadir * files[sig])
            for sig in signals
    )
    if oscilloscope
        df = read_csv(DataFrame, datadir * "oscilloscope_p1_m9.csv")
        df.timestamp_utc = floor.(df.timestamp_utc, Second(1))
        data[:oscilloscope_current] = combine(groupby(df, :timestamp_utc), :MEAS1 => mean => :MEAS1)
    end
    return data
end


"""
    insert_missings(df, ts = Second(1); time = :time) -> DataFrame

Put `df` on a regular `ts` grid spanning its own time range. Timestamps with no sample become `missing`.
"""
function insert_missings(df, ts = Second(1); time = :time)
    t0 = first(df[:, time])
    t1 = last(df[:, time])

    df2 = DataFrame(time => t0:ts:t1)
    leftjoin!(df2, df, on = time)
    sort!(df2, time)

    return df2
end

function interpolate(df, ti; Ts = 1)
    t = Dates.value.(df.time - first(ti)) .÷ 1000
    tr = 0:Ts:(Dates.value(last(ti) - first(ti)) .÷ 1000)
    f = LinearInterpolation(df.value, t)

    dt = first(ti):Second(Ts):last(ti)

    return DataFrame(; time = dt, t = tr, value = f(tr))
end


"""
    fit_zscore(n = 1) -> (; v, σ, i, q, r)

Z-score transforms taking the filter's signals to normalized units: `v` for voltage, `σ` for
voltage spreads and standard deviations, `i` for current, `q` for charge and `r` for
resistance. `n` is the number of series cells, so `fit_zscore(12)` gives module-scale
transforms. The ranges are fixed rather than fitted to a dataset, so every unit shares one
scaling.
"""
function fit_zscore(n = 1)
    v = StatsBase.fit(ZScoreTransform, (n * 3.3):(n * 0.01):(n * 4.1))
    σ = StatsBase.fit(ZScoreTransform, (n * 3.3):(n * 0.01):(n * 4.1), center = false)
    i = StatsBase.fit(ZScoreTransform, -50:0.1:50, center = false)
    q = StatsBase.fit(ZScoreTransform, -50:0.1:50, center = false)
    r = ZScoreTransform(1, 1, [0.0], [σ.scale[1] / i.scale[1]])
    return (; v, σ, i, q, r)
end


# module sensor temperature, interpolated onto the ti grid
function _temperature(data, ti, p, m; Ts)
    df_T = copy(data[:battery_temperature])
    select!(df_T, "_time" => "time", "battery_sensor_temperature_$(p)_$(m)_1" => "value")
    return interpolate(df_T, ti; Ts)
end

# Coulomb-count the current and assemble the filter inputs. `df_v.value` must arrive z-scored.
function _assemble_uy(df_v, df_i, df_T; Ts, zt)
    q = cumsum(df_i.value) * Ts / 3600
    î = StatsBase.transform(zt.i, df_i.value)
    q̂ = StatsBase.transform(zt.q, q)
    u = [(; i, q, T) for (i, q, T) in zip(î, q̂, df_T.value)]
    y = [SA[v] for v in df_v.value]
    return (; u, y)
end


"""
    cell_dataset(data, ti, p, m, c; Ts = 1.0, zt = fit_zscore()) -> (; u, y)

Filter inputs and observations for cell `c` of module `m` in phase `p`, over the window `ti`
on a `Ts`-second grid. `u` holds `(; i, q, T)` per step — current, Coulomb-counted charge and
module temperature — and `y` the cell voltage. Both are in `zt`'s normalized units, except
`T`, which stays in °C for the Arrhenius correction. Voltage samples missing from the grid
come through as `missing`, which [`run_kf!`](@ref) skips.
"""
function cell_dataset(data, ti, p, m, c; Ts = 1.0, zt = fit_zscore())
    # voltage
    df_v = copy(data[:cell_voltage])
    select!(df_v, "_time" => "time", "cell_voltage_$(p)_$(m)_1_$(c)" => "value")
    subset!(df_v, :time => ByRow(∈(ti)))
    df_v[!, :value] = StatsBase.transform(zt.v, df_v.value)
    df_v = insert_missings(df_v, Second(Ts))

    # current
    df_i = copy(data[:module_current])
    select!(df_i, "_time" => "time", "module_average_current_$(p)_$(m)" => ByRow(x -> -x) => "value")
    df_i = interpolate(df_i, ti; Ts)

    df_T = _temperature(data, ti, p, m; Ts)
    return _assemble_uy(df_v, df_i, df_T; Ts, zt)
end


"""
    module_dataset(data, ti, p, m; Ts = 1.0, zt = fit_zscore(12)) -> (; u, y)

As [`cell_dataset`](@ref), but `y` is the terminal voltage of the whole module. Pass a
module-scale `zt` and fit with `n = 12`, so the identified parameters come out on the same
scale as the summed cell voltages.
"""
function module_dataset(data, ti, p, m; Ts = 1.0, zt = fit_zscore(12))
    # voltage
    df_v = copy(data[:module_voltage])
    select!(df_v, "_time" => "time", "module_voltage_$(p)_$(m)" => "value")
    df_v[!, :value] = StatsBase.transform(zt.v, df_v.value)
    df_v = insert_missings(df_v, Second(Ts))
    subset!(df_v, :time => ByRow(∈(ti)))

    # current
    df_i = copy(data[:module_current])
    select!(df_i, "_time" => "time", "module_average_current_$(p)_$(m)" => ByRow(x -> -x) => "value")
    df_i = interpolate(df_i, ti; Ts)

    df_T = _temperature(data, ti, p, m; Ts)
    return _assemble_uy(df_v, df_i, df_T; Ts, zt)
end


"""
    cell_dataset_osci(data, ti, c; Ts = 1.0, zt = fit_zscore()) -> (; u, y)

As [`cell_dataset`](@ref) for cell `c` of P1M9, but with the current taken from the
oscilloscope reference probe instead of the BMS sensor. The probe measures the one string
current shared by all 12 cells, so `u` is the same for every `c`.
"""
function cell_dataset_osci(data, ti, c; Ts = 1.0, zt = fit_zscore())
    p = 1
    m = 9

    df_v = copy(data[:cell_voltage])
    select!(df_v, "_time" => "time", "cell_voltage_$(p)_$(m)_1_$(c)" => "value")
    subset!(df_v, :time => ByRow(∈(ti)))
    df_v[!, :value] = StatsBase.transform(zt.v, df_v.value)
    df_v = insert_missings(df_v, Second(Ts))

    df_î = copy(data[:oscilloscope_current])
    select!(df_î, :timestamp_utc => "time", :MEAS1 => ByRow(x -> -x) => "value")
    df_î = interpolate(df_î, ti; Ts)

    df_T = _temperature(data, ti, p, m; Ts)
    return _assemble_uy(df_v, df_î, df_T; Ts, zt)
end


# Fixed priors for the cell/module fits; the selection supplies `ocv` and `r1` per unit.
function default_θ(; n = 1)
    return (;
        r1μ = n * 1.0e-3,
        r0 = (; σ0 = n * 5.0e-4, σ1 = 0.0),  # scalar R0 random-walk
        r0μ = n * 1.0e-3,
        vσ = n * 3.0e-3,
        Ts = 1.0,
        rc = (;
            v0 = n * 0.0, σ0_v = n * 1.0e-4, σ1_v = n * 5.0e-5,
            τ0 = 800.0, σ0_τ = 5.0, σ1_τ = 0.0,
        ),
        cc = (; σ0 = 0.0, σ1 = 0.0),  # first-pass parametrization trusts Coulomb counting
        arr = (; T0 = 25, k0 = 2000, σ0_k = 100.0, σ1_k = 0.0),
    )
end

function _scale_θ(u, y, θ)
    vlo, vhi = extrema(skipmissing(first(v) for v in y))
    qlo, qhi = extrema(x.q for x in u)
    v̂_span = vhi - vlo
    q̂_span = qhi - qlo
    ocv = (; σ = θ.ocv.σ * v̂_span, ℓ = θ.ocv.ℓ * q̂_span / v̂_span)
    r1 = (; σ = θ.r1.σ * v̂_span, ℓ = θ.r1.ℓ * q̂_span)
    return merge(θ, (; ocv, r1))
end

"""
    scale_θ(u, y, ϑ; n = 1) -> θ

Complete the selected hyperparameters `ϑ` with [`default_θ`](@ref) and scale them to the
data. `ϑ` must supply `ocv = (; ℓ, σ)` and `r1 = (; ℓ, σ)` — the two the selection chooses
per unit; everything else comes from the defaults. `n` is the number of series cells.

The returned `θ` is what a model builder consumes: `ocv.σ` and `r1.σ` scaled by the observed
voltage span, and the length scales by the observed charge span.
"""
function scale_θ(u, y, ϑ; n = 1)
    θ = merge(default_θ(; n), ϑ)
    return _scale_θ(u, y, θ)
end


"""
    fit_cells(data, ϑ, ti, ids) -> (; cell_models, cell_sols)

Fit a [`YuasaModel`](@ref) per cell in `ids` over the window `ti`, threaded. `ϑ` maps each id
to its selected hyperparameters, which [`scale_θ`](@ref) completes and scales per unit. Ids
whose fit throws are logged and dropped, so both dictionaries can be shorter than `ids`.
"""
function fit_cells(data, ϑ, ti, ids)
    zt = fit_zscore()
    make_uy = id -> cell_dataset(data, ti, id.p, id.m, id.c; zt)
    make_θ = (u, y, id) -> scale_θ(u, y, ϑ[id])
    (; models, sols) = fit_models_thread(YuasaModel, make_uy, make_θ, ids, zt)
    return (; cell_models = models, cell_sols = sols)
end

"""
    fit_modules(data, ϑ, ti, ids) -> (; module_models, module_sols)

As [`fit_cells`](@ref), but one model per module, fitted on the module terminal voltage with
module-scale priors (`n = 12`).
"""
function fit_modules(data, ϑ, ti, ids)
    zt = fit_zscore(12)
    make_uy = id -> module_dataset(data, ti, id.p, id.m; zt)
    make_θ = (u, y, id) -> scale_θ(u, y, ϑ[id]; n = 12)  # module-scale priors (12 series cells), matching zt = fit_zscore(12)
    (; models, sols) = fit_models_thread(YuasaModel, make_uy, make_θ, ids, zt)
    return (; module_models = models, module_sols = sols)
end

function thread_map(f, ids)
    out = Dict()
    for batch in Iterators.partition(ids, Threads.nthreads())
        tasks = Dict(id => Threads.@spawn f(id) for id in batch)
        for (id, task) in tasks
            try
                out[id] = fetch(task)
                @info "id=$id complete"
            catch e
                @error "id=$id failed" exception = e
            end
        end
    end
    return out
end

function fit_models_thread(make_model, make_uy, make_θ, ids, zt)
    runs = thread_map(ids) do id
        (; u, y) = make_uy(id)
        θ = make_θ(u, y, id)
        fit_model(make_model, u, y, θ, zt)
    end
    models = Dict(id => run.model for (id, run) in runs)
    sols = Dict(id => run.sol for (id, run) in runs)
    return (; models, sols)
end

"""
    eval_models(models, sols, ids) -> Dict

Replay each fitted model over its own inputs open loop, giving the voltage it predicts from
the identified parameters alone. Returns one reduced solution per id. See
[`eval_model`](@ref).
"""
function eval_models(models, sols, ids)
    return thread_map(ids) do id
        (; sol_eval) = eval_model(models[id], sols[id])
        sol_eval
    end
end

"""
    fit_soc_models(models, sols, ids; q0 = 0.0, Ts = 1.0, θ) -> (; models, sols)

Build a [`YuasaStateModel`](@ref) from each fitted model and run it closed loop over the same
data. The ECM parameters stay frozen at their identified values while a two-state filter
tracks charge and RC voltage, starting from charge `q0`. `θ` supplies the state model's
process and measurement noise.
"""
function fit_soc_models(models, sols, ids; q0 = 0.0, Ts = 1.0, θ)
    runs = thread_map(ids) do id
        sm = YuasaStateModel(models[id]; q0, Ts, θ)
        sol = run_kf!(sm, sols[id].u, sols[id].y)
        sol = reduce_sol(sm, sol)
        (; model = sm, sol)
    end
    soc_models = Dict(id => run.model for (id, run) in runs)
    soc_sols = Dict(id => run.sol for (id, run) in runs)
    return (; models = soc_models, sols = soc_sols)
end
