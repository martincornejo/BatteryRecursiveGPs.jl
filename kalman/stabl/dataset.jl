
function parse_datetime(str)
    if length(str) > 20
        dateformat = dateformat"y-m-dTH:M:S.sZ"
    else
        dateformat = dateformat"y-m-dTH:M:SZ" # no ms in format
    end
    DateTime(str, dateformat)
end

function load_phase_data(file)
    # read file
    df = CSV.File(file) |> DataFrame
    df[!, :_time] = parse_datetime.(df._time)

    # create lookup time tables
    data = Dict()
    for p in unique(df.phase)
        df_unit = subset(df, :phase => ByRow(==(p)))
        select!(df_unit, :_time => :time, :_value => :value)
        data[(; p)] = df_unit
    end

    return data
end

function load_module_data(file)
    # read file
    df = CSV.File(file) |> DataFrame
    df[!, :_time] = parse_datetime.(df._time)

    # create lookup time tables
    data = Dict()
    for p in unique(df.phase), m in unique(df.module)
        df_unit = subset(df,
            :phase => ByRow(==(p)),
            :module => ByRow(==(m)),
        )
        select!(df_unit, :_time => :time, :_value => :value)
        data[(; p, m)] = df_unit
    end

    return data
end

function load_module_data_debug(file, phases, modules)
    # read file
    df = CSV.File(file) |> DataFrame
    df[!, :_time] = parse_datetime.(df._time)

    # create lookup time tables
    data = Dict()
    for p in phases, m in modules
        key = "iModuleAvg_p$(p)_m$(m)"
        df_unit = subset(df, :debug_value_key => ByRow(==(key)))
        select!(df_unit, :_time => :time, :_value => :value)
        data[(; p, m)] = df_unit
    end

    return data
end

function load_cell_data(file; combine_cells=true)
    # read file
    df = CSV.File(file) |> DataFrame
    df[!, :_time] = parse_datetime.(df._time)

    # create lookup time tables
    data = Dict()
    for p in unique(df.phase), m in unique(df.module), c in unique(df.cell)
        df_unit = subset(df,
            :phase => ByRow(==(p)),
            :module => ByRow(==(m)),
            :cell => ByRow(==(c)),
        )
        select!(df_unit, :_time => :time, :_value => :value)
        data[(; p, m, c)] = df_unit
    end

    combine_cells || return data

    mdata = Dict()
    cells = unique(df.cell)
    for p in unique(df.phase), m in unique(df.module)
        mdata[(; p, m)] = combine_cell_data(data, (; p, m), cells)
    end

    return mdata
end

function combine_cell_data(cell_data, id, cells)
    (; p, m) = id
    c = first(cells)
    df = DataFrame(; time=cell_data[(; p, m, c)].time) # assumes that all cells have the same timestamps
    for c in cells
        df_cell = cell_data[(; p, m, c)]
        df[!, Symbol("cell_voltage_$c")] = df_cell.value
    end

    # remove rows with value == 0
    cell_voltage_cols = filter(col -> startswith(col, "cell_voltage_"), names(df))
    df = subset(df, cell_voltage_cols .=> ByRow(!=(0)))

    # min/max cell voltage
    transform!(df, AsTable(cell_voltage_cols) => ByRow(x -> minimum(values(x))) => :min_voltage)
    transform!(df, AsTable(cell_voltage_cols) => ByRow(x -> maximum(values(x))) => :max_voltage)
    transform!(df, [:max_voltage, :min_voltage] => ByRow((x, y) -> x - y) => :Δv)

    return df
end

function load_yuasa_dataset(files; combine_cells=true)
    data = Dict()
    data[:cell_voltage] = load_cell_data(files[:cell_voltage]; combine_cells)
    data[:module_voltage] = load_module_data(files[:module_voltage])
    data[:module_current_avg] = load_module_data_debug(files[:module_current_avg], 1:3, 1:9)
    data[:module_current_rms] = load_module_data(files[:module_current_rms])
    data[:module_temperature] = load_module_data(files[:module_temperature])
    return data
end




## === TODO: the next part could be simplified ===

function make_module_dataframe(data, tr, id; cells=1:12)
    df_cells = data[:cell_voltage][id]
    df_module_avg = data[:module_current_avg][id]
    df_module_temperature = data[:module_temperature][id]
    df_module_voltage = data[:module_voltage][id]

    # intialize df, set time
    t0 = first(tr)
    time = Dates.value.(tr .- t0) * 1e-3 # ms -> s
    df = DataFrame(; datetime=tr, time)

    # module current
    t_module_avg = Dates.value.(df_module_avg.time - t0) * 1e-3
    interp_iavg = LinearInterpolation(df_module_avg.value, t_module_avg) # ConstantInterpolation ?
    df[!, :module_current] = -interp_iavg(time) # positive -> charging

    # charge throughput
    Δt = diff(t_module_avg)
    Δq = [0; Δt .* -df_module_avg.value[1:end-1] ./ 3600]
    interp_q = LinearInterpolation(cumsum(Δq), t_module_avg)
    df[!, :q] = interp_q(time) .- interp_q(0)

    # module temperature
    t_temperature = Dates.value.(df_module_temperature.time - t0) * 1e-3
    interp_v = LinearInterpolation(df_module_temperature.value, t_temperature) # ConstantInterpolation ?
    df[!, :module_temperature] = interp_v(time)

    # module voltage
    t_voltage = Dates.value.(df_module_voltage.time - t0) * 1e-3
    interp_v = LinearInterpolation(df_module_voltage.value, t_voltage) # ConstantInterpolation ?
    df[!, :module_voltage] = interp_v(time)

    # cell voltage
    t_cell = Dates.value.(df_cells.time - t0) * 1e-3
    for c in 1:12
        v = df_cells[!, "cell_voltage_$c"]
        interp_vcell = ConstantInterpolation(v, t_cell)
        df[!, "cell_voltage_$c"] = interp_vcell(time)
    end

    return df
end

function make_module_dataframes(data, tr; cell_timestamps=false)
    mdata = Dict()
    t0 = first(tr)
    t1 = last(tr)
    for p in 1:3, m in 1:9
        id = (; p, m)
        if cell_timestamps
            # overwrite timerange with cell timestamps (within this timerange)
            data_cell = data[:cell_voltage]
            time_cell = data_cell[(; p, m)].time # assuming all cells have the same timestamp
            tr_cell = [t for t in time_cell if t0 <= t <= t1]
            mdata[id] = make_module_dataframe(data, tr_cell, id; cells=1:12)
        else
            mdata[id] = make_module_dataframe(data, tr, id; cells=1:12)
        end
    end
    return mdata
end


