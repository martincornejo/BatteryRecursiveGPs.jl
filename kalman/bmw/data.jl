function parse_datetime(str)
    if length(str) > 20
        dateformat = dateformat"y-m-dTH:M:S.sZ"
    else
        dateformat = dateformat"y-m-dTH:M:SZ" # no ms in format
    end
    DateTime(str, dateformat)
end

function load_phase_data(file, t0)
    # read file
    df = CSV.File(file) |> DataFrame
    df[!, :_time] = parse_datetime.(df._time)

    # create lookup time tables
    data = Dict()
    for p in unique(df.phase)
        df_unit = subset(df, :phase => ByRow(==(p)))
        Δt = Dates.value.(df_unit._time - t0) * 1e-3 # in seconds
        data[(; p)] = ConstantInterpolation(df_unit._value, Δt)
    end

    return data
end

function load_module_data(file, t0)
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
        Δt = Dates.value.(df_unit._time - t0) * 1e-3
        data[(; p, m)] = ConstantInterpolation(df_unit._value, Δt)
    end

    return data
end

function load_cell_data(file, t0)
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
        Δt = Dates.value.(df_unit._time - t0) * 1e-3
        data[(; p, m, c)] = ConstantInterpolation(df_unit._value, Δt)
    end

    return data
end

function load_dataset(files, t0)
    data = Dict()

    # system
    for var in (:phase_current,)
        data[var] = load_phase_data(files[var], t0)
    end

    # module
    for var in (:module_current, :module_voltage, :module_temperature)
        data[var] = load_module_data(files[var], t0)
    end

    # cell
    for var in (:cell_voltage,)
        data[var] = load_cell_data(files[var], t0)
    end

    return data
end


# =
function sample_dataset(data::Dict, ti, t0, battery::NamedTuple; cells=Int[])
    t = Dates.value.(ti - t0) * 1e-3

    df = DataFrame(:datetime => ti, :t => t)

    (; p, m) = battery

    # system
    for var in (:phase_current,)
        profile = data[var][(; p)]
        df[!, var] = profile(t)
    end

    # module
    for var in (:module_current, :module_voltage, :module_temperature)
        profile = data[var][(; p, m)]
        df[!, var] = profile(t)
    end

    # cell
    for c in cells
        profile = data[:cell_voltage][(; p, m, c)]
        df[!, Symbol("cell_voltage_$c")] = profile(t)
    end

    return df
end

function sample_cell_dataset(data, ti, t0, battery, cell)
    df = sample_dataset(data, ti, t0, battery, cells=[cell])
    df_cell = rename(df, [:module_current => :i, :module_temperature => :T, Symbol("cell_voltage_$cell") => :v])
    select!(df_cell, [:t, :i, :v, :T])
    df_cell[!, :i] = -df_cell.i
    df_cell[!, :q] = cumsum(df_cell.i) / 3600
    return df_cell
end


function load_debug_avg_current(file, t0)
    # read file
    df = CSV.File(file) |> DataFrame
    df[!, :_time] = parse_datetime.(df._time)

    # filter rows 
    subset!(df, :debug_value_key => ByRow(x -> occursin(r"^moduleCurrent_avg_(\d+)_(\d+)$", x)))

    # set phase and module ids
    id = map(df[:, "debug_value_key"]) do key
        m = match(r"^moduleCurrent_avg_(\d+)_(\d+)$", key)
        (parse(Int, m.captures[1]) + 1, parse(Int, m.captures[2]) + 1)
    end
    df[!, :phase] = id .|> first
    df[!, :module] = id .|> last

    # create lookup time tables
    data = Dict()
    for p in unique(df.phase), m in unique(df.module)
        df_unit = subset(df,
            :phase => ByRow(==(p)),
            :module => ByRow(==(m)),
        )
        Δt = Dates.value.(df_unit._time - t0) * 1e-3
        data[(; p, m)] = ConstantInterpolation(df_unit._value, Δt)
    end

    return data
    # return df
end


function load_debug_dataset(files, t0)
    data = Dict()

    # system
    for var in (:phase_current,)
        data[var] = load_phase_data(files[var], t0)
    end

    # module avg current
    for var in (:module_current,)
        data[var] = load_debug_avg_current(files[var], t0)
    end

    # module
    for var in (:module_voltage, :module_temperature)
        data[var] = load_module_data(files[var], t0)
    end

    # cell
    for var in (:cell_voltage,)
        data[var] = load_cell_data(files[var], t0)
    end

    return data
end