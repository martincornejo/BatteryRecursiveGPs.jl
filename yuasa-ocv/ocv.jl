function integrate_current(df; current_col, negate = true, offset = 0.0)
    transform = negate ? (x -> -(x .+ offset)) : (x -> x .+ offset)
    df_i = select(df, :timestamp_utc, current_col => transform => :i)
    dropmissing!(df_i)
    Δt = [Dates.value.(diff(df_i.timestamp_utc)); 0] * 1.0e-3  # in seconds
    df_i[!, :q] = cumsum(df_i.i .* Δt) ./ 3600  # in Ah
    return df_i
end


function find_tek_offset(df; m = 7, bms_thresh = 0.1)
    df_both = select(df, :timestamp_utc, "m$(m)_current" => :bms, :tek_m_cur_ref => :tek)
    dropmissing!(df_both)

    # Only use rest periods where BMS reads ~0
    df_rest = subset(df_both, :bms => ByRow(x -> abs(x) < bms_thresh))

    # Offset = mean tek value at rest (so that tek + offset ≈ 0)
    return -mean(df_rest.tek)
end


function average_charge_discharge(fc, fd; n_samples = 500)
    Qmax = max(last(fc.t), last(fd.t))
    q = collect(range(0, Qmax; length = n_samples))
    μ = (fd.(q) .+ fc.(q)) ./ 2
    return (; q, μ)
end


function clean_ocv(df, id; dch::Bool, i_thresh = 0.5, current_col = "bms", tek_offset = 0.0)
    (; m, c) = id

    cell = @sprintf("%02d", c)
    df_cell = select(df, :timestamp_utc, "m$(m)_cell$cell" => :v)
    dropmissing!(df_cell)

    if current_col == "tek"
        df_i = integrate_current(df; current_col = "tek_m_cur_ref", negate = true, offset = tek_offset)
    else
        df_i = integrate_current(df; current_col = "m$(m)_current", negate = true)
    end

    q0 = minimum(df_i.q)
    df_i[!, :q] = df_i.q .- q0

    df1 = innerjoin(df_cell, df_i, on = :timestamp_utc, makeunique = true)
    sort!(df1, :timestamp_utc)

    df2 = subset(df1, :i => ByRow(<=(i_thresh) ∘ abs))
    df2[!, :q_floor] = floor.(df2.q; digits = 1)

    df3 = combine(groupby(df2, :q_floor)) do subdf
        sort(subdf, :v; rev = dch) |> first
    end
    sort!(df3, :q)

    return LinearInterpolation(df3.v, df3.q; extrapolation = ExtrapolationType.Constant)
end


function invert_ocv(f; n_samples = 500, extrapolation = ExtrapolationType.Constant)
    q = range(first(f.t), last(f.t); length = n_samples)
    v = f.(q)
    # Ensure strict monotonicity: remove duplicate/decreasing V values
    mask = [true; diff(v) .> 0]
    return LinearInterpolation(q[mask], v[mask]; extrapolation)
end


function smooth_ocv(ocv; window = 5)
    q = collect(ocv.t)
    v = collect(ocv.u)
    n = length(v)
    v_smooth = copy(v)
    for i in (window + 1):(n - window)
        v_smooth[i] = mean(@view v[(i - window):(i + window)])
    end
    return LinearInterpolation(v_smooth, q; extrapolation = ExtrapolationType.Constant)
end
