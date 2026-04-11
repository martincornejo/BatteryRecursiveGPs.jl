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


# Average aligned cells into a single OCV on a common Q axis
# Each cell is mapped via: Q_common = Q_cell * s + Q0
function average_ocv(ocvs, params; n_samples = 500, bin_size = 0.1)
    all_q = Float64[]
    all_v = Float64[]

    for (i, f) in enumerate(ocvs)
        Q0, s = params[i]
        q_cell = range(first(f.t), last(f.t); length = n_samples)
        q_aligned = q_cell .* s .+ Q0
        append!(all_q, q_aligned)
        append!(all_v, f.(q_cell))
    end

    order = sortperm(all_q)
    all_q = all_q[order]
    all_v = all_v[order]

    q_binned = Float64[]
    v_binned = Float64[]
    q_start = floor(minimum(all_q); digits = 1)
    q_end = ceil(maximum(all_q); digits = 1)
    for qb in q_start:bin_size:q_end
        mask = qb .<= all_q .< qb + bin_size
        if count(mask) > 0
            push!(q_binned, qb + bin_size / 2)
            push!(v_binned, mean(all_v[mask]))
        end
    end

    return LinearInterpolation(v_binned, q_binned; extrapolation = ExtrapolationType.Constant)
end


# Build composite OCV via Q(V) linear regression
# Returns: (composite_ocv, per_cell_ocvs, params, stats)
#   params[i] = [Q0, s] where Q_common = Q_cell * s + Q0
#   stats[i] = (R2, rms_q, se_s, se_Q0)
function build_composite_ocv(
        df_c, df_d; m = 7, cells = 1:12,
        tek_offset_c, tek_offset_d,
        n_iters = 10, n_samples = 500
    )
    # Extract per-cell OCV V(Q) (charge+discharge average)
    ocvs = map(cells) do c
        id = (; m, c)
        fc = clean_ocv(df_c, id; dch = false, current_col = "tek", tek_offset = tek_offset_c)
        fd = clean_ocv(df_d, id; dch = true, current_col = "tek", tek_offset = tek_offset_d)
        Qmax = max(last(fc.t), last(fd.t))
        Q = range(0, Qmax; length = n_samples)
        v = (fd.(Q) .+ fc.(Q)) ./ 2
        LinearInterpolation(v, Q; extrapolation = ExtrapolationType.Constant)
    end

    # Invert each cell: V(Q) -> Q(V)
    qv_cells = [invert_ocv(f; n_samples) for f in ocvs]

    # Common voltage grid: overlapping range of all cells
    v_min = maximum(minimum(f.u) for f in ocvs)
    v_max = minimum(maximum(f.u) for f in ocvs)
    v_grid = range(v_min + 0.001, v_max - 0.001; length = 300)

    # Sample Q_i(V) for each cell at the common voltage grid
    Q_matrix = hcat([qv.(v_grid) for qv in qv_cells]...)

    n = length(cells)

    # Initialize: Q0=0, s=1 for all cells
    params = [[0.0, 1.0] for _ in 1:n]

    for iter in 1:n_iters
        # E-step: Q_common(V) = mean_i[ Q_i(V) * s_i + Q0_i ]
        Q_aligned = similar(Q_matrix)
        for i in 1:n
            Q0, s = params[i]
            Q_aligned[:, i] = Q_matrix[:, i] .* s .+ Q0
        end
        Q_common = vec(mean(Q_aligned; dims = 2))

        # M-step: for each cell, linear regression Q_common = s_i * Q_i + Q0_i
        for i in 1:n
            qi = Q_matrix[:, i]
            A = hcat(ones(length(qi)), qi)
            coeffs = A \ Q_common
            params[i] = [coeffs[1], coeffs[2]]
        end
    end

    # Shift so composite starts at 0
    q_min = minimum(first(f.t) * p[2] + p[1] for (f, p) in zip(ocvs, params))
    for p in params
        p[1] -= q_min
    end

    # Build composite OCV by averaging aligned curves
    ocv_ref = average_ocv(ocvs, params; n_samples)

    # Compute regression stats per cell
    Q_aligned = similar(Q_matrix)
    for i in 1:n
        Q0, s = params[i]
        Q_aligned[:, i] = Q_matrix[:, i] .* s .+ Q0
    end
    Q_common = vec(mean(Q_aligned; dims = 2))

    stats = map(1:n) do i
        qi = Q_matrix[:, i]
        q_pred = qi .* params[i][2] .+ params[i][1]
        ss_res = sum((Q_common .- q_pred) .^ 2)
        ss_tot = sum((Q_common .- mean(Q_common)) .^ 2)
        R2 = 1 - ss_res / ss_tot
        rms_q = sqrt(mean((Q_common .- q_pred) .^ 2))

        n_pts = length(qi)
        sigma2 = ss_res / (n_pts - 2)
        se_s = sqrt(sigma2 / sum((qi .- mean(qi)) .^ 2))
        se_Q0 = sqrt(sigma2 * mean(qi .^ 2) / sum((qi .- mean(qi)) .^ 2))

        (; R2, rms_q, se_s, se_Q0)
    end

    return ocv_ref, ocvs, params, stats
end


# ----------------------------------------------------------------------------
# Composite OCV with calibrated per-cell (Q, s0) uncertainty.
#
# Unweighted OLS EM over per-cell q(V) curves (same inner loop as
# build_composite_ocv), plus a gauge pin + joint Hessian UQ over
# θ = [Q_common[1..K], s_1, Q0_1, ..., s_N, Q0_N]. Readouts use the union
# gauge (Q_common endpoints at the observed voltage overlap).
#
# Recipe validated on the yuasa-synthetic benchmark in
# yuasa-synthetic-validation/approach_5_simple_qv_em.jl — see notes there
# for why weights are unit, why the gauge pin is post-hoc, and why the
# Hessian's two smallest eigenvalues are zeroed.
#
# ocvs: Vector of per-cell V(Q) interpolators (as produced by clean_ocv /
#       build_composite_ocv's `ocvs` field).
#
# Returns a NamedTuple with:
#   Q_cell::Vector{Float64}, σ_Q::Vector{Float64}       Ah
#   s0_cell::Vector{Float64}, σ_s0::Vector{Float64}     fraction of observed window
#   params::Vector{@NamedTuple{s, Q0}}                  gauge-pinned raw params
#   Q_common, v_grid                                    composite curve
#   Q_full, Q_at_Vmin, Q_at_Vmax                        union-gauge anchors
#   rss, σ²_hat                                         residual-variance estimate
#   H_eig_bottom3, n_iters_run                          diagnostics
# ----------------------------------------------------------------------------
function composite_ocv_with_uq(ocvs; n_v_grid = 300, n_iters = 30, tol = 1.0e-9)
    N = length(ocvs)

    # Invert V(Q) → q(V) on a shared voltage grid (overlap of all cells).
    qv_cells = [invert_ocv(f; n_samples = 500) for f in ocvs]
    v_min = maximum(minimum(f.u) for f in ocvs) + 1.0e-3
    v_max = minimum(maximum(f.u) for f in ocvs) - 1.0e-3
    v_min >= v_max && error("No voltage overlap across cells — cannot align.")
    v_grid = collect(range(v_min, v_max; length = n_v_grid))
    K = n_v_grid

    Q_mat = zeros(K, N)
    for i in 1:N, k in 1:K
        Q_mat[k, i] = qv_cells[i](v_grid[k])
    end

    # ---- EM loop: unit-weight OLS -------------------------------------------
    params = [(s = 1.0, Q0 = 0.0) for _ in 1:N]
    sums = zeros(N, 3)
    Q_common = zeros(K)

    n_iters_run = 0
    for iter in 1:n_iters
        n_iters_run = iter

        fill!(Q_common, 0.0)
        for i in 1:N
            (; s, Q0) = params[i]
            @inbounds for k in 1:K
                Q_common[k] += s * Q_mat[k, i] + Q0
            end
        end
        Q_common ./= N

        max_delta = 0.0
        for i in 1:N
            qi = @view Q_mat[:, i]
            sq = 0.0
            sqq = 0.0
            sy = 0.0
            sqy = 0.0
            @inbounds for k in 1:K
                q = qi[k]
                y = Q_common[k]
                sq += q
                sqq += q * q
                sy += y
                sqy += q * y
            end
            sums[i, 1] = float(K)
            sums[i, 2] = sq
            sums[i, 3] = sqq

            detA = K * sqq - sq * sq
            abs(detA) < 1.0e-14 && continue
            Q0_new = (sqq * sy - sq * sqy) / detA
            s_new = (-sq * sy + K * sqy) / detA

            max_delta = max(
                max_delta,
                abs(s_new - params[i].s),
                abs(Q0_new - params[i].Q0)
            )
            params[i] = (s = s_new, Q0 = Q0_new)
        end

        max_delta < tol && break
    end

    fill!(Q_common, 0.0)
    for i in 1:N
        (; s, Q0) = params[i]
        @inbounds for k in 1:K
            Q_common[k] += s * Q_mat[k, i] + Q0
        end
    end
    Q_common ./= N

    # Gauge pin at (s_1, Q0_1) = (1, 0). Loss is scale-invariant along the
    # multiplicative direction, so rss doesn't reflect real shape mismatch
    # until we fix the gauge.
    s1 = params[1].s
    Q01 = params[1].Q0
    for i in 1:N
        params[i] = (s = params[i].s / s1, Q0 = (params[i].Q0 - Q01) / s1)
    end
    Q_common .= (Q_common .- Q01) ./ s1

    # ---- Union-gauge readout ------------------------------------------------
    Q_at_Vmin = Q_common[1]
    Q_at_Vmax = Q_common[end]
    Q_full = Q_at_Vmax - Q_at_Vmin
    @assert Q_full > 0 "Union gauge requires Q_common monotonic increasing in V"

    Q_cell = [Q_full / params[i].s for i in 1:N]
    s0_cell = [(params[i].Q0 - Q_at_Vmin) / Q_full for i in 1:N]

    rss = 0.0
    for i in 1:N
        (; s, Q0) = params[i]
        @inbounds for k in 1:K
            r = Q_common[k] - s * Q_mat[k, i] - Q0
            rss += r * r
        end
    end

    # ---- Joint Hessian UQ ---------------------------------------------------
    P = K + 2N
    H = zeros(P, P)
    @inbounds for j in 1:K
        H[j, j] = float(N)
    end
    @inbounds for i in 1:N
        si, qi = K + 2i - 1, K + 2i
        for j in 1:K
            H[j, si] = -Q_mat[j, i]
            H[si, j] = H[j, si]
            H[j, qi] = -1.0
            H[qi, j] = H[j, qi]
        end
        H[si, si] = sums[i, 3]
        H[si, qi] = sums[i, 2]
        H[qi, si] = sums[i, 2]
        H[qi, qi] = sums[i, 1]
    end

    dof = N * K - (P - 2)
    σ²_hat = rss / dof

    F = eigen(Symmetric(H))
    λ = F.values
    order = sortperm(abs.(λ))
    H_eig_bottom3 = λ[order[1:3]]
    λ_inv = similar(λ)
    for k in eachindex(λ)
        λ_inv[k] = (k == order[1] || k == order[2]) ? 0.0 : 1.0 / λ[k]
    end
    Σθ = σ²_hat .* (F.vectors * Diagonal(λ_inv) * F.vectors')

    σ_Q = zeros(N)
    σ_s0 = zeros(N)
    for i in 1:N
        si_val = params[i].s
        idx = [K + 2i - 1, K + 2i]
        Σ_block = Σθ[idx, idx]
        J = [
            -Q_full / si_val^2 0.0
            0.0 1.0 / Q_full
        ]
        Σ_out = J * Σ_block * J'
        σ_Q[i] = sqrt(max(Σ_out[1, 1], 0.0))
        σ_s0[i] = sqrt(max(Σ_out[2, 2], 0.0))
    end

    return (;
        Q_cell, σ_Q, s0_cell, σ_s0,
        params, Q_common, v_grid,
        Q_full, Q_at_Vmin, Q_at_Vmax,
        rss, σ²_hat, H_eig_bottom3, n_iters_run,
    )
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
