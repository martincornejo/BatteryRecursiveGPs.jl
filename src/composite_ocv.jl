"""
    fit_composite_ocv(cells; n_v_grid=50, n_v_pair=50)

Pairwise-OLS composite OCV identification.  Instead of building a single
voltage grid on the intersection of all cells (where one narrow cell shrinks
the grid for everyone), this formulation works with per-pair voltage overlaps.

For each pair `(i, j)` the aligned curves must agree on their mutual overlap:
`s_i·q_i(v) + Q0_i ≈ s_j·q_j(v) + Q0_j`.  The gauge degeneracy (additive +
multiplicative) is broken by anchor constraints at the intersection-boundary
voltages V_lo and V_hi where all cells have data: the composite is normalised
to `q*(V_lo) = 0`, `q*(V_hi) = 1`.  Every cell contributes two anchor rows,
so no cell is special and `A'A` is full rank — standard OLS covariance applies.

The returned named tuple includes `Q_common` and readout evaluated on the
global intersection grid.
"""
function fit_composite_ocv(cells; n_v_grid::Int = 50, n_v_pair::Int = 50)
    N = length(cells)
    N >= 2 || error("Need at least 2 cells for pairwise fit")
    fqs = [_as_v_frame_function(collect(c.q), collect(c.μ)) for c in cells]

    n_free = 2 * N

    v_anchor_lo = maximum(minimum(fq.t) for fq in fqs) + 1.0e-3
    v_anchor_hi = minimum(maximum(fq.t) for fq in fqs) - 1.0e-3
    v_anchor_lo >= v_anchor_hi && error("No voltage overlap across cells — cannot align.")

    pair_grids = Tuple{Int, Int, Vector{Float64}}[]
    n_total = 0
    for i in 1:N, j in (i + 1):N
        v_lo = max(minimum(fqs[i].t), minimum(fqs[j].t)) + 1.0e-3
        v_hi = min(maximum(fqs[i].t), maximum(fqs[j].t)) - 1.0e-3
        v_lo >= v_hi && continue
        vg = collect(range(v_lo, v_hi; length = n_v_pair))
        push!(pair_grids, (i, j, vg))
        n_total += n_v_pair
    end
    isempty(pair_grids) && error("No pair has voltage overlap — cannot align.")
    n_total += 2 * N

    A = zeros(n_total, n_free)
    b = zeros(n_total)

    row = 0
    for (i, j, vg) in pair_grids
        for v in vg
            row += 1
            qi_v = fqs[i](v)
            qj_v = fqs[j](v)

            A[row, 2 * i - 1] = 1.0
            A[row, 2 * i] = qi_v
            A[row, 2 * j - 1] = -1.0
            A[row, 2 * j] = -qj_v
        end
    end

    for i in 1:N
        row += 1
        A[row, 2 * i - 1] = 1.0
        A[row, 2 * i] = fqs[i](v_anchor_lo)

        row += 1
        A[row, 2 * i - 1] = 1.0
        A[row, 2 * i] = fqs[i](v_anchor_hi)
        b[row] = 1.0
    end

    θ = A \ b

    params = Vector{Vector{Float64}}(undef, N)
    for i in 1:N
        params[i] = [θ[2 * i - 1], θ[2 * i]]
    end

    Q_mat, v_grid = _build_shared_qv_grid(cells, n_v_grid)
    Q_common = _final_e_step(Q_mat, params)
    sums = _precompute_hessian_sums(Q_mat)
    readout = _union_gauge_readout(params, Q_common)
    rss = _residual_sum_of_squares(Q_mat, Q_common, params)

    pairwise_rss = sum(abs2, A * θ - b)

    return (;
        Q_mat, sums, rss, params, Q_common,
        Q_cell = readout.Q_cell, s0 = readout.s0,
        Q_full = readout.Q_full,
        Q_at_Vmin = readout.Q_at_Vmin,
        Q_at_Vmax = readout.Q_at_Vmax,
        v_grid, n_v_pair,
        ols_AtA = A' * A, pairwise_rss, n_equations = n_total,
    )
end


"""
    composite_ocv_uncertainty(fit)

OLS-based 1σ uncertainty for the pairwise composite-OCV fit returned by
`fit_composite_ocv`.

The anchor constraints at V_lo and V_hi make `A'A` full rank — no
pseudoinverse or eigenvalue zeroing is needed.  The covariance is the
standard OLS formula `Σ_θ = σ̂² · (A'A)⁻¹` where
`σ̂² = pairwise_rss / (n_eq - 2N)`.  Every cell gets nonzero uncertainty.

Returns `(; est, Σθ, σ²_hat)` where
`est::Vector{@NamedTuple{Q::Measurement, s0::Measurement}}` is vector-indexed
in the same order as `fit.Q_cell`.
"""
function composite_ocv_uncertainty(fit)
    (; params, Q_full, ols_AtA, pairwise_rss, n_equations) = fit
    N = length(params)
    n_free = 2 * N

    σ²_hat = pairwise_rss / (n_equations - n_free)
    Σθ = σ²_hat .* inv(ols_AtA)

    est = Vector{@NamedTuple{Q::Measurement{Float64}, s0::Measurement{Float64}}}(undef, N)
    for i in 1:N
        idx = [2 * i - 1, 2 * i]
        Σ_block = Σθ[idx, idx]
        si_val = params[i][2]
        J = [
            0.0 -Q_full / si_val^2
            1.0 / Q_full 0.0
        ]
        Σ_out = J * Σ_block * J'
        σ_Q = sqrt(max(Σ_out[1, 1], 0.0))
        σ_s0 = sqrt(max(Σ_out[2, 2], 0.0))
        est[i] = (; Q = fit.Q_cell[i] ± σ_Q, s0 = fit.s0[i] ± σ_s0)
    end

    return (; est, Σθ, σ²_hat)
end


"""
    extend_composite_ocv(fit, cells; n_v_grid=200)

Extend the composite OCV to the full union voltage range of all cells,
using the fitted per-cell `(Q0, s)` parameters from `fit.params`.

The fit operates on the voltage *intersection* (where all cells overlap).
This function re-evaluates each cell's aligned `q(V)` on a wider grid
spanning the *union* of all cells' voltage ranges, averaging only the
cells that have data at each voltage point.

Returns `(; Q_common, v_grid, counts)` where `counts[k]` is the number of
cells contributing at `v_grid[k]`.
"""
function extend_composite_ocv(fit, cells; n_v_grid::Int = 200)
    N = length(cells)
    fqs = [_as_v_frame_function(collect(c.q), collect(c.μ)) for c in cells]

    v_min = minimum(minimum(fq.t) for fq in fqs) + 1.0e-3
    v_max = maximum(maximum(fq.t) for fq in fqs) - 1.0e-3
    v_grid = collect(range(v_min, v_max; length = n_v_grid))

    Q_common = zeros(n_v_grid)
    counts = zeros(Int, n_v_grid)
    for i in 1:N
        fq = fqs[i]
        Q0, s = fit.params[i]
        v_lo_i, v_hi_i = extrema(fq.t)
        for (k, v) in enumerate(v_grid)
            v_lo_i <= v <= v_hi_i || continue
            Q_common[k] += s * fq(v) + Q0
            counts[k] += 1
        end
    end

    for k in eachindex(Q_common)
        counts[k] > 0 && (Q_common[k] /= counts[k])
    end

    return (; Q_common, v_grid, counts)
end


# === Private helpers ===

function _as_v_frame_function(q::AbstractVector, μ::AbstractVector)
    order = sortperm(μ)
    v_sorted = μ[order]
    q_sorted = q[order]
    mask = [true; diff(v_sorted) .> 1.0e-9]
    return LinearInterpolation(
        q_sorted[mask], v_sorted[mask];
        extrapolation = ExtrapolationType.Linear,
    )
end


function _build_shared_qv_grid(cells, n_v_grid::Int)
    fqs = [_as_v_frame_function(collect(c.q), collect(c.μ)) for c in cells]

    v_min = maximum(minimum(fq.t) for fq in fqs) + 1.0e-3
    v_max = minimum(maximum(fq.t) for fq in fqs) - 1.0e-3
    v_min >= v_max && error("No voltage overlap across cells — cannot align.")
    v_grid = collect(range(v_min, v_max; length = n_v_grid))

    K = n_v_grid
    N = length(cells)
    Q_mat = zeros(K, N)
    for i in 1:N
        fq = fqs[i]
        for (k, v) in enumerate(v_grid)
            Q_mat[k, i] = fq(v)
        end
    end
    return Q_mat, v_grid
end


function _precompute_hessian_sums(Q_mat::AbstractMatrix)
    K, N = size(Q_mat)
    sums = zeros(N, 3)
    for i in 1:N
        sq = 0.0
        sqq = 0.0
        @inbounds for k in 1:K
            q = Q_mat[k, i]
            sq += q
            sqq += q * q
        end
        sums[i, 1] = float(K)
        sums[i, 2] = sq
        sums[i, 3] = sqq
    end
    return sums
end


function _final_e_step(Q_mat::AbstractMatrix, params::Vector{Vector{Float64}})
    K, N = size(Q_mat)
    Q_common = zeros(K)
    for i in 1:N
        Q0, s = params[i][1], params[i][2]
        @inbounds for k in 1:K
            Q_common[k] += s * Q_mat[k, i] + Q0
        end
    end
    Q_common ./= N
    return Q_common
end


function _union_gauge_readout(params::Vector{Vector{Float64}}, Q_common::AbstractVector)
    Q_at_Vmin = first(Q_common)
    Q_at_Vmax = last(Q_common)
    Q_full = Q_at_Vmax - Q_at_Vmin
    @assert Q_full > 0 "Union gauge requires Q_common monotonic increasing in V"

    N = length(params)
    Q_cell = [Q_full / params[i][2] for i in 1:N]
    s0 = [(params[i][1] - Q_at_Vmin) / Q_full for i in 1:N]
    return (; Q_cell, s0, Q_full, Q_at_Vmin, Q_at_Vmax)
end


function _residual_sum_of_squares(
        Q_mat::AbstractMatrix,
        Q_common::AbstractVector,
        params::Vector{Vector{Float64}},
    )
    K, N = size(Q_mat)
    rss = 0.0
    for i in 1:N
        Q0, s = params[i][1], params[i][2]
        @inbounds for k in 1:K
            r = Q_common[k] - s * Q_mat[k, i] - Q0
            rss += r * r
        end
    end
    return rss
end
