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

Returns `(; Q_cell, s0, params, soc_grid, v_grid, v_anchor_lo, v_anchor_hi)`.
`Q_cell` and `s0` are `Measurement{Float64}` vectors with OLS-based 1σ bars.
`soc_grid` and `v_grid` define the composite OCV curve on the union voltage range.
`v_anchor_lo` and `v_anchor_hi` delimit the SOC window used for normalisation —
use them with a lab reference to convert to absolute Ah.
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

    v_min = minimum(minimum(fq.t) for fq in fqs) + 1.0e-3
    v_max = maximum(maximum(fq.t) for fq in fqs) - 1.0e-3
    v_grid = collect(range(v_min, v_max; length = n_v_grid))
    Q_common = _mean_aligned_qv(fqs, params, v_grid)
    Q_at_Vmin = first(Q_common)
    Q_full = last(Q_common) - Q_at_Vmin
    soc_grid = (Q_common .- Q_at_Vmin) ./ Q_full

    # uncertainty quantification
    pairwise_rss = sum(abs2, A * θ - b)
    σ²_hat = pairwise_rss / (n_total - n_free)
    Σθ = σ²_hat .* inv(A' * A)

    Q_cell = Vector{Measurement{Float64}}(undef, N)
    s0 = Vector{Measurement{Float64}}(undef, N)
    for i in 1:N
        Σ_block = Σθ[[2 * i - 1, 2 * i], [2 * i - 1, 2 * i]]
        si_val = params[i][2]
        J = [
            0.0 -Q_full / si_val^2
            1.0 / Q_full 0.0
        ]
        Σ_out = J * Σ_block * J'
        Q_cell[i] = Q_full / si_val ± sqrt(max(Σ_out[1, 1], 0.0))
        s0[i] = (params[i][1] - Q_at_Vmin) / Q_full ± sqrt(max(Σ_out[2, 2], 0.0))
    end

    return (; Q_cell, s0, params, soc_grid, v_grid, v_anchor_lo, v_anchor_hi)
end


"""
    rescale_composite_ocv(fit; v_ref, soc_ref)

Transform the normalised `(Q_cell, s0)` from `fit_composite_ocv` into
absolute units given two voltage–SOC reference pairs.

`v_ref` and `soc_ref` are 2-tuples: e.g. `v_ref = (3.3, 4.05)`,
`soc_ref = (0.05, 0.95)` means 3.3 V corresponds to 5 % SOC and
4.05 V to 95 %. The composite OCV curve is interpolated at the
reference voltages to determine the affine mapping.

Returns `(; Q_cell, s0)` as `Measurement{Float64}` vectors in the
reference gauge (Q in Ah, s0 as absolute SOC fraction).
"""
function rescale_composite_ocv(fit; v_ref, soc_ref)
    composite = LinearInterpolation(fit.v_grid, fit.soc_grid)
    soc_at_ref = composite.(v_ref)
    soc_span = (soc_at_ref[2] - soc_at_ref[1]) / (soc_ref[2] - soc_ref[1])
    soc_zero = soc_at_ref[1] - soc_ref[1] * soc_span

    Q_cell = fit.Q_cell ./ soc_span
    s0 = soc_span .* fit.s0 .+ soc_zero
    return (; Q_cell, s0)
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


function _mean_aligned_qv(fqs, params, v_grid)
    N = length(fqs)
    K = length(v_grid)
    Q_sum = zeros(K)
    counts = zeros(Int, K)
    for i in 1:N
        Q0, s = params[i]
        v_lo_i, v_hi_i = extrema(fqs[i].t)
        for (k, v) in enumerate(v_grid)
            v_lo_i <= v <= v_hi_i || continue
            Q_sum[k] += s * fqs[i](v) + Q0
            counts[k] += 1
        end
    end
    for k in eachindex(Q_sum)
        counts[k] > 0 && (Q_sum[k] /= counts[k])
    end
    return Q_sum
end
