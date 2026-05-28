"""
    fit_composite_ocv(cells; n_v_grid=50, n_v_pair=50, uq=true)

Pairwise-OLS composite OCV identification.  Instead of building a single
voltage grid on the intersection of all cells (where one narrow cell shrinks
the grid for everyone), this formulation works with per-pair voltage overlaps.

For each pair `(i, j)` the aligned curves must agree on their mutual overlap:
`s_i·q_i(v) + Q0_i ≈ s_j·q_j(v) + Q0_j`.  The gauge degeneracy (additive +
multiplicative) is broken by anchor constraints at the intersection-boundary
voltages V_lo and V_hi where all cells have data: the composite is normalised
to `q*(V_lo) = 0`, `q*(V_hi) = 1`.  Every cell contributes two anchor rows,
so no cell is special and `A'A` is full rank — standard OLS covariance applies.

Returns `(; Q_cell, s0, params, soc_grid, v_grid)`.
`Q_cell` and `s0` are `Measurement{Float64}` vectors with OLS-based 1σ bars.
`soc_grid` and `v_grid` define the composite OCV curve on the union voltage range.
Use `rescale_composite_ocv` with a voltage–SOC reference to convert to absolute Ah.
"""
function fit_composite_ocv(cells; n_v_grid::Int = 50, n_v_pair::Int = 50, uq::Bool = true)
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

    # Normal-equations accumulation: G = AᵀA (n_free×n_free), d = Aᵀb, s = bᵀb.
    # Each row of the design matrix A is sparse (≤4 nonzeros), so we stream the
    # contributions instead of materialising A — which is multi-GB at fleet
    # scale (N(N-1)/2 pairs × n_v_pair rows). θ = G \ d is the identical OLS
    # solution to A \ b; G also gives inv(AᵀA) for UQ and RSS = bᵀb − θᵀd.
    G = zeros(n_free, n_free)
    d = zeros(n_free)
    s = 0.0

    @inbounds for (i, j, vg) in pair_grids
        cols = (2 * i - 1, 2 * i, 2 * j - 1, 2 * j)
        for v in vg
            vals = (1.0, fqs[i](v), -1.0, -fqs[j](v))  # b row = 0
            for p in 1:4, r in 1:4
                G[cols[p], cols[r]] += vals[p] * vals[r]
            end
        end
    end

    @inbounds for i in 1:N
        cols = (2 * i - 1, 2 * i)
        for (v, b_rhs) in ((v_anchor_lo, 0.0), (v_anchor_hi, 1.0))
            vals = (1.0, fqs[i](v))
            for p in 1:2, r in 1:2
                G[cols[p], cols[r]] += vals[p] * vals[r]
            end
            d[cols[1]] += vals[1] * b_rhs
            d[cols[2]] += vals[2] * b_rhs
            s += b_rhs^2
        end
    end

    θ = G \ d  # A \ b → solution without materialising A

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

    Q_cell = Vector{Measurement{Float64}}(undef, N)
    s0 = Vector{Measurement{Float64}}(undef, N)
    if uq
        pairwise_rss = s - θ' * d   # = ‖Aθ − b‖² at the OLS solution (Gθ = d)
        σ²_hat = pairwise_rss / (n_total - n_free)
        Σθ = σ²_hat .* inv(G)
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
    else
        for i in 1:N
            si_val = params[i][2]
            Q_cell[i] = measurement(Q_full / si_val, NaN)
            s0[i] = measurement((params[i][1] - Q_at_Vmin) / Q_full, NaN)
        end
    end

    return (; Q_cell, s0, params, soc_grid, v_grid, Q_full)
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
    composite = LinearInterpolation(fit.soc_grid, fit.v_grid)
    soc_at_ref = composite.(v_ref)
    soc_span = (soc_ref[2] - soc_ref[1]) / (soc_at_ref[2] - soc_at_ref[1])
    soc_zero = soc_ref[1] - soc_span * soc_at_ref[1]

    Q_cell = fit.Q_cell ./ soc_span
    s0 = soc_span .* fit.s0 .+ soc_zero
    return (; Q_cell, s0)
end


"""
    rescaled_ocv_curve(fit; v_ref, soc_ref)

Return the composite OCV curve in absolute SOC gauge as `(v_grid, soc_grid)`.
Mirrors `rescale_composite_ocv` but returns the curve, not per-cell params.
Use as a portable reference OCV across experiments.
"""
function rescaled_ocv_curve(fit; v_ref, soc_ref)
    v_of_s = LinearInterpolation(fit.soc_grid, fit.v_grid)
    soc_at_ref = v_of_s.(v_ref)
    soc_span = (soc_ref[2] - soc_ref[1]) / (soc_at_ref[2] - soc_at_ref[1])
    soc_zero = soc_ref[1] - soc_span * soc_at_ref[1]
    soc_grid = soc_span .* fit.soc_grid .+ soc_zero
    return (; v_grid = fit.v_grid, soc_grid)
end


"""
    fit_cells_to_reference(cells, ref_soc_of_v, v_range; n_v_grid = 200)

Per-cell `(Q_cell, s0)` from least-squares alignment to a fixed external
reference OCV `ref_soc_of_v(v) → SOC`, valid over `v_range = (v_lo, v_hi)`.

Each cell is fit independently — no pairwise coupling, no gauge to fix.
Working in v-frame: `q_i(v) ≈ Q_i · ref_soc_of_v(v) − Q_i · s0_i`, linear
in `(Q_i, Q_i·s0_i)`. The reference fixes the absolute gauge, so
`(Q_cell, s0)` come back in the reference's units.

Returns `(; Q_cell, s0)` as `Measurement{Float64}` vectors with OLS 1σ.
"""
function fit_cells_to_reference(
        cells, ref_soc_of_v, v_range::Tuple{<:Real, <:Real};
        n_v_grid::Int = 200,
    )
    N = length(cells)
    Q_cell = Vector{Measurement{Float64}}(undef, N)
    s0 = Vector{Measurement{Float64}}(undef, N)

    for i in 1:N
        fq = _as_v_frame_function(collect(cells[i].q), collect(cells[i].μ))
        v_lo = max(minimum(fq.t), v_range[1]) + 1.0e-3
        v_hi = min(maximum(fq.t), v_range[2]) - 1.0e-3
        v_lo >= v_hi && error("Cell $i has no v overlap with the reference.")

        v_grid = collect(range(v_lo, v_hi; length = n_v_grid))
        A = hcat(ref_soc_of_v.(v_grid), ones(n_v_grid))
        b = [fq(v) for v in v_grid]
        β = A \ b

        rss = sum(abs2, A * β - b)
        σ²_hat = rss / (n_v_grid - 2)
        Σβ = σ²_hat .* inv(A' * A)

        # (Q, s0) = (β1, -β2/β1); J = [1 0; β2/β1² -1/β1]
        J = [1.0 0.0; β[2] / β[1]^2 -1.0 / β[1]]
        Σ_out = J * Σβ * J'

        Q_cell[i] = β[1] ± sqrt(max(Σ_out[1, 1], 0.0))
        s0[i] = -β[2] / β[1] ± sqrt(max(Σ_out[2, 2], 0.0))
    end

    return (; Q_cell, s0)
end


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
