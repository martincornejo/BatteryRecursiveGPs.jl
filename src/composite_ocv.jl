"""
    fit_composite_ocv(cells; n_v_grid=50, n_v_pair=50, uq=true, n_v_uq=80, ℓ_uq=nothing)

Pairwise-OLS composite OCV identification: the aligned curves of every cell
pair must agree on their mutual voltage overlap,
`s_i·q_i(v) + Q0_i ≈ s_j·q_j(v) + Q0_j`. Anchor rows at the common boundary
voltages fix the gauge: `q*(V_lo) = 0`, `q*(V_hi) = 1`.

Returns `(; Q_cell, s0, params, soc_grid, v_grid, Q_full)` with `Q_cell`/`s0`
as `Measurement` vectors. The 1σ bars treat each cell's residual against the
composite as a correlated curve (SE kernel; length scale `ℓ_uq`, estimated
from the residual autocorrelation when `nothing`).
Use `rescale_composite_ocv` to convert to absolute Ah, or pass `refs`
directly: `fit_composite_ocv(cells, refs)` fits and rescales in one step.
"""
function fit_composite_ocv(
        cells; n_v_grid = 50, n_v_pair = 50,
        uq = true, n_v_uq = 80, ℓ_uq = nothing,
    )
    N = length(cells)
    N >= 2 || error("Need at least 2 cells for pairwise fit")
    fqs = [_as_v_frame_function(collect(c.q), collect(c.μ)) for c in cells]

    n_free = 2 * N

    v_anchor_lo = maximum(minimum(fq.t) for fq in fqs) + 1.0e-3
    v_anchor_hi = minimum(maximum(fq.t) for fq in fqs) - 1.0e-3
    v_anchor_lo >= v_anchor_hi && error("No voltage overlap across cells — cannot align.")

    pair_grids = Tuple{Int, Int, Vector{Float64}}[]
    for i in 1:N, j in (i + 1):N
        v_lo = max(minimum(fqs[i].t), minimum(fqs[j].t)) + 1.0e-3
        v_hi = min(maximum(fqs[i].t), maximum(fqs[j].t)) - 1.0e-3
        v_lo >= v_hi && continue
        vg = collect(range(v_lo, v_hi; length = n_v_pair))
        push!(pair_grids, (i, j, vg))
    end
    isempty(pair_grids) && error("No pair has voltage overlap — cannot align.")

    # Normal-equations accumulation: G = AᵀA (n_free×n_free), d = Aᵀb.
    # Each row of the design matrix A is sparse (≤4 nonzeros), so we stream the
    # contributions instead of materialising A — which is multi-GB at fleet
    # scale (N(N-1)/2 pairs × n_v_pair rows). θ = G \ d is the identical OLS
    # solution to A \ b.
    G = zeros(n_free, n_free)
    d = zeros(n_free)

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
        # Composite (consensus) curve, interpolable at any voltage.
        v_dense = collect(range(v_min, v_max; length = 400))
        qstar = LinearInterpolation(_mean_aligned_qv(fqs, params, v_dense), v_dense)

        # Per-cell residual curves e_i(v) = s_i·q_i(v) + Q0_i − q*(v), sampled
        # on each cell's own window with one global step (median window width
        # / (n_v_uq - 1), i.e. scale-free in voltage). The shared step makes
        # lag-k pairs exactly k·Δv apart for every cell.
        Δv = median(maximum(fq.t) - minimum(fq.t) for fq in fqs) / (n_v_uq - 1)
        residuals = map(1:N) do i
            Q0_i, s_i = params[i]
            v_lo, v_hi = extrema(fqs[i].t)
            vg = collect((v_lo + 1.0e-3):Δv:(v_hi - 1.0e-3))
            e = [s_i * fqs[i](v) + Q0_i - qstar(v) for v in vg]
            (; vg, e)
        end

        ℓ = something(ℓ_uq, _estimate_lengthscale(residuals))
        kernel = with_lengthscale(SEKernel(), ℓ)

        for i in 1:N
            (; vg, e) = residuals[i]
            A = hcat(ones(length(vg)), fqs[i].(vg))    # design of the local fit: [1  q_i(v)]
            H = (A' * A) \ A'                          # sensitivity of (Q0_i, s_i) to e_i
            P = I - A * H                              # projects onto the visible residual
            K0 = kernelmatrix(kernel, vg)
            c = sum(abs2, e) / tr(P * K0 * P)          # amplitude: model matches observed residual
            Σ_block = c .* (H * K0 * H')               # Cov(Q0_i, s_i) under e_i ~ (0, c·K0)
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
    fit_composite_ocv(cells, refs; kwargs...)

Fit the composite OCV and rescale it to absolute units in one step.
`refs` holds two voltage–SOC reference points as a named tuple, e.g.
`refs = (v_low = 3.45, soc_low = 0.15, v_high = 4.05, soc_high = 0.95)`
means 3.45 V corresponds to 15 % SOC and 4.05 V to 95 %. Keyword
arguments are forwarded to the normalised fit.
"""
function fit_composite_ocv(cells, refs; kwargs...)
    fit = fit_composite_ocv(cells; kwargs...)
    return rescale_composite_ocv(fit, refs)
end

"""
    rescale_composite_ocv(fit, refs)

Transform the normalised `(Q_cell, s0)` from `fit_composite_ocv` into
absolute units given two voltage–SOC reference points,
`refs = (v_low = 3.3, soc_low = 0.05, v_high = 4.05, soc_high = 0.95)`.
The composite OCV curve is interpolated at the reference voltages to
determine the affine mapping.

Returns the fit with `Q_cell`, `s0` and `soc_grid` in the reference
gauge (Q in Ah, s0 as absolute SOC fraction).
"""
function rescale_composite_ocv(fit, refs)
    (; v_low, soc_low, v_high, soc_high) = refs
    composite = LinearInterpolation(fit.soc_grid, fit.v_grid)
    soc_at_low = composite(v_low)
    soc_at_high = composite(v_high)
    soc_span = (soc_high - soc_low) / (soc_at_high - soc_at_low)
    soc_zero = soc_low - soc_span * soc_at_low

    Q_cell = fit.Q_cell ./ soc_span
    s0 = soc_span .* fit.s0 .+ soc_zero
    soc_grid = soc_span .* fit.soc_grid .+ soc_zero
    return (; Q_cell, s0, params = fit.params, soc_grid, v_grid = fit.v_grid, Q_full = fit.Q_full)
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
        n_v_grid = 200,
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


# Correlation length of the residual curves, from their pooled autocorrelation:
# within-curve pairs k grid steps (= k·Δv exactly) apart, one correlation per
# lag; the SE kernel exp(-d²/2ℓ²) crosses exp(-1/2) at d = ℓ, so ℓ is read off
# where the measured decay passes that level.
function _estimate_lengthscale(residuals; max_lag = 40)
    Δv = residuals[1].vg[2] - residuals[1].vg[1]
    ρ = map(1:max_lag) do k
        xs = Float64[]
        ys = Float64[]
        for (; e) in residuals
            n = length(e)
            k < n || continue
            append!(xs, @view e[1:(n - k)])
            append!(ys, @view e[(1 + k):n])
        end
        cor(xs, ys)
    end

    target = exp(-0.5)
    k = findfirst(<(target), ρ)
    k === nothing && error("Residual autocorrelation never decays below exp(-1/2); increase max_lag.")
    k == 1 && error("Residuals decorrelate within one grid step; increase n_v_uq to resolve ℓ.")
    return Δv * ((k - 1) + (ρ[k - 1] - target) / (ρ[k - 1] - ρ[k]))
end
