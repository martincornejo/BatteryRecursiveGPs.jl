"""
    fit_composite_ocv(cells; n_v_grid=50, n_iters=30, tol=1.0e-9)

Identify a composite OCV curve and per-cell `(Q, s0)` from a set of GP-style
posterior means via unweighted EM in q(V).

Each entry of `cells` is a `(; q, μ)` named tuple — `q` in Ah, `μ` (mean OCV)
in V — with the cell's own q-grid. Cells need not share a grid; the routine
builds a shared overlap grid in V automatically.

The estimator alternates E-steps (averaging affinely-aligned per-cell q(V)
curves into a composite `Q_common`) with closed-form 2-parameter OLS M-steps
that re-fit `[Q0_i, s_i]` for each cell. After convergence the gauge is
pinned post-hoc to `(s_1, Q0_1) = (1, 0)` so residuals reflect real shape
mismatch and downstream uncertainty quantification has a well-defined origin.

Returns a NamedTuple with:

- `Q_cell` — per-cell full capacity (Ah), in the union gauge
- `s0` — per-cell initial-charge fraction in `[0, 1]`, union gauge
- `Q_full`, `Q_at_Vmin`, `Q_at_Vmax` — composite capacity at the V-grid edges
- `Q_common`, `v_grid` — composite q(V) curve
- `params` — per-cell `[Q0, s]` after the post-hoc gauge pin
- `Q_mat`, `sums`, `rss`, `n_iters_run` — internals consumed by
  `composite_ocv_uncertainty`

Pass the result to `composite_ocv_uncertainty` for joint-Hessian 1σ bars
on `Q_cell` and `s0`. Results are vector-indexed in the same order as
`cells`; use `Dict(ids .=> fit.Q_cell)` if an ID-keyed view is needed.

See `notes/approach-5-simple-qv-em.md` for derivation and design notes.
"""
function fit_composite_ocv(cells; n_v_grid::Int=50, n_iters::Int=30, tol::Float64=1.0e-9)
    Q_mat, v_grid = _build_shared_qv_grid(cells, n_v_grid)
    sums = _precompute_hessian_sums(Q_mat)

    K, N = size(Q_mat)
    params = [[0.0, 1.0] for _ in 1:N]

    n_iters_run = 0
    for iter in 1:n_iters
        n_iters_run = iter

        Q_aligned = hcat((Q_mat[:, i] .* params[i][2] .+ params[i][1] for i in 1:N)...)
        Q_common = vec(mean(Q_aligned; dims=2))

        max_delta = 0.0
        for i in 1:N
            A = hcat(ones(K), Q_mat[:, i])
            new_params = A \ Q_common
            max_delta = max(max_delta, maximum(abs, new_params .- params[i]))
            params[i] = new_params
        end

        max_delta < tol && break
    end

    Q_common = _final_e_step(Q_mat, params)
    _gauge_pin!(params, Q_common)
    readout = _union_gauge_readout(params, Q_common)
    rss = _residual_sum_of_squares(Q_mat, Q_common, params)

    return (;
        Q_mat, sums, rss, params, Q_common,
        Q_cell=readout.Q_cell, s0=readout.s0,
        Q_full=readout.Q_full,
        Q_at_Vmin=readout.Q_at_Vmin,
        Q_at_Vmax=readout.Q_at_Vmax,
        v_grid, n_iters_run,
    )
end


"""
    composite_ocv_uncertainty(fit)

Joint-Hessian 1σ uncertainty for the composite-OCV fit returned by
`fit_composite_ocv`.

Assembles the full Hessian over `θ = [Q_common[1..K], s_1, Q0_1, …, s_N, Q0_N]`,
inflates by the residual variance `σ²_hat = rss / (N*K - (P-2))` where the
`-2` accounts for the rank-2 null space of the unweighted OLS problem (one
exact additive gauge, one near-null multiplicative direction), then
pseudoinverts via eigendecomposition with the two smallest-|λ| eigenvalues
zeroed.

Each cell's 2×2 sub-block is propagated through the readout Jacobian
`J = [-Q_full/s_i²  0; 0  1/Q_full]` to give 1σ bars on `Q_cell` and `s0`.

Returns `(; est, H, Σθ, σ²_hat, H_eig_bottom3)` where
`est::Vector{@NamedTuple{Q::Measurement, s0::Measurement}}` is vector-indexed
in the same order as `fit.Q_cell`. Use `Dict(ids .=> uq.est)` for an
ID-keyed view.

See `notes/approach-5-simple-qv-em.md` for derivation and design notes.
"""
function composite_ocv_uncertainty(fit)
    (; Q_mat, sums, rss, params, Q_full) = fit
    K, N = size(Q_mat)
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

    # H has a rank-2 null space (exact additive gauge + near-null multiplicative direction); subtract 2 from the parameter count.
    dof = N * K - (P - 2)
    σ²_hat = rss / dof

    F = eigen(Symmetric(H))
    λ = F.values
    order = sortperm(abs.(λ))
    H_eig_bottom3 = λ[order[1:3]]
    # Zero the two smallest-|λ| eigenvalues to invert across the null space.
    λ_inv = similar(λ)
    for k in eachindex(λ)
        λ_inv[k] = (k == order[1] || k == order[2]) ? 0.0 : 1.0 / λ[k]
    end
    Σθ = σ²_hat .* (F.vectors * Diagonal(λ_inv) * F.vectors')

    est = Vector{@NamedTuple{Q::Measurement{Float64}, s0::Measurement{Float64}}}(undef, N)
    for i in 1:N
        si_val = params[i][2]
        idx = [K + 2i - 1, K + 2i]
        Σ_block = Σθ[idx, idx]
        J = [
            -Q_full/si_val^2 0.0
            0.0 1.0/Q_full
        ]
        Σ_out = J * Σ_block * J'
        σ_Q = sqrt(max(Σ_out[1, 1], 0.0))
        σ_s0 = sqrt(max(Σ_out[2, 2], 0.0))
        est[i] = (; Q=fit.Q_cell[i] ± σ_Q, s0=fit.s0[i] ± σ_s0)
    end

    return (; est, H, Σθ, σ²_hat, H_eig_bottom3)
end


"""
    extend_composite_ocv(fit, cells; n_v_grid=200)

Extend the composite OCV from `fit_composite_ocv` to the full union voltage
range of all cells, using the EM-fitted per-cell `(Q0, s)` parameters.

The EM fits on the voltage *intersection* (where all cells overlap). This
function re-evaluates each cell's aligned `q(V)` on a wider grid spanning
the *union* of all cells' voltage ranges, averaging only the cells that
have data at each voltage point.

Returns `(; Q_common, v_grid, counts)` where `counts[k]` is the number of
cells contributing at `v_grid[k]`.
"""
function extend_composite_ocv(fit, cells; n_v_grid::Int=200)
    N = length(cells)
    fqs = [_as_v_frame_function(collect(c.q), collect(c.μ)) for c in cells]

    v_min = minimum(minimum(fq.t) for fq in fqs) + 1.0e-3
    v_max = maximum(maximum(fq.t) for fq in fqs) - 1.0e-3
    v_grid = collect(range(v_min, v_max; length=n_v_grid))

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
        extrapolation=ExtrapolationType.Linear,
    )
end


function _build_shared_qv_grid(cells, n_v_grid::Int)
    fqs = [_as_v_frame_function(collect(c.q), collect(c.μ)) for c in cells]

    v_min = maximum(minimum(fq.t) for fq in fqs) + 1.0e-3
    v_max = minimum(maximum(fq.t) for fq in fqs) - 1.0e-3
    v_min >= v_max && error("No voltage overlap across cells — cannot align.")
    v_grid = collect(range(v_min, v_max; length=n_v_grid))

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


function _gauge_pin!(params::Vector{Vector{Float64}}, Q_common::AbstractVector)
    # Loss is scale-invariant in s, so EM drifts and rss only reflects shape mismatch after pinning (s_1, Q0_1) = (1, 0).
    Q01 = params[1][1]
    s1 = params[1][2]
    for i in eachindex(params)
        params[i] = [(params[i][1] - Q01) / s1, params[i][2] / s1]
    end
    Q_common .= (Q_common .- Q01) ./ s1
    return nothing
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
