# Figure showing the ECM (OCV + R0/R1) at one filter step, next to the measured voltage and
# the voltage the ECM reproduces on its own, with a cursor marking how much data the filter has
# seen. Returns the figure plus a `set_frame!(i)` closure that moves it to frame `i` — shared by
# `animate_model` and `plot_animation_snapshot`.
#
# NOTE: pass the FULL sol from run_kf! BEFORE reduce_sol (needs per-timestep xt/Rt), and pass a
# FRESHLY BUILT model — `run_kf!` mutates the KF in place, so re-running a fitted model starts
# from its converged posterior and the ECM is already learned at step 1.
#
# `prior = (; x, R)` prepends the pre-observation state as frame 1: `run_kf!` records the state
# AFTER each correction, so `sol.xt[1]` has already absorbed one measurement. Capture it off the
# freshly built model (`copy(model.kf.x)`, `copy(model.kf.R)`) before running the filter.
#
# Mutates `model.kf` (see `predict_v` below) — the ECM curves read the passed-in x/R rather than
# the filter state, so the sol stays valid, but do not rely on `model` afterwards.
function _model_frame(model::YuasaModel, sol; prior = nothing)
    kf = model.kf
    zt = kf.p.zt
    (; yt) = sol

    xt = isnothing(prior) ? sol.xt : vcat([prior.x], sol.xt)
    Rt = isnothing(prior) ? sol.Rt : vcat([prior.R], sol.Rt)
    cursor_at = i -> clamp(isnothing(prior) ? i : i - 1, 1, length(yt))

    q̂ = collect(range(extrema(kf.p.r1.b0)...; step = 0.01))
    q = StatsBase.reconstruct(zt.q, q̂)

    # ECM curves at a given filter state (x, R)
    ecm(x, R) = begin
        ocv = predict_gp(kf, q̂, x, R, :ocv)
        ocvμ = StatsBase.reconstruct(zt.v, ocv.μ)
        ocvσ = StatsBase.reconstruct(zt.σ, sqrt.(diag(ocv.Σ)))
        xc = ComponentVector(x, kf.p.xid)
        r0μ = (StatsBase.reconstruct(zt.r, [abs(xc.r0.r)]) |> first) * 1.0e3
        r1 = predict_gp(kf, q̂, x, R, :r1)
        r1μ = StatsBase.reconstruct(zt.r, r1.μ) * 1.0e3
        r1σ = StatsBase.reconstruct(zt.r, sqrt.(diag(r1.Σ))) * 1.0e3
        (; ocvμ, ocvσ, rμ = r0μ .+ r1μ, rσ = r1σ)
    end

    # Voltage the ECM frozen at (x, R) reproduces on its own: `reinit_kf!` resets charge and RC
    # voltage to the start of the window while keeping the GP/R0 posteriors, and `tt = 0` skips
    # every correction — a pure simulation over the whole dataset, not the one-step-ahead
    # prediction in `sol.yμ` (which is corrected each step and therefore always tracks).
    # The simulation is uncorrected across the whole window, but the parameters come from data up
    # to the cursor only: left of it the curve reconstructs data the filter has seen, right of it
    # it extrapolates. Hence the plain "Model" label rather than "prediction".
    predict_v = (x, R) -> begin
        reinit_kf!(model; x, R)
        ol = run_kf!(model, sol.u, sol.y; tt = 0)
        μ = StatsBase.reconstruct(zt.v, first.(ol.yμ))
        σ = StatsBase.reconstruct(zt.σ, sqrt.(first.(ol.yΣ)))
        (; μ, σ)
    end

    fig = Figure(size = (900, 400))
    axo = Axis(fig[1, 2]; ylabel = "OCV / V")
    axr = Axis(fig[2, 2]; ylabel = "R / mΩ", xlabel = "Charge / Ah")
    axv = Axis(fig[1:2, 1]; ylabel = "Voltage / V", xlabel = "Step")
    linkxaxes!(axo, axr)
    hidexdecorations!(axo; ticks = false, grid = false)

    (; ocvμ, ocvσ, rμ, rσ) = ecm(xt[1], Rt[1])
    lo = lines!(axo, q, ocvμ; color = Cycled(1))
    bo = band!(axo, q, ocvμ + 2ocvσ, ocvμ - 2ocvσ; color = Cycled(1), alpha = 0.3)
    lr = lines!(axr, q, rμ; color = Cycled(1))
    br = band!(axr, q, rμ + 2rσ, rμ - 2rσ; color = Cycled(1), alpha = 0.3)

    v = StatsBase.reconstruct(zt.v, first.(yt))
    k = eachindex(v)
    v̂ = predict_v(xt[1], Rt[1])
    lines!(axv, k, v; color = :gray, label = "Measured")
    lv = lines!(axv, k, v̂.μ; color = Cycled(2), label = "Model")
    bv = band!(axv, k, v̂.μ - 2v̂.σ, v̂.μ + 2v̂.σ; color = Cycled(2), alpha = 0.3)
    cursor = vlines!(axv, cursor_at(1); color = :red)
    axislegend(axv; position = :rb)

    # Fixed limits: the prior bands are far wider than the converged ones, so autoscaling would
    # make the early frames unreadable and the late ones flat. Anchor each ECM panel on its own
    # converged (final-state) curve — the measured voltage range does not cover the OCV, which
    # extends past it at the edges of the charge window. The early model traces and their bands
    # run well outside these ranges and are deliberately left to clip.
    fin = ecm(xt[end], Rt[end])
    olo, ohi = extrema(vcat(fin.ocvμ - 2fin.ocvσ, fin.ocvμ + 2fin.ocvσ))
    ylims!(axo, olo - 0.1(ohi - olo), ohi + 0.1(ohi - olo))
    rlo, rhi = extrema(vcat(fin.rμ - 2fin.rσ, fin.rμ + 2fin.rσ))
    ylims!(axr, rlo - 0.5(rhi - rlo), rhi + 0.5(rhi - rlo))
    xlims!(axo, extrema(q)...)  # linked to axr

    vlo, vhi = extrema(v)
    ylims!(axv, vlo - 0.15(vhi - vlo), vhi + 0.15(vhi - vlo))
    xlims!(axv, first(k), last(k))

    set_frame! = i -> begin
        (; ocvμ, ocvσ, rμ, rσ) = ecm(xt[i], Rt[i])
        Makie.update!(lo, arg2 = ocvμ)
        Makie.update!(bo, arg2 = ocvμ + 2ocvσ, arg3 = ocvμ - 2ocvσ)
        Makie.update!(lr, arg2 = rμ)
        Makie.update!(br, arg2 = rμ + 2rσ, arg3 = rμ - 2rσ)
        v̂ = predict_v(xt[i], Rt[i])
        Makie.update!(lv, arg2 = v̂.μ)
        Makie.update!(bv, arg2 = v̂.μ - 2v̂.σ, arg3 = v̂.μ + 2v̂.σ)
        Makie.update!(cursor, arg1 = cursor_at(i))
    end

    return (; fig, set_frame!, n_frames = length(xt))
end

"""
    animate_model(file, model, sol; step = 60, framerate = 24, prior = nothing)

Animate the identified ECM as the filter learns over the dataset, writing to `file`. `step` is
the stride in filter steps between frames, so the result runs `length(1:step:n_frames) /
framerate` seconds. Pass the pre-fit state as `prior` to draw it behind the current estimate.

Each frame re-simulates the whole dataset open loop, about 3 s, so a run costs roughly
`n_frames / step × 3 s` — check the frame count first. Lengthening the animation through
`framerate` is free; through `step` it is not.
"""
function animate_model(file, model::YuasaModel, sol; step = 60, framerate = 24, prior = nothing)
    (; fig, set_frame!, n_frames) = _model_frame(model, sol; prior)
    return record(set_frame!, fig, file, 1:step:n_frames; framerate)
end

# Single frame of the learning animation, at frame `i` — debugging aid for checking the figure
# without paying for the full video. Unexported: call it as `YuasaAnalysis.plot_animation_snapshot`.
# With `prior` given, frame 1 is the pre-observation state and filter step `k` is frame `k + 1`.
function plot_animation_snapshot(model::YuasaModel, sol, i; prior = nothing)
    (; fig, set_frame!) = _model_frame(model, sol; prior)
    set_frame!(i)
    return fig
end
