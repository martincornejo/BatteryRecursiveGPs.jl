
function loss(θ̂, p)
    (; ϑ, zt, u, y) = p

    θ⁺ = softplus.(θ̂) # transform to force positive values
    θ = ComponentVector(; θ⁺..., ϑ...)
    kf = build_kf(θ, u, zt)

    sol = run_kf!(kf, u, y)
    return sum(abs2, sol.e)
end

function residuals(θ̂, p)
    (; ϑ, zt, u, y) = p

    θ⁺ = softplus.(θ̂) # transform to force positive values
    θ = ComponentVector(; θ⁺..., ϑ...)
    kf = build_kf(θ, u, zt)

    sol = run_kf!(kf, u, y)
    return sol.e
end

function tune_hyperparams(θ0, p)
    u0 = invsoftplus.(θ0)

    adtype = AutoForwardDiff()
    f = OptimizationFunction(loss, adtype)
    prob = OptimizationProblem(f, u0, p)

    alg = LBFGS(linesearch=LineSearches.BackTracking())
    sol = solve(prob,
        alg,
        reltol=1e-4,
        show_trace=true
    )

    θ = softplus.(sol.u)
    return θ
end

function tune_hyperparams_nlls(θ0, p; maxiters=10)
    u0 = invsoftplus.(θ0)

    nlls_prob = NonlinearLeastSquaresProblem(residuals, u0, p)

    sol = solve(nlls_prob, LevenbergMarquardt();
        maxiters, show_trace=Val(true),
        # trace_level=TraceWithJacobianConditionNumber(25)
    )

    θ = softplus.(sol.u)
    return θ
end
