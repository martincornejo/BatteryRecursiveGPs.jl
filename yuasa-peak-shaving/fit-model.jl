function fit_model(u, y, θ, zt, id)
    (; m, c) = id
    model = YuasaModel(θ, u, zt)

    stats = @timed begin
        sol = run_kf!(model, u, y)
    end
    sol = reduce_sol(model, sol)

    return (; model, sol, time = stats.time)
end
