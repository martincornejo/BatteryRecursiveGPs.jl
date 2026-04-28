function default_θ()
    return (;
        ocv = (; σ = 0.5, ℓ = 0.5),
        r0 = (; σ = 0.01, ℓ = 2.0),
        vσ = 3.0e-3,
        Ts = 1.0,
        r0μ = 1.5e-3,
        rc = (;
            v0 = 0.0, σ0_v = 1.0e-5, σ1_v = 5.0e-5,
            r0 = 1.5e-3, σ0_r = 10.0e-6, σ1_r = 0.0,
            τ0 = 250.0, σ0_τ = 10.0, σ1_τ = 0.0,
        ),
        cc = (; σ1 = 0.1e-5),
        arr = (; T0 = 25, k0 = 20, σ0_k = 0.0, σ1_k = 0.0),
    )
end

function fit_cells(data, ti, ids; θ = default_θ(), zt = fit_zscore(), distributed = true)
    make_uy = id -> cell_dataset(data, ti, zt, id.m, id.c)
    runner = distributed ? fit_models_distributed : fit_models_threaded
    return runner(YuasaModel, make_uy, ids, θ, zt)
end
