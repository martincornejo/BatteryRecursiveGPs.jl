using BatteryDigitalTwin
using RecursiveGPs: predict_gp
using CSV
using JSON
using DataFrames
using StatsBase
using Measurements
using DataInterpolations
using StaticArrays
using LinearAlgebra
import ComponentArrays: ComponentVector, ComponentMatrix
using Printf

include("fit-model.jl")


# ============================================================
# 1. Load data & fit KF models (shared by both approaches)
# ============================================================

data = CSV.File("data/data-yuasa-synthetic/data-yuasa-synthetic.csv") |> DataFrame
params_real = JSON.parsefile("data/data-yuasa-synthetic/battery-params.json")

f = let
    df = CSV.File("data/ocv/ocv-yuasa.csv") |> DataFrame
    ocv⁻¹ = LinearInterpolation(df.soc, df.ocv)
    ocv = LinearInterpolation(df.ocv, df.soc)
    (; ocv⁻¹, ocv)
end

θ = ComponentVector(;
    ocv = (; σ = 0.5, ℓ = 0.3),
    r0 = (; σ = 0.5, ℓ = 2.0),
    vσ = 2.0e-3, Ts = 1.0,
    r0μ = 1.5e-3,
    rc = (;
        v0 = 0.0, σ0_v = 1.0e-5, σ1_v = 5.0e-5,
        r0 = 1.5e-3, σ0_r = 5.0e-6, σ1_r = 0.0,
        τ0 = 250.0, σ0_τ = 1.0, σ1_τ = 0.0,
    ),
    cc = (; σ0 = 0.0, σ1 = 0.1e-5),
    arr = (; T0 = 25, k0 = 1500, σ0_k = 20, σ1_k = 0.0),
)

(; models, sols) = fit_models(data, 1:12, θ);

posteriors = Dict(id => extract_posterior(models[id], sols[id]) for id in 1:12)
ids = sort(collect(keys(posteriors)))
cells = [(; q = collect(posteriors[id].q), μ = collect(posteriors[id].μ)) for id in ids]


# ============================================================
# 2. Run both composite OCV methods
# ============================================================

fit_em = fit_composite_ocv(cells)
uq_em = composite_ocv_uncertainty(fit_em)

fit_pw = fit_composite_ocv_pairwise(cells)
uq_pw = composite_ocv_pairwise_uncertainty(fit_pw)


# ============================================================
# 3. Lab-gauge transform (shared helper)
# ============================================================

function to_lab_gauge(fit, uq, ids, f)
    V_lab_lo = minimum(f.ocv.u)
    V_lab_hi = maximum(f.ocv.u)
    v_lo = fit.v_grid[1]
    v_hi = fit.v_grid[end]
    V_lo_u = clamp(v_lo, V_lab_lo, V_lab_hi)
    V_hi_u = clamp(v_hi, V_lab_lo, V_lab_hi)
    s_lab_lo = f.ocv⁻¹(V_lo_u)
    s_lab_hi = f.ocv⁻¹(V_hi_u)
    lab_soc_span = s_lab_hi - s_lab_lo

    param_cells = Dict(
        ids[i] => let
                Q_u = uq.est[i].Q
                s0_u = uq.est[i].s0
                Dict(:Q => Q_u / lab_soc_span, :soc => lab_soc_span * s0_u + s_lab_lo)
        end for i in eachindex(ids)
    )
    return param_cells
end

pc_em = to_lab_gauge(fit_em, uq_em, ids, f)
pc_pw = to_lab_gauge(fit_pw, uq_pw, ids, f)


# ============================================================
# 4. Compare
# ============================================================

println("\n" * "="^80)
println("  COMPOSITE OCV COMPARISON — EM (intersection) vs Pairwise (per-pair overlap)")
println("="^80)

println("\n--- Fit summary ---")
@printf("  EM:       rss = %.6e   iters = %d\n", fit_em.rss, fit_em.n_iters_run)
@printf("  Pairwise: rss = %.6e   σ²_hat = %.6e\n", fit_pw.rss, uq_pw.σ²_hat)
@printf("  EM UQ:    σ²_hat = %.6e\n", uq_em.σ²_hat)

println("\n--- Per-cell Q (Ah) — lab gauge ---")
@printf("  %4s  %8s  %10s  %10s  %10s  %10s\n",
    "Cell", "Q_true", "Q_em", "Q_pw", "err_em", "err_pw")
for id in 1:12
    Q_true = params_real["cell_$id"]["Q"]
    Q_em = Measurements.value(pc_em[id][:Q])
    Q_pw = Measurements.value(pc_pw[id][:Q])
    σ_em = Measurements.uncertainty(pc_em[id][:Q])
    σ_pw = Measurements.uncertainty(pc_pw[id][:Q])
    @printf("  %4d  %8.3f  %7.3f±%.2f  %7.3f±%.2f  %+8.3f  %+8.3f\n",
        id, Q_true, Q_em, σ_em, Q_pw, σ_pw, Q_em - Q_true, Q_pw - Q_true)
end

println("\n--- Per-cell s0 (%) — lab gauge ---")
@printf("  %4s  %8s  %10s  %10s  %10s  %10s\n",
    "Cell", "s0_true", "s0_em", "s0_pw", "err_em", "err_pw")
for id in 1:12
    s0_true = params_real["cell_$id"]["soc"] * 100
    s0_em = Measurements.value(pc_em[id][:soc]) * 100
    s0_pw = Measurements.value(pc_pw[id][:soc]) * 100
    σ_em = Measurements.uncertainty(pc_em[id][:soc]) * 100
    σ_pw = Measurements.uncertainty(pc_pw[id][:soc]) * 100
    @printf("  %4d  %8.2f  %6.2f±%.2f  %6.2f±%.2f  %+7.2f  %+7.2f\n",
        id, s0_true, s0_em, σ_em, s0_pw, σ_pw, s0_em - s0_true, s0_pw - s0_true)
end

println("\n--- Aggregate errors ---")
Q_err_em = [Measurements.value(pc_em[id][:Q]) - params_real["cell_$id"]["Q"] for id in 1:12]
Q_err_pw = [Measurements.value(pc_pw[id][:Q]) - params_real["cell_$id"]["Q"] for id in 1:12]
s0_err_em = [(Measurements.value(pc_em[id][:soc]) - params_real["cell_$id"]["soc"]) * 100 for id in 1:12]
s0_err_pw = [(Measurements.value(pc_pw[id][:soc]) - params_real["cell_$id"]["soc"]) * 100 for id in 1:12]

@printf("  Q  RMSE:  EM = %.4f Ah   Pairwise = %.4f Ah\n",
    sqrt(mean(abs2, Q_err_em)), sqrt(mean(abs2, Q_err_pw)))
@printf("  Q  MAE:   EM = %.4f Ah   Pairwise = %.4f Ah\n",
    mean(abs, Q_err_em), mean(abs, Q_err_pw))
@printf("  s0 RMSE:  EM = %.3f %%    Pairwise = %.3f %%\n",
    sqrt(mean(abs2, s0_err_em)), sqrt(mean(abs2, s0_err_pw)))
@printf("  s0 MAE:   EM = %.3f %%    Pairwise = %.3f %%\n",
    mean(abs, s0_err_em), mean(abs, s0_err_pw))

println("\n--- Max |Q_em - Q_pw| and |s0_em - s0_pw| (method disagreement) ---")
dQ = [abs(Measurements.value(pc_em[id][:Q]) - Measurements.value(pc_pw[id][:Q])) for id in 1:12]
ds = [abs(Measurements.value(pc_em[id][:soc]) - Measurements.value(pc_pw[id][:soc])) * 100 for id in 1:12]
@printf("  max |ΔQ| = %.4f Ah    max |Δs0| = %.3f %%\n", maximum(dQ), maximum(ds))
