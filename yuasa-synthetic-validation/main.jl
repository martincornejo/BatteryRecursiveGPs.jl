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
using CairoMakie
using Printf

include("fit-model.jl")
include("plots.jl")


# ============================================================
# 1. Load data
# ============================================================

data = CSV.File("data/data-yuasa-synthetic/data-yuasa-synthetic.csv") |> DataFrame
params_real = JSON.parsefile("data/data-yuasa-synthetic/battery-params.json")

f = let
    df = CSV.File("data/ocv/ocv-yuasa.csv") |> DataFrame
    ocv⁻¹ = LinearInterpolation(df.soc, df.ocv)
    ocv = LinearInterpolation(df.ocv, df.soc)
    (; ocv⁻¹, ocv)
end

fR0(s, T; kT = 2000, T0 = 25) = (0.0015 + 0.0004 * sinpi(-0.2 + s * 1.5)) * exp(kT * (1 / (T + 273.15) - 1 / (T0 + 273.15)))
fR025(s) = fR0(s, 25) # reference R0 at 25°C


# ============================================================
# 2. Fit parameters
# ============================================================

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


# ============================================================
# 3. Lab-reference-free identification — composite OCV
# ============================================================
# Joint identification of (composite OCV, Q_i, s0_i) from the per-cell GP
# OCV posteriors alone (no lab reference curve). Uses union-gauge anchoring
# so s=0 and s=1 correspond to the lowest/highest voltage observed across
# all cells — no extrapolation.

begin
    posteriors_comp = Dict(id => extract_posterior(models[id], sols[id]) for id in 1:12)
    ids_comp = sort(collect(keys(posteriors_comp)))
    cells_comp = [(; q = collect(posteriors_comp[id].q), μ = collect(posteriors_comp[id].μ)) for id in ids_comp]
    fit_comp = fit_composite_ocv(cells_comp)
    uq_comp = composite_ocv_uncertainty(fit_comp)
    composite_u = (;
        soc_grid = (fit_comp.Q_common .- fit_comp.Q_at_Vmin) ./ fit_comp.Q_full,
        v_grid = fit_comp.v_grid,
        v_lo = fit_comp.v_grid[1],
        v_hi = fit_comp.v_grid[end],
    )
end;


# Union-gauge → lab-gauge transform (for apples-to-apples tables).
# Clamp into the lab-curve voltage domain — the composite edge voltages are
# the extrema across all cells and can overshoot the lab reference's
# sampled voltage range by a few mV.
(; lab_soc_span, s_lab_lo, s_lab_hi) = let
    V_lab_lo = minimum(f.ocv.u)
    V_lab_hi = maximum(f.ocv.u)
    V_lo_u = clamp(composite_u.v_lo, V_lab_lo, V_lab_hi)
    V_hi_u = clamp(composite_u.v_hi, V_lab_lo, V_lab_hi)
    s_lab_lo = f.ocv⁻¹(V_lo_u)
    s_lab_hi = f.ocv⁻¹(V_hi_u)
    lab_soc_span = s_lab_hi - s_lab_lo
    (; lab_soc_span, s_lab_lo, s_lab_hi)
end

# Per-cell (Q, s0) in the union gauge — the library returns these directly.
param_cells_composite = Dict(
    ids_comp[i] => Dict(:Q => uq_comp.est[i].Q, :soc => uq_comp.est[i].s0)
        for i in eachindex(ids_comp)
)

# Per-cell (Q, s0) back-projected into the lab gauge for direct comparison
# with the existing `param_cells` table.
param_cells = Dict(
    id => let
            Q_u = param_cells_composite[id][:Q]
            s0_u = param_cells_composite[id][:soc]
            Dict(:Q => Q_u / lab_soc_span, :soc => lab_soc_span * s0_u + s_lab_lo)
    end for id in 1:12
)

df_params_comp = params_to_df(param_cells, params_real)


fig_qs = plot_qs_scatter(param_cells, params_real)

fig_compose = plot_composite_ocv(
    posteriors_comp, param_cells, composite_u, f.ocv;
    s_lab_lo, s_lab_hi, lab_soc_span
)

# OCV GP vs reference — estimated and true Q/s0 mapping
fig_ocv_diag = plot_ocv_diagnostics(models, sols, param_cells, params_real, f.ocv)

# ============================================================
# 4. Model accuracy — closed-loop vs open-loop
# ============================================================
# Open-loop: reinitialise the KF from the terminal state of each fitting run,
# then predict the full profile with no measurement corrections (tt=0).

df_volt = voltage_accuracy_to_df(models, sols, data)

# Representative voltage simulation plot (cell 1)
fig_sim = let id = 1
    fig = plot_sim(models[id], sols[id])
    fig.content[1].title = "Cell $(id) — closed-loop"
    fig
end


# ============================================================
# 5. Parameter identification accuracy
# ============================================================

# --- ECM parameters
df_ecm = ecm_params_to_df(models, sols, params_real)

# --- RC and Arrhenius trajectories (cell 1)
fig_rc = let id = 1
    real = params_real["cell_$id"]
    plot_rc_param_trajectory(
        models[id], sols[id];
        r1 = real["R1"], τ1 = real["tau1"]
    )
end

fig_arr = let id = 1
    plot_arrhenius_param_trajectory(models[id], sols[id]; k = 2000)
end

# --- OCV accuracy (fig_ocv_diag already computed in §3)
df_ocv = ocv_residuals_to_df(models, sols, param_cells, params_real, f.ocv)

# --- R0 accuracy
fig_r0 = plot_r0_diagnostics(models, sols, param_cells, fR025)
df_r0 = r0_residuals_to_df(models, sols, param_cells, fR025)


# ============================================================
# 6.  SOC estimation — YuasaStateModel
# ============================================================

θ_state = (; q = (; σ0 = 1.0e-3, σ1 = 0.5e-5), rc = (; σ0 = 1.0e-4, σ1 = 1.0e-4))

soc_models = Dict()
sols_state = Dict()
for id in 1:12
    sm = YuasaStateModel(models[id]; q0 = 0.0, Ts = 1.0, θ = θ_state)
    u, y = cell_dataset(data, id)
    sols_state[id] = run_kf!(sm, u, y)
    soc_models[id] = sm
end


println("\n=== Q/s0 — YuasaStateModel trajectory ===")
df_soc = q_estimation_to_df(data, sols_state, param_cells)

# Representative q/SOC estimation plot (cell 1)
fig_q_state = let id = 1
    fig = plot_q_estimation_state(data, sols_state[id])
    fig.content[1].title = "Cell $(id)"
    fig
end

fig_soc_state = let id = 1
    fig = plot_soc_estimation_state(data, sols_state[id], id, param_cells)
    fig.content[1].title = "Cell $(id)"
    fig
end


# ============================================================
# §7  Module-level evaluation
# ============================================================

begin
    Q_nom = 100.0  # Ah, nominal cell capacity
    Q_pack = calc_Q_pack(param_cells)
    soh_pack = calc_soh_pack(param_cells, Q_nom)
    util_pack = calc_Q_utilization(param_cells)

    @printf("\nPack Q:         %.2f Ah\n", Q_pack)
    @printf("Pack SOH:         %.1f %%\n", soh_pack * 100)
    @printf("Pack utilisation: %.1f %%\n", util_pack * 100)
end

# calc_soc_pack expects Symbol("cell_$i") keys
pack_params_sym = Dict(Symbol("cell_$id") => param_cells[id] for id in 1:12)
fig_module = plot_module_soc(data, pack_params_sym)
