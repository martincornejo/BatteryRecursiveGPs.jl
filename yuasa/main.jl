using YuasaAnalysis
using BatteryRecursiveGPs

using DataFrames, Dates, Intervals
using JSON
using CairoMakie

# === Data ===

# cycling dataset
datadir = "yuasa/data/cycles/"
paramdir = "yuasa/data/hyperparams/"
valdir = "yuasa/data/validation/"
data = load_dataset(datadir; oscilloscope = true)
ti = Interval(DateTime("2025-12-10T14:00:20"), DateTime("2025-12-11T02:30:20"))

df_completeness = calc_data_completeness(data, ti)  # data availability per signal

# the two modules carried through the paper as examples: one nominal, one outlier
id_norm = (; p = 3, m = 7)
id_out = (; p = 3, m = 5)

fig_dataset = plot_dataset_overview(data; id_norm, id_out)                        # Fig 1
fig_cell_voltages = plot_cell_voltage_system(data)                                # Fig S1
fig_module_data = plot_module_data(data)
fig_data_resolution = plot_data_resolution(data; completeness = df_completeness)  # Fig S2


# === Model fits ===

cell_ids = [(; p, m, c) for p in 1:3, m in 1:9, c in 1:12] |> vec |> sort
cell_ϑ = load_hyperparams(paramdir * "cell_hyperparams.json", cell_ids)
@time (; cell_models, cell_sols) = fit_cells(data, cell_ϑ, ti, cell_ids);

module_ids = [(; p, m) for p in 1:3, m in 1:9] |> vec |> sort
module_ϑ = load_hyperparams(paramdir * "module_hyperparams.json", module_ids)
@time (; module_models, module_sols) = fit_modules(data, module_ϑ, ti, module_ids);

fig_ecms = plot_ecms_comparison(cell_models, cell_sols, module_models, module_sols; n_mod = 12)  # Fig 2

ecm_cell = calc_ecm_parameters(cell_models, cell_sols, cell_ids)
ecm_module = calc_ecm_parameters(module_models, module_sols, module_ids; n = 12)

fig_ecm_params = plot_ecm_parameters(ecm_cell, ecm_module)


# === Voltage accuracy ===

@time cell_sols_ol = eval_models(cell_models, cell_sols, cell_ids);
@time module_sols_ol = eval_models(module_models, module_sols, module_ids);

df_v_cell = calc_v_summary(cell_models, cell_sols_ol, cell_ids)
df_v_module = calc_module_v_summary(cell_models, cell_sols_ol, module_models, module_sols_ol, module_ids)

fig_v_overview = let id = (; p = 3, m = 4, c = 2)  # Fig 7: example fit (A) + system-wide accuracy (B)
    plot_v_accuracy_overview(cell_models[id], cell_sols_ol[id], df_v_cell, df_v_module; title = "Cell P$(id.p)M$(id.m)C$(id.c)")
end


# === SOH estimation ===

# reconstructed OCV posteriors
cell_ocvs = [gp_ocv(cell_models[id], cell_sols[id]) for id in cell_ids];
module_ocvs = [gp_ocv(module_models[id], module_sols[id]) for id in module_ids];

# Two voltage–SOC reference points that put the composite fit in absolute units. They are the
# whole absolute capacity scale — `Q_i = charge_i(v_low → v_high) / (soc_high − soc_low)`, one
# global gain — so they are read off the low-power rig reference rather than asserted here.
# `reference.jl` derives them; re-run it with `export_json = true` to refresh the file.
refs_cell = (; (Symbol(k) => v for (k, v) in JSON.parsefile(valdir * "reference_refs.json"))...)
refs_module = (; refs_cell..., v_low = 12 * refs_cell.v_low, v_high = 12 * refs_cell.v_high)

# align cells to extract SOH and initial SOC
cell_fit = fit_composite_ocv(cell_ocvs, refs_cell)
module_fit = fit_composite_ocv(module_ocvs, refs_module)
df_soh = calc_module_soh_summary(cell_ids, cell_fit, module_ids, module_fit)

# per-module spread of the per-cell fit in physical units: SOH as capacity (Ah), initial SOC (%)
df_spread = calc_cell_spread(cell_fit, cell_ids)

# per-cell voltage misfit to the composite consensus OCV (mV): fit quality of the alignment
df_comp_rmse = calc_composite_rmse(cell_fit, cell_ocvs, cell_ids)

fig_cell_soh = plot_cell_soh(cell_fit, cell_ocvs)  # Fig 3
fig_module_soh = plot_module_summary(df_soh)  # Fig 4
fig_cell_soh_hist = plot_cell_soh_hist(cell_fit)


# === SOC estimation ===

# closed-loop run: frozen ECM parameters, 2-state EKF estimates charge + RC voltage
# tuning in physical units: q in Ah, rc in V
θ_soc_cell = (; q = (; σ0 = 0.3, σ1 = 3.0e-4), rc = (; σ0 = 25.0e-3, σ1 = 50.0e-6))
θ_soc_module = (; q = (; σ0 = 0.3, σ1 = 3.0e-4), rc = (; σ0 = 12 * 25.0e-3, σ1 = 12 * 50.0e-6))
@time cell_soc = fit_soc_models(cell_models, cell_sols, cell_ids; θ = θ_soc_cell);
@time module_soc = fit_soc_models(module_models, module_sols, module_ids; θ = θ_soc_module);

tg = 0:60:45000  # common time grid / s
soc_cell = calc_soc_trajectories(cell_soc.models, cell_soc.sols, cell_fit, cell_ids; tg)
soc_module = calc_soc_trajectories(module_soc.models, module_soc.sols, module_fit, module_ids; tg)

soc_err = calc_soc_error(soc_cell, soc_module, cell_fit, cell_ids, module_ids)
df_soc = calc_module_soc_summary(soc_err, module_ids)
df_soc_norm = calc_module_soc(id_norm, tg, soc_cell, soc_module, cell_fit, cell_ids, module_ids)
df_soc_out = calc_module_soc(id_out, tg, soc_cell, soc_module, cell_fit, cell_ids, module_ids)

fig_soc_overview = plot_soc_overview(
    df_soc_norm, df_soc_out, soc_err;
    titles = ("Module P$(id_norm.p)M$(id_norm.m)", "Module P$(id_out.p)M$(id_out.m)"),
)
fig_soc_heatmap = plot_soc_discrepancy_heatmap(tg, soc_err)  # Fig S3


# === EKF vs Coulomb counting ===

# validate the EKF charge estimate against the Coulomb counting baseline
p1m9_ids = [(; p = 1, m = 9, c) for c in 1:12]
Q_p1m9 = cell_capacities(cell_fit, cell_ids, p1m9_ids)
df_q_acc = calc_charge_accuracy(cell_soc.models, cell_soc.sols, data, ti, p1m9_ids, Q_p1m9)
df_q_err = calc_charge_error(cell_soc.models, cell_soc.sols, data, ti, p1m9_ids, Q_p1m9; tg)

# evaluate EKF with current sensor fault scenario
id_diag = (; p = 1, m = 9, c = 2)
soc_scenarios = ((; offset = 0.0, bias = 0.0), (; offset = 3.0, bias = 0.1))  # Ah, A
soc_diag = calc_soc_diagnostic(cell_models[id_diag], cell_sols[id_diag], data, ti, id_diag.c, soc_scenarios; θ = θ_soc_cell)

fig_charge_error = plot_charge_error(df_q_err)  # charge accuracy vs oscilloscope (SOC %)
fig_soc_diag = plot_soc_diagnostic(soc_diag)  # fault rejection: EKF vs CC


# === Export ===
export_figs = false
export_json = false

# for validation.jl — every cell's (Q_cell, s0) and reconstructed OCV from the 324-cell joint fit.
# The rig reference is cycling phase 3 (P3) in module order (rig m ≡ P3Mm, NOT the oscilloscope module
# P1M9 above — the module arrangement changed between the experiments), so validation.jl slices
# phase 3 out of this; exporting all cells keeps cross-module checks possible without refitting.
if export_json
    validation_export = build_validation_export(cell_fit, cell_ocvs, cell_ids, cell_ids, refs_cell)
    write(valdir * "validation_cells.json", JSON.json(validation_export, 2))
end

if export_figs
    save("figs/dataset-modules.pdf", fig_dataset)
    save("figs/dataset-modules-overlay.pdf", fig_module_data)
    save("figs/dataset-cell-voltages.pdf", fig_cell_voltages)
    save("figs/dataset-resolution.pdf", fig_data_resolution)
    save("figs/ecms.pdf", fig_ecms)
    save("figs/ecm-parameters.pdf", fig_ecm_params)
    save("figs/model-voltage-accuracy.pdf", fig_v_overview)
    save("figs/cell-soh.pdf", fig_cell_soh)
    save("figs/module-soh.pdf", fig_module_soh)
    save("figs/cell-soh-hist.pdf", fig_cell_soh_hist)
    save("figs/soc-ekf-diagnostic.pdf", fig_soc_diag)
    save("figs/ekf-charge-error.pdf", fig_charge_error)
    save("figs/soc-error.pdf", fig_soc_overview)
    save("figs/soc-discrepancy-heatmap.pdf", fig_soc_heatmap)
end
