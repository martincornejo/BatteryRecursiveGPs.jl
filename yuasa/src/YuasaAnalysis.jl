module YuasaAnalysis

using DataFrames
using JSON
using Dates
using DataInterpolations
using StatsBase
using Statistics
using Measurements
using Printf
using CairoMakie
using ColorSchemes
using StaticArrays
using BatteryRecursiveGPs
import ComponentArrays: ComponentVector, ComponentMatrix
using LinearAlgebra: diag
using RecursiveGPs: predict_gp
using Distributed: remotecall

include("model.jl")
include("ocv.jl")
include("hyperparams.jl")
include("analysis.jl")
include("plot.jl")

# data preparation + model fitting (model.jl)
export fit_zscore, scale_θ, cell_dataset, module_dataset, cell_dataset_osci
export load_hyperparams, fit_cells, fit_modules, fit_soc_models, eval_models

# OCV reconstruction validation (ocv.jl)
export clean_ocv, average_charge_discharge, validate_cell_ocvs, eval_cell_ocv_validation
export compare_current_sources

# GP hyperparameter selection (hyperparams.jl)
export select_hyperparams, build_hyperparam_export, selection_counts
export calc_scaled_hyperparams, calc_hyperparam_selection

# analysis / metrics (analysis.jl)
export calc_v_summary, calc_module_v_summary
export calc_module_soh_summary, calc_composite_rmse
export calc_soc_trajectories, calc_module_soc, calc_soc_error, calc_module_soc_summary
export calc_charge_accuracy, calc_charge_error, calc_soc_diagnostic, eval_soc_range
export calc_throughput, calc_data_completeness

# figures (plot.jl)
export plot_dataset_overview, plot_cell_voltage_system, plot_data_resolution
export plot_ecms_comparison, plot_sim, plot_q_estimation
export plot_v_accuracy, plot_v_accuracy_overview
export plot_cell_soh, plot_cell_soh_hist, plot_module_summary
export plot_soc_discrepancy, plot_soc_discrepancy_heatmap, plot_soc_comparison, plot_soc_overview
export plot_charge_error, plot_soc_diagnostic
export plot_cell_ocv_validation, plot_ocv_cleaning, plot_ocv_extrapolation
export plot_soh_heatmap, plot_composite_ocv, plot_module_soh, plot_module_inhomogeneity
export plot_hyperparam_selection, plot_hyperparam_scales

end # module YuasaAnalysis
