module YuasaAnalysis

using QuackIO
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

include("plot/theme.jl")
include("plot/datasets.jl")
include("plot/ecm.jl")
include("plot/ecm_animation.jl")
include("plot/soh.jl")
include("plot/soc.jl")
include("plot/gp_hyperparams.jl")
include("plot/validation.jl")

# data preparation + model fitting (model.jl)
export load_dataset
export fit_zscore, scale_θ, cell_dataset, module_dataset, cell_dataset_osci
export fit_cells, fit_modules, fit_soc_models, eval_models

# OCV reconstruction validation (ocv.jl)
export build_reference_curves
export calc_ocv_shape_validation, calc_ocv_curves, calc_reference_floor
export calc_soh_validation, calc_validation_summary
export validate_module, calc_validation_table, calc_pooled_validation
export build_validation_export, load_validation_export
export soc_gauge, compare_current_sources

# GP hyperparameter selection (hyperparams.jl)
export select_hyperparams, build_hyperparam_export, load_hyperparams, selection_counts
export calc_scaled_hyperparams, calc_hyperparam_selection

# analysis / metrics (analysis.jl)
export calc_v_summary, calc_module_v_summary
export calc_module_soh_summary, calc_cell_spread, calc_composite_rmse
export calc_soc_trajectories, calc_module_soc, calc_soc_error, calc_module_soc_summary
export cell_capacities, calc_charge_accuracy, calc_charge_error, calc_soc_diagnostic, eval_soc_range
export calc_throughput, calc_data_completeness
export calc_ecm_parameters, calc_parameter_summary

# figures (plot/*.jl)
export plot_dataset_overview, plot_cell_voltage_system, plot_module_data, plot_data_resolution
export plot_ecms_comparison, plot_sim, plot_q_estimation
export plot_v_accuracy, plot_v_accuracy_overview
export plot_cell_soh, plot_cell_soh_hist, plot_ecm_parameters, plot_module_summary
export plot_soc_discrepancy, plot_soc_discrepancy_heatmap, plot_soc_comparison, plot_soc_overview
export plot_charge_error, plot_soc_diagnostic
export plot_validation, plot_validation_rmse, plot_ocv_cleaning, plot_ocv_extrapolation
export animate_model
export plot_soh_heatmap, plot_composite_ocv, plot_module_soh, plot_module_inhomogeneity
export plot_hyperparam_selection, plot_hyperparam_scales

end # module YuasaAnalysis
