using QuackIO
using DataFrames
using DataInterpolations
using LinearAlgebra
using Measurements
using Printf
using Statistics
using Dates

using BatteryDigitalTwin: fit_composite_ocv

# using GLMakie
using CairoMakie

include("ocv.jl")
include("analysis.jl")
include("plot.jl")


# === Load datasets ===

df_d = read_parquet(DataFrame, "data/yuasa-ocv-test/combined_log_20260131_200948.parquet")
df_c = read_parquet(DataFrame, "data/yuasa-ocv-test/combined_log_20260201_181000.parquet")

plot_module_dataset(df_d, 7)

# === Current sensor comparison: CMU vs. Oscilloscope ===

# calculate oscilloscope offsets
tek_offset_d = find_tek_offset(df_d)
tek_offset_c = find_tek_offset(df_c)
compare_current_sources(df_d; tek_offset = tek_offset_d) |> display
compare_current_sources(df_c; tek_offset = tek_offset_c) |> display


# === Build composite OCV ===

per_cell = map(1:12) do c
    id = (; m = 7, c)
    fc = clean_ocv(df_c, id; dch = false, current_col = "tek", tek_offset = tek_offset_c)
    fd = clean_ocv(df_d, id; dch = true, current_col = "tek", tek_offset = tek_offset_d)
    avg = average_charge_discharge(fc, fd)
    ocv_vq = LinearInterpolation(avg.μ, avg.q; extrapolation = ExtrapolationType.Constant)
    (; q = avg.q, μ = avg.μ, ocv = ocv_vq)
end

cells = [(; q = c.q, μ = c.μ) for c in per_cell]
ocvs = [c.ocv for c in per_cell]

fit = fit_composite_ocv(cells)

composite = LinearInterpolation(
    fit.v_grid, fit.soc_grid;
    extrapolation = ExtrapolationType.Constant
)
params = fit.params
ocv_smooth = smooth_ocv(composite)


# === Evaluate and plot ===

eval_fit_parameters(fit)
eval_cell_parameters(composite, fit; V_ref = (3.3, 4.05), soc_ref = (0.05, 0.95))
eval_ocv_residuals(composite, ocvs, params)
eval_soc_range(composite)

plot_composite_ocv(composite, ocvs, params) |> display
plot_ocv_residuals(composite, ocvs, params) |> display
plot_ocv_extrapolation(composite) |> display
