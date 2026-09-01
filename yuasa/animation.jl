using YuasaAnalysis
using BatteryRecursiveGPs

using DataFrames, Dates, Intervals
using CairoMakie


# === Data ===

datadir = "yuasa/data/cycles/"
paramdir = "yuasa/data/hyperparams/"
id = (; p = 1, m = 1, c = 1)

data = load_dataset(datadir; signals = (:cell_voltage, :module_current, :battery_temperature))
ti = Interval(DateTime("2025-12-10T14:00:20"), DateTime("2025-12-11T02:30:20"))


# === Model fit ===

cell_ϑ = load_hyperparams(paramdir * "cell_hyperparams.json", [id])

zt = fit_zscore()
(; u, y) = cell_dataset(data, ti, id.p, id.m, id.c; zt)

θ = scale_θ(u, y, cell_ϑ[id])
model = YuasaModel(θ, u, zt)
prior = (; x = copy(model.kf.x), R = copy(model.kf.R))  # frame 1: before any observation

@time sol_full = run_kf!(model, u, y);


# === Animation ===

export_anim = false

if export_anim
    animate_model("figs/ecm-learning-P$(id.p)M$(id.m)C$(id.c).mp4", model, sol_full; step = 10, framerate = 24, prior)
end
