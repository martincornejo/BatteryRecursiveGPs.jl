using LowLevelParticleFilters
using Distributions
using LinearAlgebra
using MLUtils: DataLoader
using DataFrames
using CSV
using Statistics
using AbstractGPs
using DataInterpolations
using CairoMakie
using StatsBase
using BenchmarkTools
using Profile
using Revise
using StaticArrays
using Optim
using UnPack
using ForwardDiff
using PreallocationTools
using Dates
using ProgressMeter
includet("../battModel/rc.jl")
includet("../battModel/r0_ocv.jl")
includet("../battModel/batt.jl")
includet("../battModel/rgp.jl")
includet("utils.jl")
include("data.jl")
import ComponentArrays: ComponentVector, getaxes, ComponentMatrix, @static_unpack

files = Dict(
    :phase_current => "data_bmw/phase_current_sbc-stage_container_bmw_1718973873-1719079383.csv",
    :module_current => "data_bmw/module_current_sbc-stage_container_bmw_1718973873-1719079383.csv",
    :module_voltage => "data_bmw/module_voltage_sbc-stage_container_bmw_1718973873-1719079383.csv",
    :module_temperature => "data_bmw/module_temperature_sbc-stage_container_bmw_1718973873-1719079383.csv",
    :cell_voltage => "data_bmw/cell_voltage_sbc-stage_container_bmw_1718973873-1719079383.csv"
)

t0 = DateTime("2024-06-21T12")
data = load_dataset(files, t0)

ti = DateTime("2024-06-21T12:45"):Second(1):DateTime("2024-06-22T18")

ids = [(; p=2, m=1)]#, (; p=2, m=3), (; p=2, m=5)]

m = fit_modules(data, ti, t0, ids, N_points=20_000, cells=1:1)


begin
    f = Figure(size=(1200, 1200))
    for (i, id) in enumerate(ids)
        m_ = m[id]
        for (j, cell) in enumerate(m_)
            cell = m_[:1]
            f = plot_ocv(cell, i, j, f=f, fig_x=j, fig_y=i)
        end

    end
    display(f)

end

begin
    f = Figure(size=(1200, 1200))
    for (i, id) in enumerate(ids)
        m_ = m[id]
        for (j, cell) in enumerate(m_)
            cell = m_[:1]
            f = plot_r0(cell, i, j, f=f, fig_x=j, fig_y=i)
        end

    end
    display(f)

end

begin
    for (i, id) in enumerate(ids)
        m_ = m[id]
        for (j, cell) in enumerate(m_)
            cell = m_[:1]
            f = plot_rc(cell)
        end

    end
end




