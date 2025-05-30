module battModel

using ForwardDiff
using LinearAlgebra
using DataFrames
using AbstractGPs
using StatsBase
using ComponentArrays

using ..RecursiveGPs


export BATTModel, RTS_Storage
"""
Battery model struct and it's helper functions
"""

mutable struct RTS_Storage
    μ::Vector{Vector{Float64}}
    Σ::Vector{Matrix{Float64}}
    μ_predict::Vector{Vector{Float64}}
    Σ_predict::Vector{Matrix{Float64}}
    A::Vector{Any}
end


mutable struct BATTModel
    """
    Battery model struct:
        - Current implementation enables changing noise and RC parameters
        - Number of RC not tuneable once the struct has been Build
        - In case no RC parameters/set to false or dummy, or wrong specification one RC is assumed with R = 15e-3 and τ = 60
        - model_R and rgp_ocv::Any: Temporal code for testing purposes, to be deleted once adaptive_extended_kf works with RGPs.
    """

    rgp_ocv::Any
    rgp_r::RecursiveGPs.RGPModel
    μ::Vector{Float64}
    Σ::Matrix{Float64}

    μ_params::Vector{Float64}
    Σ_params::Matrix{Float64}

    model_noise::ComponentArray
    param_noise::ComponentArray

    histogram_model::RTS_Storage
    histogram_params::RTS_Storage

    i::Float64
    dt::NamedTuple

    function BATTModel(
        rgp_ocv,
        rgp_r,
        μ_params,
        dt;
        model_R=true
    )

        ## Checking number of RCs
        if size(μ_params, 1) == 2
            N_rc = 1
        elseif size(μ_params, 1) == 4
            @warn "2 RC functions not implemented yet: Set first RC only"
            μ_params = μ_params[1:2]
            N_rc = 1
        else
            @warn "No RC specified or wrong initialization, Set one RC with R = 15e-3 and τ = 60"
            N_rc = 1
            μ_params = [15e-3; 60.0]
        end

        Σ_params = 1e-6 .* I(length(μ_params))

        if model_R
            ## Initializing model
            filler = zeros(size(rgp_ocv.μ, 1), size(rgp_r.μ, 1))
            filler_rc = zeros(N_rc, size(rgp_ocv.μ, 1) + size(rgp_r.μ, 1))

            μ = vcat(rgp_ocv.μ, rgp_r.μ, zeros(N_rc))

            Σ = vcat(
                hcat(rgp_ocv.Σ, filler, zeros(size(rgp_ocv.Σ, 1), N_rc)),
                hcat(filler', rgp_r.Σ, zeros(size(rgp_r.Σ, 1), N_rc)),
                hcat(filler_rc, 1e-4 * I(N_rc)),
            )

            ### Building Model Noise
            # w is motion model noise, only affecting RC
            # v is measurement model noise, affection all
            N_state = size(μ, 1)
            N_basis = size(rgp_ocv.μ, 1) + size(rgp_r.μ, 1)
            N_out = 1
            N_params = size(μ_params)

        else

            filler_rc = zeros(N_rc, size(rgp_ocv.μ, 1))

            μ = vcat(rgp_ocv.μ, zeros(N_rc))

            Σ = vcat(
                hcat(rgp_ocv.Σ, zeros(size(rgp_ocv.Σ, 1), N_rc)),
                hcat(filler_rc, 1e-4 * I(N_rc)),
            )

            N_state = size(μ, 1)
            N_basis = size(rgp_ocv.μ, 1)
            N_out = 1
            N_params = size(μ_params)
        end

        model_noise = ComponentArray(
            w=ComponentArray(
                μ=zeros(N_state),
                Σ=vcat(
                    hcat(zeros(N_basis, N_basis), zeros(N_basis, 1)),
                    hcat(zeros(N_rc, N_basis), 10e-6 .* I(N_rc)),
                )
            ),
            v=ComponentArray(
                μ=zeros(N_out),
                Σ=0.001 .* I(N_out)
            ),
        )

        ## Building param_noise
        w_noise_R = 1e-9
        w_noise_τ = 1e-4

        param_noise = ComponentArray(
            w=ComponentArray(
                μ=zeros(N_params),
                Σ=kron(
                    Diagonal(ones(N_rc)),
                    vcat(
                        hcat(w_noise_R, 0),
                        hcat(0, w_noise_τ)
                    )
                )
            ),
            v=ComponentArray(
                μ=zeros(N_out),
                Σ=1e-5 .* I(N_out)
            )
        )


        ### Building histogram of values for RTS model_rts_smoother
        histogram_model = RTS_Storage(
            Vector{Vector{Float64}}(),
            Vector{Matrix{Float64}}(),
            Vector{Vector{Float64}}(),
            Vector{Matrix{Float64}}(),
            Vector{Matrix{Float64}}()
        )

        histogram_params = RTS_Storage(
            Vector{Vector{Float64}}(),
            Vector{Matrix{Float64}}(),
            Vector{Vector{Float64}}(),
            Vector{Matrix{Float64}}(),
            Vector{Matrix{Float64}}()
        )

        i = 0.0
        new(rgp_ocv, rgp_r, μ, Σ, μ_params, Σ_params, model_noise, param_noise, histogram_model, histogram_params, i, dt)
    end
end


end
