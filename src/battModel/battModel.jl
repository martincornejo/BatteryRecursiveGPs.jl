module battModel

using ForwardDiff
using LinearAlgebra
using DataFrames
using AbstractGPs
using StatsBase
using ComponentArrays

using ..RecursiveGPs


export BATTModel, RTS_Storage, set_mode!, set_v_model_noise!, set_w_model_noise!, set_v_param_noise!, set_w_param_noise!
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
        - mode: 3 Modes: dual_kf, joint_kf or test. Default will always be dual_kf
    """

    rgp_ocv::RecursiveGPs.RGPModel
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
    ts::Float64
    mode::Symbol

    function BATTModel(
        rgp_ocv,
        rgp_r,
        μ_params,
        dt,
        ts
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

        ### Initializing params cov
        Σ_params = 1e-6 .* I(length(μ_params))


        ## Initializing model mean and cov
        μ = vcat(rgp_ocv.μ, rgp_r.μ, zeros(N_rc))

        filler = zeros(size(rgp_ocv.μ, 1), size(rgp_r.μ, 1))
        filler_rc = zeros(N_rc, size(rgp_ocv.μ, 1) + size(rgp_r.μ, 1))

        Σ = vcat(
            hcat(rgp_ocv.Σ, filler, zeros(size(rgp_ocv.Σ, 1), N_rc)),
            hcat(filler', rgp_r.Σ, zeros(size(rgp_r.Σ, 1), N_rc)),
            hcat(filler_rc, 1e-4 * I(N_rc)),
        )

        ### Building Model Noise
        N_state = size(μ, 1)
        N_basis = size(rgp_ocv.μ, 1) + size(rgp_r.μ, 1)
        N_out = 1
        N_params = size(μ_params)

        w_model_noise = 1e-5
        v_model_noise = 1e-3

        model_noise = ComponentArray(
            w=ComponentArray(
                μ=zeros(N_state),
                Σ=vcat(
                    hcat(zeros(N_basis, N_basis), zeros(N_basis, 1)),
                    hcat(zeros(N_rc, N_basis), w_model_noise .* I(N_rc)),
                )
            ),
            v=ComponentArray(
                μ=zeros(N_out),
                Σ=v_model_noise .* I(N_out)
            ),
        )

        ## Building param_noise
        w_noise_R = 1e-9
        w_noise_τ = 1e-4
        v_param_noise = v_model_noise

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
                Σ=v_param_noise .* I(N_out)
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
        mode = :dual_kf

        @info "Initializing batt Model with:"
        @info "mode = dual_kf"
        @info "parameter w noise: R1 = $(w_noise_R), τ1 =  $(w_noise_τ)"
        @info "parameter v noise: $(v_param_noise)"
        @info "model Vrc w noise : $(w_model_noise)"
        @info "model v noise: $(v_model_noise)"
        new(rgp_ocv, rgp_r, μ, Σ, μ_params, Σ_params, model_noise, param_noise, histogram_model, histogram_params, i, dt, ts, mode)
    end
end



### Changing mode of battery
function set_mode!(batt, mode)
    """
    For before training use. Sets the battery model in one of three modes:
        - dual_kf: RC parameters and voltage are separated states
        - join_kf: RC parameters and voltage are on joint state
        - test: Dual_kf with R0 as Constant
    """
    if mode == :dual_kf
        set_dual_kf_mode!(batt)

    elseif mode == :join_kf
        set_join_kf_mode!(batt)

    elseif mode == :test
        set_test_mode!(batt)

    else
        error("Specified mode does not exist: Choose 'dual_kf', 'join_kf' or 'test'")
    end
    batt.mode = mode

end



function set_dual_kf_mode!(batt)
    """
    Sets the model in dual Kalman filter states
    """
    rgp_ocv = batt.rgp_ocv
    rgp_r = batt.rgp_r
    N_rc = 1

    batt.μ = vcat(rgp_ocv.μ, rgp_r.μ, zeros(N_rc))

    filler = zeros(size(rgp_ocv.μ, 1), size(rgp_r.μ, 1))
    filler_rc = zeros(N_rc, size(rgp_ocv.μ, 1) + size(rgp_r.μ, 1))

    batt.Σ = vcat(
        hcat(rgp_ocv.Σ, filler, zeros(size(rgp_ocv.Σ, 1), N_rc)),
        hcat(filler', rgp_r.Σ, zeros(size(rgp_r.Σ, 1), N_rc)),
        hcat(filler_rc, 1e-4 * I(N_rc)),
    )
end

function set_join_kf_mode!(batt)
    """
    Sets the model to assume join state
    """
    rgp_ocv = batt.rgp_ocv
    rgp_r = batt.rgp_r
    μ_params = copy(batt.μ_params)
    N_params = size(μ_params, 1)
    N_rc = 1
    N_basis = size(rgp_ocv.μ, 1) + size(rgp_r.μ, 1)
    N_state = N_basis + N_rc + N_params
    N_out = 1

    ## Making join state

    batt.μ = vcat(rgp_ocv.μ, rgp_r.μ, zeros(N_rc), μ_params)

    Σ_params = copy(batt.Σ_params)
    filler = zeros(size(rgp_ocv.μ, 1), size(rgp_r.μ, 1))
    filler_rc = zeros(N_rc, size(rgp_ocv.μ, 1) + size(rgp_r.μ, 1))

    batt.Σ = vcat(
        hcat(rgp_ocv.Σ, filler, zeros(size(rgp_ocv.Σ, 1), N_rc + N_params)),
        hcat(filler', rgp_r.Σ, zeros(size(rgp_r.Σ, 1), N_rc + N_params)),
        hcat(filler_rc, rc_Σ * I(N_rc), zeros(N_rc, N_params)),
        hcat(zeros(N_params, N_basis + N_rc), Σ_params)
    )

    ## Updating model_noise
    w_noise_R = 1e-9
    w_noise_τ = 1e-4
    v_param_noise = 1e-3
    w_Σ_params = kron(
        Diagonal(ones(N_rc)),
        vcat(
            hcat(w_noise_R, 0),
            hcat(0, w_noise_τ)
        )
    )

    batt.model_noise = ComponentArray(
        w=ComponentArray(
            μ=zeros(N_state),
            Σ=vcat(
                hcat(zeros(N_basis, N_basis), zeros(N_basis, N_rc + N_params)),
                hcat(zeros(N_rc, N_basis), 1e-5 .* I(N_rc), zeros(N_rc, N_params)),
                hcat(zeros(N_params, N_basis + N_rc), w_Σ_params)
            )
        ),
        v=ComponentArray(
            μ=zeros(N_out),
            Σ=v_param_noise .* I(N_out)
        ),
    )


end

function set_test_mode!(batt)
    """
    Sets the model with dual kalman filter state, assuming constant R0
    """
    rgp_ocv = batt.rgp_ocv
    N_rc = 1
    filler_rc = zeros(N_rc, size(rgp_ocv.μ, 1))

    batt.μ = vcat(rgp_ocv.μ, zeros(N_rc))

    batt.Σ = vcat(
        hcat(rgp_ocv.Σ, zeros(size(rgp_ocv.Σ, 1), N_rc)),
        hcat(filler_rc, 1e-4 * I(N_rc)),
    )

end



### Setters of battModel noises

function set_w_param_noise!(batt, w_param_noise)
    @assert size(batt.param_noise.w.Σ) == size(w_param_noise) "Shape mismatch: expected $(size(batt.param_noise.w.Σ)), got $(size(w_param_noise))"

    if batt.mode == :join_kf
        N_model = size(batt.rgp_ocv.μ, 1) + size(batt.rgp_r.μ, 1) + 1
        batt.param_noise.w.Σ = w_param_noise
        batt.model_noise.w.Σ[N_model+1:end, N_model+1:end] = w_param_noise
    else
        batt.param_noise.w.Σ = w_param_noise
    end
end

function set_v_param_noise!(batt, v_param_noise)
    @assert size(batt.param_noise.v.Σ) == size(v_param_noise) "Shape mismatch: expected $(size(batt.param_noise.v.Σ)), got $(size(v_param_noise))"

    batt.param_noise.v.Σ = v_param_noise
    batt.model_noise.v.Σ = v_param_noise

end

function set_w_model_noise!(batt, w_noise)
    @assert size(batt.param_noise.w.Σ) == size(w_noise) "Shape mismatch: expected $(size(batt.model_noise.w.Σ)), got $(size(w_noise))"
    batt.model_noise.w.Σ = w_noise
end

function set_v_model_noise!(batt, v_noise)
    @assert size(batt.param_noise.v.Σ) == size(v_noise) "Shape mismatch: expected $(size(batt.model_noise.v.Σ)), got $(size(v_noise))"
    batt.param_noise.v.Σ = v_noise
    batt.model_noise.v.Σ = v_noise
end
end