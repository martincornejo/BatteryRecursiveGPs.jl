
softplus(x) = log(1 + exp(x))
inv_softplus(x) = log(exp(x) - 1)


function run_sim!(kf, us, ys, ut;predict_fun = model_predict, step_size = 1)
    vμ = Float64[]
    vσ = Float64[]
    evoμ = []
    evoΣ = []
    (; xid, Σid) = kf.p

    n = 1
    for (u, y) in zip(us, ys)
        (vμᵢ, vσᵢ) = predict_fun(kf, u)
        push!(vμ, vμᵢ)
        push!(vσ, vσᵢ)
        kf(u, y)
        if n % step_size == 0
            push!(evoμ, copy(ComponentVector(kf.x, xid)))
            push!(evoΣ, copy(ComponentMatrix(kf.R, Σid)))
        end
        n += 1
    end

    for u in ut
        vμᵢ, vσᵢ = predict_fun(kf, u)
        push!(vμ, vμᵢ)
        push!(vσ, vσᵢ)

        LLPF.predict!(kf, u)
    end
    v_sim = (; vμ, vσ)
    evo = (;evoμ, evoΣ)
    
    (; v_sim, evo)
end



function predict_kf(kf::LowLevelParticleFilters.AbstractExtendedKalmanFilter{IPD}, u, p=LowLevelParticleFilters.parameters(kf), t::Real=index(kf) * kf.Ts; R1=LowLevelParticleFilters.get_mat(kf.R1, kf.x, u, p, t), α=kf.α) where IPD
    (; x, R) = kf
    A = kf.Ajac(x, u, p, t)
    if IPD
        x⁻ = similar(x)
        kf.dynamics(x⁻, x, u, p, t)
    else
        x⁻ = kf.dynamics(x, u, p, t)
    end
    if α == 1
        Σ⁻ = LowLevelParticleFilters.symmetrize(A * R * A') + R1
    else
        Σ⁻ = LowLevelParticleFilters.symmetrize(α * A * R * A') + R1
    end
    (x⁻, Σ⁻)
end

function measurement_kf(kf::LowLevelParticleFilters.AbstractExtendedKalmanFilter{IPD}, x⁻, Σ⁻, u, p=LowLevelParticleFilters.parameters(kf), t::Real=index(kf); R2=LowLevelParticleFilters.get_mat(kf.measurement_model.R2, x⁻, u, p, t)) where IPD
    (; Cjac, measurement) = kf.measurement_model
    ny = kf.kf.ny
    if false ### False for now, IPD not working well here
        μ = zeros(ny)
        measurement(μ, x⁻, u, p, t)
    else
        μ = measurement(x⁻, u, p, t)
    end

    C = Cjac(x⁻, u, p, t)
    S = LowLevelParticleFilters.symmetrize(C * Σ⁻ * C') + R2
    (μ, S)
end


function model_predict_2(
    kf_copy,
    u,
)  

    x⁻, Σ⁻ = predict_kf(kf_copy, u)
    kf_copy.x = x⁻
    kf_copy.R = Σ⁻
    μ, S = measurement_kf(kf_copy, x⁻, Σ⁻, u)
    σ = sqrt.(S)

    (; μ = μ[1] , σ = σ[1])
end

function merge(a::ComponentVector, b::ComponentVector)
    ComponentVector(Base.merge(NamedTuple(a), NamedTuple(b)))
end
function merge(a::ComponentVector, b::NamedTuple) 
    ComponentVector(Base.merge(NamedTuple(a), b))
end
function merge(a::NamedTuple, b::ComponentVector) 
    ComponentVector(Base.merge(a, NamedTuple(b)))
end


