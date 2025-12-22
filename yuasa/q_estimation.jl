
# === model
function dynamics_dual!(x⁺, x⁻, u, p, t)
    (; kf_ecm, xid) = p
    kfx = ComponentVector(kf_ecm.x, kf_ecm.p.xid)
    xc⁻ = ComponentVector(x⁻, xid)
    xc⁺ = ComponentVector(x⁺, xid)
    xc⁺ .= xc⁻

    kfx.rc.v = dynamics_rc(kfx.rc, u.i, kf_ecm.p)
    xc⁺.cc.q = dynamics_cc(xc⁻.cc, u.i, p, t)
    nothing # IPD
end

function measurement_dual(x, u, p, t)
    (; kf_ecm, xid) = p
    xc = ComponentVector(x, xid)
    kfx = ComponentVector(kf_ecm.x, kf_ecm.p.xid)

    ocv = measurement_gp(kf_ecm.p.ocv, kfx.ocv, u.q)
    r0 = measurement_gp(kf_ecm.p.r0, kfx.r0, u.q)
    vrc = kfx.rc.v # measurement rc

    ocv + u.i * r0 + vrc|> SVector{1}
end

function R2_dual(x, u, p, t)
    (; vσ²) = p
    vσ² |> SMatrix{1,1}
end

##
function build_kf_dual(kf_ecm, θ, ϑ, zt; n=21)
    θ´ = merge_componentvectors(θ, ϑ)
    Ts = θ´.Ts
    cc = ColoumbCounting(σ1=θ´.q.σ1)
    kf_ecm.x
    # measurement / model noise
    vσ² = StatsBase.transform(zt.σ, [θ´.vσ^2]) |> first

    p = (;kf_ecm=kf_ecm, Ts, vσ², zt)
    comp = (; cc)

    make_ekf(comp, dynamics_dual!, measurement_dual, R2_dual; p)
end


function model_predict_q(kf, u)
    kf = kf.p.kf_ecm
    (; xid, vσ²) = kf.p
    xc = ComponentVector(kf.x, xid)

    vrc = xc.rc.v
    ocv = predict_gp(kf, [u.q], :ocv)
    r0 = predict_gp(kf, [u.q], :r0)
    μ = ocv.μ[1] + u.i * r0.μ[1] + vrc
    σ = sqrt(ocv.σ[1]^2 + u.i^2 * r0.σ[1]^2 + vσ²) # TODO: vσ
    # μ = StatsBase.reconstruct(zt.v, [μ̂]) |> first
    # σ = StatsBase.reconstruct(zt.σ, [σ̂]) |> first
    (; μ, σ)

end

