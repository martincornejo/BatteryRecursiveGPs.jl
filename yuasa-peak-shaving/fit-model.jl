function estimate_initial_charge(model, v_init_normalized)
    ismissing(v_init_normalized) && error("no voltage observation available for initial charge estimation")
    kf = model.kf
    zt = kf.p.zt
    q̂min, q̂max = extrema(kf.p.ocv.b0)
    q̂ = collect(q̂min:0.01:q̂max)
    ocv = predict_gp(kf, q̂, :ocv)
    v_pred = StatsBase.reconstruct(zt.v, ocv.μ)
    v_init = only(StatsBase.reconstruct(zt.v, [v_init_normalized]))
    order = sortperm(v_pred)
    q_from_v = LinearInterpolation(q̂[order], v_pred[order]; extrapolation = ExtrapolationType.Constant)
    return q_from_v(v_init)
end

function init_refine(model_orig, y; σ0_q = 1.0e-4)
    model = deepcopy(model_orig)
    idx_first = findfirst(yᵢ -> !any(yᵢ .=== missing), y)
    v_init_normalized = first(y[idx_first])
    q_init = estimate_initial_charge(model_orig, v_init_normalized)

    reinit_kf!(model; x = model_orig.kf.x, R = model_orig.kf.R)

    # Set estimated q AFTER reinit (reinit always resets cc.q = 0)
    xid = model.kf.p.xid
    x_ca = ComponentVector(model.kf.x, xid)
    x_ca.cc.q = q_init
    model.kf.x .= x_ca

    # Give the filter enough initial uncertainty to correct q
    Σid = model.kf.p.Σid
    Σ_ca = ComponentMatrix(copy(model.kf.R), Σid)
    Σ_ca[:cc, :cc] .= σ0_q^2
    model.kf.R .= Σ_ca

    return model
end

function init_open_loop(model, y)
    model_ol = deepcopy(model)
    idx_first = findfirst(yᵢ -> !any(yᵢ .=== missing), y)
    v_init_normalized = first(y[idx_first])
    q_init = estimate_initial_charge(model, v_init_normalized)

    reinit_kf!(model_ol)

    xid = model_ol.kf.p.xid
    x_ca = ComponentVector(model_ol.kf.x, xid)
    x_ca.cc.q = q_init
    model_ol.kf.x .= x_ca

    return model_ol
end


function extract_battery_params(models, sols, cells, abs_fit, ids)
    params = Dict{String, Any}()
    for (i, id) in enumerate(ids)
        kf = models[id].kf
        zt = kf.p.zt
        xs = ComponentVector(sols[id].x_end, kf.p.xid)

        r0 = StatsBase.reconstruct(zt.r, [abs(xs.r0.r)]) |> first
        rcs = if hasproperty(xs, :rc)
            [Dict(
                "R" => StatsBase.reconstruct(zt.r, [abs(xs.rc.r)]) |> first,
                "tau" => abs(xs.rc.τ),
            )]
        elseif hasproperty(xs, :rc1) && hasproperty(xs, :rc2)
            [
                Dict(
                    "R" => StatsBase.reconstruct(zt.r, [abs(xs.rc1.r)]) |> first,
                    "tau" => abs(xs.rc1.τ),
                ),
                Dict(
                    "R" => StatsBase.reconstruct(zt.r, [abs(xs.rc2.r)]) |> first,
                    "tau" => abs(xs.rc2.τ),
                ),
            ]
        else
            error("unsupported model: state has neither rc nor rc1/rc2")
        end
        k = abs(xs.arr.k)

        Q_i = Measurements.value(abs_fit.Q_cell[i])
        s0_i = Measurements.value(abs_fit.s0[i])

        ocv_soc = cells[i].q ./ Q_i .+ s0_i
        ocv_v = cells[i].μ
        order = sortperm(ocv_soc)

        bat_key = "battery_$(id.m)"
        cell_key = "cell_$(id.c)"
        haskey(params, bat_key) || (params[bat_key] = Dict{String, Any}())
        params[bat_key][cell_key] = Dict(
            "Q" => Q_i,
            "soc" => s0_i,
            "rc" => rcs,
            "R0" => r0,
            "k" => k,
            "ocv_soc" => ocv_soc[order],
            "ocv_v" => ocv_v[order],
        )
    end
    return params
end

function extract_initial_states(smoothed, abs_fit, ids)
    states = Dict{String, Any}()
    for (i, id) in enumerate(ids)
        sm = smoothed[id]
        zt = sm.model.kf.p.kf.p.zt
        x0 = sm.smoothed.xT[1]
        R0 = sm.smoothed.RT[1]

        Q_i = Measurements.value(abs_fit.Q_cell[i])
        s0_i = Measurements.value(abs_fit.s0[i])

        q0 = StatsBase.reconstruct(zt.q, [first(x0)]) |> first
        q0_std = StatsBase.reconstruct(zt.q, [sqrt(R0[1, 1])]) |> first

        soc0 = q0 / Q_i + s0_i
        soc0_std = q0_std / Q_i

        bat_key = "battery_$(id.m)"
        cell_key = "cell_$(id.c)"
        haskey(states, bat_key) || (states[bat_key] = Dict{String, Any}())
        cell = Dict{String, Any}(
            "soc0" => soc0,
            "soc0_std" => soc0_std,
        )

        inner_kf = sm.model.kf.p.kf
        xs = ComponentVector(inner_kf.x, inner_kf.p.xid)
        if hasproperty(xs, :rc)
            cell["v_rc"] = StatsBase.reconstruct(zt.σ, [x0[2]]) |> first
            cell["v_rc_std"] = StatsBase.reconstruct(zt.σ, [sqrt(R0[2, 2])]) |> first
        elseif hasproperty(xs, :rc1) && hasproperty(xs, :rc2)
            for j in 1:2
                cell["v_rc$j"] = StatsBase.reconstruct(zt.σ, [x0[1 + j]]) |> first
                cell["v_rc$(j)_std"] = StatsBase.reconstruct(zt.σ, [sqrt(R0[1 + j, 1 + j])]) |> first
            end
        else
            error("unsupported model: inner KF has neither rc nor rc1/rc2")
        end

        states[bat_key][cell_key] = cell
    end
    return states
end
