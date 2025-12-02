

@component function ECM(; name)
    @named oneport = OnePort()

    params = @parameters begin
        Q
        R0
        R1
        τ1
        focv(::Real)::Real
    end

    vars = @variables begin
        v(t)
        i(t)
        v1(t) = 0
        vr(t)
        ocv(t)
        soc(t)
    end

    eqs = [
        ocv ~ focv(soc)
        D(soc) ~ i / (Q * 3600.0)
        D(v1) ~ -v1 / τ1 + i * (R1 / τ1)
        vr ~ i * R0
        v ~ ocv + vr + v1
    ]

    extend(System(eqs, t, vars, params; name), oneport)
end

@component function BatteryModule(s; name)

    # TODO: system derating

    params = @parameters begin
        fi(::Real)::Real
    end

    systems = @named begin
        cell[1:s] = ECM()
        current = Current()
        voltage = VoltageSensor()
        ground = Ground()
    end

    eqs = [
        connect(current.p, voltage.p, cell[1].p)
        [connect(cell[i].n, cell[i+1].p) for i in 1:s-1]
        connect(current.n, voltage.n, cell[end].n, ground.g)
        current.i ~ fi(t)
    ]

    System(eqs, t; systems, name)
end

function get_var(sys, var1::Symbol, var2::Symbol)
    getproperty(getproperty(sys, var1), var2)
end

function get_var(sys, var1, var2)
    get_var(sys, Symbol(var1), Symbol(var2))
end

function simulate_module(params, tspan; Ts=1.0)
    # create model
    n = length(params)
    @mtkcompile pack = BatteryModule(n)

    # initial condition
    u0 = Pair[
        pack.fi=>fi
    ]
    for (cell, p) in params
        for (var, val) in p
            push!(u0, get_var(pack, cell, var) => val)
        end
    end

    # simulate
    prob = ODEProblem(pack, u0, tspan)
    sol = solve(prob, Vern7(); saveat=Ts, reltol=1e-10)

    return DataFrame(
        "t" => sol.t,
        "v" => sol[pack.voltage.v],
        "i" => -sol[pack.current.i],
        "q" => -cumsum(sol[pack.current.i]) * Ts / 3600,
        ["v_cell_$i" => sol[get_var(pack, "cell_$i", "v")] for i in 1:n]...,
        ["soc_cell_$i" => sol[get_var(pack, "cell_$i", "soc")] for i in 1:n]...,
    )
end

function select_cell_dataset(df, i)
    v_cell = Symbol("v_cell_$i")
    s_cell = Symbol("soc_cell_$i")
    select(df, :t, :i, :q, v_cell => :v, s_cell => :s)
end

function fit_zscore(df)
    v = StatsBase.fit(ZScoreTransform, df.v)
    σ = StatsBase.fit(ZScoreTransform, df.v, center=false)
    i = StatsBase.fit(ZScoreTransform, df.i, center=false)
    q = StatsBase.fit(ZScoreTransform, df.q)
    r = ZScoreTransform(1, 1, [0.0], [σ.scale[1] / i.scale[1]])
    return (; v, σ, i, q, r)
end

function normalize_data(zt, df)
    v = StatsBase.transform(zt.v, df.v)
    i = StatsBase.transform(zt.i, df.i)
    q = StatsBase.transform(zt.q, df.q)
    return DataFrame(; df.t, v, i, q)
end
