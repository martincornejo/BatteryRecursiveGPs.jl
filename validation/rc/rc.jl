function RC(; ts, τ0, R0, Vrc0=0.0, Vrc_σ=1.0e-6, τ_σ=(τ0 - 60.0), R_σ=(R0 - 15e-3), σ1=[1e-2, 2e-6, 3e-11], σ2=sqrt(1e-3))
    """
    Main function and only functions user need to know
    """
    x0 = ComponentVector(
        Vrc=Vrc0,
        τ=τ0,
        R=R0
    )


    R1 = Diagonal(σ1 .^ 2)
    nx = length(x0)
    ny = 1
    Σ0 = zeros(eltype(τ_σ), nx, nx)

    Σ0[1, 1] = Vrc_σ^2
    Σ0[2, 2] = τ_σ^2
    Σ0[3, 3] = R_σ^2

    Σ0_ = false .* x0 * x0'
    d0 = MvNormal(x0, Σ0)


    xid = getaxes(x0)
    Σid = getaxes(Σ0_)
    p = generate_p_rc(ts, xid, Σid, σ1, σ2)

    rc = (;
        dynamics=dynamics_rc,
        dynamics_ip=dynamics_rc_ip,
        measurement=measurement_rc,
        R1=R1,
        R2=R2fun_rc,
        μ0=x0,
        Σ0 = Σ0,
        nx=nx,
        ny=ny,
        p=p
    )

    return rc
end

function generate_p_rc(ts, xid, Σid, σ1, σ2, i=[0.0])
    p = ComponentVector(;
        ts,
        xid,
        Σid,
        σ1,
        σ2,
        i
    )
    return p
end

function R2fun_rc(x, u, p, t)
    (; σ2) = p.p
    return σ2^2
end


function dynamics_rc(x, u, p, t)
    (; ts, xid) = p.p
    c = ComponentVector(x, xid)
    i = u.i[1]

    c.Vrc = exp(-ts / c.τ) * c.Vrc + i * c.R * (1 - exp(-ts / c.τ))
    return c
end


function dynamics_rc_ip(dx, x, u, p, t)
    (; ts, xid) = p.p
    dx .= ComponentVector(dx, xid)
    i = u.i[1]
    dx.Vrc = exp(-ts / c.τ) * dx.Vrc + i * dx.R * (1 - exp(-ts / dx.τ))

end

function measurement_rc(x, u, p, t)
    (; xid) = p.p
    c = ComponentVector(x, xid)
    return c.Vrc
end