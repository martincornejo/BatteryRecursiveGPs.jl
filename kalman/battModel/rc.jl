function RC(ts, τ0, R0; Vrc0=0.0, Vrc_σ=1e-6, σ1=[1e-2, 2e-6, 3e-11], σ2=sqrt(1e-3), τh=60, Rh=15e-3)
    """
    Main function and only functions user need to know
    """


    R1 = Diagonal(σ1 .^ 2)
    R2(x, u, p, t) = Diagonal(fill(σ2^2, 1))
    nx = 1
    ny = 1

    x0 = ComponentVector(
        Vrc=Vrc0,
        τ=τ0,
        R=R0
    )

    Σ0 = false .* x0 * x0'
    Σ0[:Vrc, :Vrc] = Vrc_σ^2
    Σ0[:τ, :τ] = (τh - τ0)^2
    Σ0[:R, :R] = (Rh - R0)^2
    Σ0 = Σ0 + 1e-6 * I

    d0 = MvNormal(x0, Σ0)


    xid = getaxes(x0)
    Σid = getaxes(Σ0)
    p = generate_p_rc(ts, xid, Σid)

    rc = (;
        dynamics=dynamics_rc,
        measurement=measurement_rc,
        R1=R1,
        R2=R2,
        d0=d0,
        nx=nx,
        ny=ny,
        p=p
    )

    return rc
end

function generate_p_rc(ts, xid, Σid, i=[0.0])
    p = ComponentVector(;
        ts,
        xid,
        Σid,
        i
    )
    return p
end


function dynamics_rc(x, u, p, t)
    (; ts, xid) = p
    c = ComponentVector(x, xid)
    i = u.i[1]

    c.Vrc = exp(-ts / c.τ) * c.Vrc + i * c.R * (1 - exp(-ts / c.τ))
    x .= c
    return x
end

function measurement_rc(x, u, p, t)
    (; xid) = p
    c = ComponentVector(x, xid)
    return [c.Vrc]
end