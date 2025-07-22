
function R0_OCV(ocv, r0, soc)
    """
    Main function and only functions user need to know
    """

    x0 = ComponentVector(;
        c_ocv=mean(ocv.d0),
        c_r0=mean(r0.d0),
        c_soc=mean(soc.d0)
    )

    Σ0 = false .* x0 * x0'

    Σ0[:c_ocv, :c_ocv] = cov(ocv.d0)
    Σ0[:c_r0, :c_r0] = cov(ocv.d0)
    Σ0[:c_soc, :c_soc] = cov(soc.d0)

    xid = getaxes(x0)
    Σid = getaxes(Σ0)
    d0 = MvNormal(x0, Σ0)

    p = generate_p_r0_ocv(ocv, r0, soc, xid, Σid)

    n_ocv = size(x0.c_ocv, 1)
    n_r0 = size(x0.c_r0, 1)
    n_soc = size(x0.c_soc, 1)

    R1 = vcat(
        hcat(ocv.R1, zeros(n_ocv, n_r0 + n_soc)),
        hcat(zeros(n_r0, n_ocv), r0.R1, zeros(n_r0, n_soc)),
        hcat(zeros(n_soc, n_ocv + n_r0), soc.R1)
    )



    nx = length(x0)
    ny = 1

    r0_ocv = (;
        dynamics=dynamics_r0_ocv,
        measurement=measurement_r0_ocv,
        R1=R1,
        R2=R2fun_r0_ocv,
        d0=d0,
        nx=nx,
        ny=ny,
        p=p
    )
    return r0_ocv
end

function generate_p_r0_ocv(ocv, r0, soc, xid, Σid)
    cache = (;
        u_=(;
            b=0,
            i=0)
    )
    p = (
        ocv=ocv,
        soc=soc,
        r0=r0,
        xid=xid,
        Σid=Σid,
        cache=cache
    )
    return p
end


function R2fun_r0_ocv(x, u, p, t)
    (; xid, ocv, r0, cache) = p
    c = ComponentVector(x, xid)
    #c. in all
    @unpack c_ocv, c_r0, c_soc = c
    u_ = (;
        b=c_soc,
        i=u.i
    )
    R2ocv = ocv.R2(c_ocv, u_, ocv.p, t)
    R2r0 = r0.R2(c_r0, u_, r0.p, t)
    R2 = R2ocv + R2r0

    return R2
end

function dynamics_r0_ocv(x, u, p, t)
    (; xid, ocv, r0, soc, cache) = p
    c = ComponentVector(x, xid)
    #c. in all
    @unpack c_ocv, c_r0, c_soc = c
    u_ = (;
        b=c_soc,
        i=u.i
    )

    c_ocv .= ocv.dynamics(c_ocv, u_, ocv.p, t)
    c_r0 .= r0.dynamics(c_r0, u_, r0.p, t)
    c_soc .= soc.dynamics(c_soc, u_, soc.p, t)

    return c
end

function measurement_r0_ocv(x, u, p, t)
    (; xid, ocv, r0, cache) = p
    c = ComponentVector(x, xid)
    @unpack c_ocv, c_r0, c_soc = c
    u_ = (;
        b=c_soc,
        i=u.i
    )

    v_ocv = ocv.measurement(c_ocv, u_, ocv.p, t)
    v_r0 = r0.measurement(c_r0, u_, r0.p, t)
    v = v_ocv + v_r0

    return v

end

function SOC(; Q=4.8 * 3600, soc0=0.5, σ1=0.01, Σ_soc=0.1)
    """
    Main function and only functoins user need to know
    """
    R1 = Diagonal(fill(σ1, 1))
    R2(x, u, p, t) = nothing
    nx = length(b0)
    ny = 1
    x0 = [soc0]
    Σ0 = [Σ_soc,]

    p = generate_p_soc(Q)

    d0 = MvNormal(x0, Σ0)

    nx = length(x0)
    ny = 1
    soc = (;
        dynamics=dynamics_soc,
        measurement=measurement_soc,
        R1=R1,
        R2=R2,
        d0=d0,
        nx=nx,
        ny=ny,
        p=p
    )

    return soc
end

function generate_p_soc(Q)

    p = (;
        Q=Q,
        ts=1.0
    )
    return p
end

function dynamics_soc(x, u, p, t)
    (; Q, ts) = p
    return x .+ u.i[1] / Q * ts # identity
end

function measurement_soc(x, u, p, t)
    return x
end



function Q(; q0, σ1=0.01, Σ_q=0.1, ts=1)
    R1 = Diagonal(fill(σ1, 1))
    R2(x, u, p, t) = nothing

    x0 = [q0]
    Σ0 = [Σ_q]

    nx = length(x0)
    ny = 1

    p = generate_p_q(ts)

    d0 = MvNormal(x0, Σ0)

    nx = length(x0)
    ny = 1
    q = (;
        dynamics=dynamics_q,
        measurement=measurement_q,
        R1=R1,
        R2=R2,
        d0=d0,
        nx=nx,
        ny=ny,
        p=p
    )

    return q
end

function generate_p_q(ts)

    p = (; ts=ts
    )
    return p
end

function dynamics_q(x, u, p, t)
    (; ts) = p
    return x .+ u.i[1] * ts # identity
end

function measurement_q(x, u, p, t)
    return x
end

function OCV(gp, b0; σ2=1e-5, tr=ZScoreTransform(1, 1, [0.0], [1.0]), tr_b=ZScoreTransform(1, 1, [0.0], [1.0]))
    """
    Main function and only functions user need to know
    """
    R1 = Diagonal(zero(b0))
    nx = length(b0)
    ny = 1

    x0 = mean(gp, b0)
    Σ0 = cov(gp, b0) + 1e-6I
    d0 = MvNormal(x0, Σ0)

    p = generate_p_ocv(gp, d0, b0, σ2, tr, tr_b)


    ocv = (;
        dynamics=dynamics_ocv,
        measurement=measurement_ocv,
        R1=R1,
        R2=R2fun_ocv,
        d0=d0,
        nx=nx,
        ny=ny,
        p=p
    )

    return ocv
end

function generate_p_ocv(gp, d0, b0, σ2, tr, tr_b)
    return generate_p_gp(gp, d0, b0, σ2, tr, tr_b)
end


function R2fun_ocv(x, u, p, t)
    return R2fun_gp(x, u, p, t)
end

function dynamics_ocv(x, u, p, t)
    return dynamics_gp(x, u, p, t)
end

function measurement_ocv(x, u, p, t)
    return measurement_gp(x, u, p, t)
end



function R0(gp, b0; σ2=1e-5, tr=ZScoreTransform(1, 1, [0.0], [1.0]), tr_b=ZScoreTransform(1, 1, [0.0], [1.0]))
    """
    Main function and only functions user need to know
    """
    R1 = Diagonal(zero(b0))
    nx = length(b0)
    ny = 1



    x0 = mean(gp, b0)
    Σ0 = cov(gp, b0) + 1e-6I

    d0 = MvNormal(x0, Σ0)
    p = generate_p_r0(gp, d0, b0, σ2, tr, tr_b)


    r0 = (;
        dynamics=dynamics_r0,
        measurement=measurement_r0,
        R1=R1,
        R2=R2fun_r0,
        d0=d0,
        nx=nx,
        ny=ny,
        p=p
    )

    return r0
end

function generate_p_r0(gp, d0, b0, σ2, tr, tr_b)
    return generate_p_gp(gp, d0, b0, σ2, tr, tr_b)
end

function R2fun_r0(x, u, p, t)
    return u.i .^ 2 .* R2fun_gp(x, u, p, t)
end

function dynamics_r0(x, u, p, t)
    return dynamics_gp(x, u, p, t)
end

function measurement_r0(x, u, p, t)
    return u.i .* measurement_gp(x, u, p, t)
end
