function BattModel(components)
    """
    Generates the Kalman filter model for the battery
    components_batt: a tuple with the components of the battery
    """


    component_names = keys(components)
    x0 = ComponentVector(; (name => mean(components[name].d0) for name in component_names)...)

    Σ0 = false .* x0 * x0'
    R1 = false .* x0 * x0'
    for name in component_names
        component = components[name]
        Σ0[name, name] = cov(component.d0)
        R1[name, name] = component.R1
    end

    d0 = MvNormal(x0, Σ0)
    xid = getaxes(x0)
    Σid = getaxes(Σ0)

    p = generate_p_batt(components, xid, Σid)


    battModel = (;
        dynamics=dynamics_batt,
        measurement=measurement_batt,
        R1=R1,
        R2=R2_batt_fun,
        d0=d0,
        nx=length(x0),
        ny=1,
        p=p
    )

    return battModel
end



function generate_p_batt(components, xid, Σid)
    """
    Can be done automatic and easier
    """
    p = (;
        xid=xid,  # axes for state vector
        Σid=Σid,
        components=components,  # components of the battery
    )

    return p

end


function R2_batt_fun(x, u, p, t)
    (; xid, components) = p
    c = ComponentVector(x, xid)
    R2 = zeros(1)

    for name in keys(components)
        component = getfield(components, name)
        x_component = view(c, name)
        R2 .+= component.R2(x_component, u, component.p, t)
    end

    return R2
end



function dynamics_batt(x, u, p, t)
    """
    Calling measurement on all components
    This functions should be automatic
    """
    (; xid, components) = p
    c = ComponentVector(x, xid)

    for name in keys(components)
        component = getfield(components, name)
        x_component = view(c, name)
        x_component .= component.dynamics(x_component, u, component.p, t)
    end

    return c
end



function measurement_batt(x, u, p, t)
    """
    This function should be generated automatically given a model_function
    """
    (; xid, components) = p
    c = ComponentVector(x, xid)
    v = zeros(1)

    for name in keys(components)
        component = getfield(components, name)
        x_component = view(c, name)

        v += component.measurement(x_component, u, component.p, t)
    end
    return v
end