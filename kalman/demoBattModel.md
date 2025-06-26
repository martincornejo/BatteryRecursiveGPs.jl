# Code demoBattModel summary

## Structure

The code is composed of different componetns structures that we can add to our Battery Model, currently only: RGP and RC.

### Components

We can create different modules for each component (RGP, RC) and the BattModel build from them:

    - RGP
    - RC
    - BattModel

Each module only function that the user can use is generate\_ which returns a named tuple with the parameters needed to do a Kalman Filter.

    generate_{component} -> (
            dynamics(x,u,p,t)::Function,
            measurement(x,u,p,t):.Function,
            R1::MAtrix{Float64},
            R2(x,u,p,t)::Function,
            d0::MvNormal,
            nx::Float64/Int64,
            ny::Float64/Int64,
            p::NamedTuple,
        )

R2 should be a functions for easiness of scalabilty
Dynamics and measurement must be a function due it's nature

### Handling of control parameter u

For the control Parameter, each component is responsible of retrieving the right value for itself.

Ej:

RGP control parameter is b, then RGP modules is responsible that when given a u to retrieve b as b = u.b.

Why?

Easier to scale/tune for future applications, if BattModel takes care of givin the right control parameter to each component we will have to refactor the whole BattModel code so it passes the right parameter, instead easier to make sure before hand that the control parameter u has the right structure before training

    u = ComponentVector(; i, b,...)

## BattModel

BattModel is the main Kalman filter, and on it's parameters stores the named tuples of it's components.

        p = (;
            ocv=components_batt.ocv,
            r0=components_batt.r0,
            rc=components_batt.rc,
            xid=xid
        )

Currently is done hardCoded for easiens to read (not anymore), but it can be easily updated to be completely automatic in function of a Model (Ej.: How is the battery voltage computed) and all the components of the battery

The main goal would be to also add a model which will define with ease all the measurement, dynamic, R1 and R2 based on it's components:

    - BattModel(components, model)

Currently:
Currently BattModel dh

## Improvements

- Make BattModel easily tuneable in functions of the components and a model as input. No need to define by hand measurement, dynamic or R2 computations.
- Make RGP work with normalization (Done)
- Implement RC parameter tuning kalman filter. (Done)
- Make ts on RC a part of the control parameter
- Make R2 on RC a function (Done)
- Make functions to update in-place and return nothing 
- Think is is a good idea so user can components or delete components once a battModel is generated -> Currently not necessary
