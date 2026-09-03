# BatteryRecursiveGPs.jl

*Online battery state and parameter estimation with equivalent-circuit models and recursive Gaussian processes.*

Model-based battery state estimation tracks state of charge (SOC) and state of health (SOH) by Kalman-filtering an equivalent circuit model (ECM). As the battery degrades, the model's parameters drift and must be adapted throughout its lifetime. Even the shape of the open-circuit voltage (OCV) curve changes with aging, so a lab characterization from the beginning of life is no longer representative. BatteryRecursiveGPs jointly estimates the battery states *and* the ECM parameters online, directly from field operation data and without prior characterization. This is enabled by recursive Gaussian process regression: functional parameters, such as the OCV curve or the SOC-dependent resistance, are reconstructed within the filter and refined with every new measurement, uncertainty quantification included.

The general-purpose recursive Gaussian process regression is provided by [RecursiveGPs.jl](https://github.com/martincornejo/RecursiveGPs.jl), this package applies it to battery ECMs.

This repository serves two purposes:

- **A Julia package** for building ECM battery models with recursive GP components and running them through extended Kalman filters and smoothers. The package is currently shaped around the models and dataset of the paper, a more general API may follow.
- **Companion code** for the paper *Estimating the Health and State of Charge of Each Cell in a Second-Life Battery System from Field Data*. The `yuasa/` directory contains the data and scripts that reproduce its results.

<!-- ECM learning animation (yuasa/src/plot/ecm_animation.jl) -->
https://github.com/user-attachments/assets/0f384f7c-6973-4115-8f35-fefba1fea5a1


## Installation

Requires Julia ≥ 1.12. The package is not registered. To use it as a dependency in another project, add RecursiveGPs.jl first (Pkg only applies a package's `[sources]` when that package is the active project, so it cannot resolve the unregistered dependency on its own):

```julia
pkg> add https://github.com/martincornejo/RecursiveGPs.jl
pkg> add https://github.com/martincornejo/BatteryRecursiveGPs.jl
```

To work on the repository itself, clone it and instantiate the project. Here the pinned [RecursiveGPs.jl](https://github.com/martincornejo/RecursiveGPs.jl) dependency is resolved automatically from its URL:

```julia
pkg> activate .
pkg> instantiate
```

## Case study: second-life battery system

The `yuasa/` directory is a workspace project with the full analysis of the paper: the field dataset of a second-life battery system (27 modules, 324 cells) and the scripts that generate all results. It doubles as the usage example of the package. From the repository root, activate the `yuasa` environment:

```julia
pkg> activate yuasa
```

The scripts can be run start to finish, but they are also meant to be explored interactively (the VS Code Julia extension is recommended, evaluating them line by line).

The scripts, in order:

1. `yuasa/scripts/reference.jl`: builds the measured OCV reference from the low-power rig measurement, upstream of everything else.
2. `yuasa/scripts/hyperparams.jl`: distributed hyperparameter selection for all cell and lumped-module models.
3. `yuasa/scripts/main.jl`: the main analysis, fits every cell and module, derives the SOH and SOC results, and produces the paper figures.
4. `yuasa/scripts/validation.jl`: validates the reconstructed OCV curves against the reference measurement.

`yuasa/scripts/animation.jl` renders the learning animation shown above.

## License

[MIT](LICENSE)
