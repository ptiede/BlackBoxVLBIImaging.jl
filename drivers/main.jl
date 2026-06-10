using Pkg
Pkg.activate(@__DIR__)

using Comonicon
using BlackBoxVLBIImaging

"""
Image a VLBI observation, configured entirely from four TOML files.

The sampler (AdvancedHMC vs Reactant NUTS) is part of the fitting TOML, since selecting
Reactant also changes the optimizer path.

# Options

- `--image <path>`: image/sky-model TOML (grid, mean model, flux, random-field order).
- `--instrument <path>`: instrument-model TOML (gain/leakage scheme + per-parameter priors).
- `--data <path>`: data TOML (file/array/averaging/noise + flag table).
- `--fitting <path>`: fitting-strategy TOML (optimizer, tempering schedule, sampler, Reactant).
- `--outpath <path>`: output base path for the run (default `Runs/run`).

# Flags

- `--restart`: resume the run from a previously serialized optimum at `--outpath` instead of
  re-optimizing. This is a one-off action, deliberately not stored in any TOML.
"""
Comonicon.@main function main(;
        image::String, instrument::String, data::String, fitting::String,
        outpath::String = "Runs/run", restart::Bool = false
    )
    return image_from_toml(image, instrument, data, fitting; outpath, restart)
end
