# TOML-name registries. Every component selectable by name from a TOML file is registered
# *at its definition site* via the `register_*!` helpers below: gain/leakage
# parameterizations (instrument/parameterizations.jl), polarization representations
# (sky/polreps.jl), and mean-image models (sky/meanmodels.jl). The config parsers
# (config/sky_config.jl, instrument/assemble.jl) only ever look names up in these dicts, so
# the string a user writes in the TOML always sits directly next to the code it selects.

const GAIN_SCHEMES = Dict{String, NamedTuple}()
const LEAKAGE_SCHEMES = Dict{String, NamedTuple}()
const POLREPS = Dict{String, Any}()
const MEAN_MODELS = Dict{String, Function}()

const SEGMENTATIONS = Dict{String, Any}(
    "integ" => IntegSeg(),
    "scan" => ScanSeg(),
    "track" => TrackSeg(),
)

_allowed(reg::AbstractDict) = sort!(collect(keys(reg)))

function _register!(reg::AbstractDict, name::String, what::String, value)
    haskey(reg, name) &&
        error("duplicate $what TOML name '$name' — already registered as $(reg[name])")
    reg[name] = value
    return value
end

"""
    register_gain_scheme!(name, parameterization; kind, params)

Register a gain parameterization under the TOML name `name` (selected by
`[gain] scheme = "<name>"` in the instrument TOML). `params` must list **exactly** the
fields the parameterization destructures from its argument — they become the required
`[priors.<param>]` entries and the keys of the instrument prior NamedTuple, so a mismatch
is a silent (or `KeyError`) bug. `kind` is `:jones` (R/L feed pair) or `:single` (single
total-intensity gain).
"""
function register_gain_scheme!(name::String, parameterization; kind::Symbol, params::Tuple)
    kind in (:jones, :single) ||
        error("gain scheme '$name': kind must be :jones or :single, got :$kind")
    return _register!(GAIN_SCHEMES, name, "gain scheme", (; parameterization, kind, params))
end

"""
    register_leakage_scheme!(name, parameterization; params)

Register a leakage parameterization under the TOML name `name` (selected by
`[leakage] scheme = "<name>"` in the instrument TOML). `params` has the same
must-match-the-destructured-fields contract as [`register_gain_scheme!`](@ref).
"""
function register_leakage_scheme!(name::String, parameterization; params::Tuple)
    return _register!(LEAKAGE_SCHEMES, name, "leakage scheme", (; parameterization, params))
end

"""
    register_polrep!(name, polrep)

Register a polarization-representation instance under the TOML name `name` (selected by
`model.polrep = "<name>"` in the image TOML).
"""
register_polrep!(name::String, polrep) = _register!(POLREPS, name, "polrep", polrep)

"""
    register_mean_model!(builder, name)

Register a mean-image model under the TOML name `name` (selected by `mean.type = "<name>"`
in the image TOML). `builder(meancfg, grid, beam)` receives the parsed `[mean]` table, the
image grid, and the observation beam (radians, or `nothing` when building standalone) and
returns the mean-model instance. Do-block friendly.
"""
register_mean_model!(builder, name::String) =
    _register!(MEAN_MODELS, name, "mean model", builder)

"""
    toml_name(x) -> String

The TOML name under which `x` (a gain/leakage parameterization or a `PolRep` instance) was
registered. Reverse lookup over the registries; errors if `x` was never registered.
"""
function toml_name(x)
    for reg in (GAIN_SCHEMES, LEAKAGE_SCHEMES)
        for (name, sch) in reg
            sch.parameterization === x && return name
        end
    end
    for (name, p) in POLREPS
        p === x && return name
    end
    return error("no TOML name registered for $x")
end
