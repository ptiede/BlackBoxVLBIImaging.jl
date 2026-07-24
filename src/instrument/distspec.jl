# Closed-allowlist parser turning a TOML distribution spec into a distribution. There is
# deliberately NO `@eval` of config strings — only the names below are constructible.
#
# The familiar names ("Normal", "Uniform", ...) map to the `VLBI*` distributions from
# VLBIImagePriors rather than plain `Distributions` types: the `VLBI*` variants are
# Reactant-traceable (so the instrument priors work on the Reactant path) *and* work on the
# CPU/Enzyme path, whereas plain `Distributions` aggregate into e.g. `DiagNormal`, which has
# no Reactant `logpdf`. Add a distribution by extending this allowlist.
#
# A spec is an (inline) table: `{ dist = "Normal", args = [0.0, 0.4] }`. An optional
# `lower`/`upper` wraps the result in `VLBITruncated`.
#
# Special case: `DiagonalVonMises` takes `args = [mean, width]` where `width` is an angular
# std-dev-like scale, converted to a concentration via `κ = inv(width^2)`.

const _DIST_ALLOWLIST = Dict{String, Function}(
    "Normal" => (a...) -> VLBIImagePriors.VLBIGaussian(a...),
    "Gaussian" => (a...) -> VLBIImagePriors.VLBIGaussian(a...),
    "VLBIGaussian" => (a...) -> VLBIImagePriors.VLBIGaussian(a...),
    "Uniform" => (a...) -> VLBIImagePriors.VLBIUniform(a...),
    "Exponential" => (a...) -> VLBIImagePriors.VLBIExponential(a...),
    "TDist" => (a...) -> VLBIImagePriors.VLBITDist(a...),
    "Beta" => (a...) -> VLBIImagePriors.VLBIBeta(a...),
    "InverseGamma" => (a...) -> VLBIImagePriors.VLBIInverseGamma(a...),
    "DiagonalVonMises" => (a...) -> DiagonalVonMises(a[1], inv(a[2]^2)),
)

"""
    parse_dist(spec::AbstractDict) -> Distribution

Parse a TOML distribution spec into a (Reactant-friendly `VLBI*`) distribution, restricted
to the closed allowlist in `_DIST_ALLOWLIST`. Throws on an unknown distribution name.
"""
function parse_dist(spec::AbstractDict)
    check_config_keys(spec, ("dist", "args", "lower", "upper"), "a distribution spec")
    haskey(spec, "dist") || error("distribution spec is missing the 'dist' key: $spec")
    name = String(spec["dist"])
    haskey(_DIST_ALLOWLIST, name) ||
        error("unknown distribution '$name'. Allowed: $(sort(collect(keys(_DIST_ALLOWLIST))))")
    args = Float64.(get(spec, "args", Float64[]))
    d = _DIST_ALLOWLIST[name](args...)
    lower = get(spec, "lower", nothing)
    upper = get(spec, "upper", nothing)
    if !isnothing(lower) && !isnothing(upper)
        d = VLBIImagePriors.VLBITruncated(d; lower = lower, upper = upper)
    elseif !isnothing(lower)
        d = VLBIImagePriors.VLBITruncated(d; lower = lower)
    elseif !isnothing(upper)
        d = VLBIImagePriors.VLBITruncated(d; upper = upper)
    end
    return d
end

# --- Gauss-Markov process specs ----------------------------------------------------------
# A `GaussMarkovSitePrior` (temporally correlated instrument prior) is parameterized by a
# continuous-time Gauss-Markov process instead of a single per-time distribution. The
# process spec is an (inline) table `{ kind = "OrnsteinUhlenbeck", sigma = ..., tau = ...,
# mu = 0.0 }`. Each hyperparameter (`sigma`, `tau`) is either a fixed `Real` or a *fitted*
# hyperparameter given by a distribution spec (parsed with `parse_dist`, so the same
# Reactant-friendly `VLBI*` variants are used — they are `<: Distributions.Distribution`,
# which is how `hyperprior` recognizes a field as fitted). Like `parse_dist`, this is a
# closed allowlist: only `OrnsteinUhlenbeck` is constructible.

# A hyperparameter is a fixed number or a fitted distribution (a distribution spec table).
_parse_hyper(x::Real) = Float64(x)
_parse_hyper(x::AbstractDict) = parse_dist(x)
_parse_hyper(x) = error(
    "process hyperparameter must be a number or a distribution spec table, got: $(repr(x))"
)

"""
    parse_process(spec::AbstractDict) -> AbstractGaussMarkovProcess

Parse a TOML process spec into a Gauss-Markov process for a [`GaussMarkovSitePrior`],
restricted to the closed allowlist (currently only `OrnsteinUhlenbeck`). `sigma`/`tau` are
each a number (fixed) or a distribution spec (fitted hyperparameter); `mu` is an optional
fixed number (default `0.0`, not fittable). Throws on an unknown process name or a missing
`sigma`/`tau`.
"""
function parse_process(spec::AbstractDict)
    check_config_keys(spec, ("kind", "sigma", "tau", "mu"), "a process spec")
    haskey(spec, "kind") || error("process spec is missing the 'kind' key: $spec")
    name = String(spec["kind"])
    name in ("OrnsteinUhlenbeck", "ou") ||
        error("unknown process '$name'. Allowed: OrnsteinUhlenbeck")
    haskey(spec, "sigma") || error("process spec is missing 'sigma': $spec")
    haskey(spec, "tau") || error("process spec is missing 'tau': $spec")
    σ = _parse_hyper(spec["sigma"])
    τ = _parse_hyper(spec["tau"])
    μraw = get(spec, "mu", 0.0)
    μraw isa Real || error("process 'mu' must be a fixed number (it is not fittable), got: $(repr(μraw))")
    return OrnsteinUhlenbeck(; σ = σ, τ = τ, μ = Float64(μraw))
end
