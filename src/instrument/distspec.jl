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
# process spec is an (inline) table whose `kind` picks the process and whose remaining keys
# are that process's hyperparameters:
#
#   { kind = "OrnsteinUhlenbeck", sigma = ..., tau = ..., mu = 0.0 }   (real line)
#   { kind = "WrappedBrownian",   tau = ... }                          (circle, unbounded)
#   { kind = "WrappedOrnsteinUhlenbeck", sigma = ..., tau = ..., mu = 0.0 }  (circle, pinned)
#
# Each hyperparameter is either a fixed `Real` or a *fitted* hyperparameter given by a
# distribution spec (parsed with `parse_dist`, so the same Reactant-friendly `VLBI*`
# variants are used — they are `<: Distributions.Distribution`, which is how `hyperprior`
# recognizes a field as fitted). Like `parse_dist`, this is a closed allowlist: only
# `OrnsteinUhlenbeck`, `WrappedBrownian`, `WrappedOrnsteinUhlenbeck`, and
# `VonMisesProcess` are constructible.

# A hyperparameter is a fixed number or a fitted distribution (a distribution spec table).
_parse_hyper(x::Real) = Float64(x)
_parse_hyper(x::AbstractDict) = parse_dist(x)
_parse_hyper(x) = error(
    "process hyperparameter must be a number or a distribution spec table, got: $(repr(x))"
)

"""
    parse_process(spec::AbstractDict) -> AbstractGaussMarkovProcess

Parse a TOML process spec into a Gauss-Markov process for a [`GaussMarkovSitePrior`],
restricted to the closed allowlist:

  - `kind = "OrnsteinUhlenbeck"` (alias `"ou"`): the stationary real-line process. `sigma`
    (marginal std) and `tau` (correlation time in hours) are each a number (fixed) or a
    distribution spec (fitted hyperparameter); `mu` is an optional fixed number (default
    `0.0`, not fittable).
  - `kind = "WrappedBrownian"` (alias `"wb"`): Brownian motion on the circle, the process
    to use for gain *phases* — its wrapped-normal transitions make the prior exactly
    `2π`-periodic, so the `2π`-shifted modes a real-line phase prior produces are all
    equivalent. `tau` is the phase coherence time in **hours** — the gap over which the
    visibility-domain coherence `E[exp(iΔθ)]` falls by `1/e` — in the same units as the
    `OrnsteinUhlenbeck` `tau`. Again either a number or a distribution spec.
  - `kind = "WrappedOrnsteinUhlenbeck"` (alias `"wou"`): the mean-reverting process on
    the circle (shortest-arc drift), for a phase pinned near a level — e.g. a gain-ratio
    phase drifting about a separate offset term. `sigma` is the circular marginal spread
    in radians, `tau` the reversion time in hours, `mu` an optional fixed circular mean.
    Centered coordinates only.
  - `kind = "VonMisesProcess"` (alias `"vm"`): the same mean-reverting circular process
    with the shortest-arc drift replaced by the smooth sine drift, which additionally
    supports the non-centered (whitened) coordinates — Comrade's default for it. Same
    `sigma`/`tau`/`mu` as `WrappedOrnsteinUhlenbeck`, and a drop-in replacement for it in
    its `σ ≪ π` regime of validity.

Throws on an unknown process name or a missing hyperparameter.
"""
function parse_process(spec::AbstractDict)
    haskey(spec, "kind") || error("process spec is missing the 'kind' key: $spec")
    name = String(spec["kind"])
    if name in ("OrnsteinUhlenbeck", "ou")
        check_config_keys(spec, ("kind", "sigma", "tau", "mu"), "an OrnsteinUhlenbeck process spec")
        haskey(spec, "sigma") || error("process spec is missing 'sigma': $spec")
        haskey(spec, "tau") || error("process spec is missing 'tau': $spec")
        σ = _parse_hyper(spec["sigma"])
        τ = _parse_hyper(spec["tau"])
        μraw = get(spec, "mu", 0.0)
        μraw isa Real ||
            error("process 'mu' must be a fixed number (it is not fittable), got: $(repr(μraw))")
        return OrnsteinUhlenbeck(; σ = σ, τ = τ, μ = Float64(μraw))
    elseif name in ("WrappedBrownian", "wb")
        # `D` was the old (pre-tau) spelling; catch it with a conversion hint rather than
        # letting check_config_keys report it as a generic unknown key.
        haskey(spec, "D") && error(
            "WrappedBrownian is parameterized by 'tau' (the coherence time in hours), not " *
                "'D'. Convert with tau = 2/D — and note the truncation bounds inverting: " *
                "an upper bound on D is a LOWER bound on tau. Got: $spec"
        )
        check_config_keys(spec, ("kind", "tau"), "a WrappedBrownian process spec")
        haskey(spec, "tau") || error(
            "WrappedBrownian process spec is missing 'tau', the phase coherence time in " *
                "hours: $spec"
        )
        return WrappedBrownian(; τ = _parse_hyper(spec["tau"]))
    elseif name in ("WrappedOrnsteinUhlenbeck", "wou")
        check_config_keys(
            spec, ("kind", "sigma", "tau", "mu"), "a WrappedOrnsteinUhlenbeck process spec"
        )
        haskey(spec, "sigma") || error("process spec is missing 'sigma': $spec")
        haskey(spec, "tau") || error("process spec is missing 'tau': $spec")
        μraw = get(spec, "mu", 0.0)
        μraw isa Real || error(
            "process 'mu' must be a fixed number (it is not fittable), got: $(repr(μraw))"
        )
        return WrappedOrnsteinUhlenbeck(;
            σ = _parse_hyper(spec["sigma"]), τ = _parse_hyper(spec["tau"]), μ = Float64(μraw)
        )
    elseif name in ("VonMisesProcess", "vm")
        check_config_keys(
            spec, ("kind", "sigma", "tau", "mu"), "a VonMisesProcess process spec"
        )
        haskey(spec, "sigma") || error("process spec is missing 'sigma': $spec")
        haskey(spec, "tau") || error("process spec is missing 'tau': $spec")
        μraw = get(spec, "mu", 0.0)
        μraw isa Real || error(
            "process 'mu' must be a fixed number (it is not fittable), got: $(repr(μraw))"
        )
        return VonMisesProcess(;
            σ = _parse_hyper(spec["sigma"]), τ = _parse_hyper(spec["tau"]), μ = Float64(μraw)
        )
    else
        error(
            "unknown process '$name'. Allowed: OrnsteinUhlenbeck, " *
                "WrappedBrownian, WrappedOrnsteinUhlenbeck, VonMisesProcess"
        )
    end
end

# --- initial-distribution specs -----------------------------------------------------------
# `p(x(t₁))`: how a chain's *first* time stamp is treated. Written either as a bare string
# (`init = "uniform"`) or as a table carrying that kind's arguments
# (`init = { kind = "fixed", value = 0.0 }`). Another closed allowlist, one entry per
# `AbstractInitialPrior` Comrade defines.

"""
    parse_init(spec, process, where_) -> AbstractInitialPrior

Parse the `init` entry of a `kind = "gaussmarkov"` site prior. `spec` is `nothing` (no
`init` given), a string, or a table with a `kind`:

  - `"stationary"`: the process's Gaussian stationary marginal (real-line processes only).
  - `"uniform"`: uniform on `(−π, π]`, the stationary law of a wrapped process — it lets
    the chain absorb the per-track phase offset, so no separate offset term is needed.
  - `{ kind = "fixed", value = 0.0 }`: the chain is conditioned to start at `value`, so it
    describes the drift *about* a separately parameterized level (e.g. a `gpratμ` offset).
  - `{ kind = "gaussian", mu = 0.0, sigma = 5.0 }`: an explicit `N(mu, sigma²)` first
    marginal, e.g. a deliberately diffuse start.

With no `init` the default is each process's own stationary law: uniform on the circle for
a wrapped process, the Gaussian stationary marginal otherwise. Comrade rejects the
combinations that do not exist (e.g. `"stationary"` for a wrapped process).
"""
function parse_init(spec, process, where_::AbstractString)
    # `is_wrapped` is the trait Comrade dispatches its own init checks on, so a new circular
    # process picks up the right default here without touching this function.
    # A stationary process starts in its own stationary marginal whether or not it is
    # wrapped — for `WrappedOrnsteinUhlenbeck` that is the wrapped normal WN(μ, σ²), NOT
    # uniform. UniformInit is the stationary circular law only of an unbounded wrapped
    # process (`WrappedBrownian`), and Comrade's `_check_init` accepts it for any wrapped
    # process, so defaulting on `is_wrapped` alone would silently pick the wrong start.
    if isnothing(spec)
        Comrade.isstationary(process) && return StationaryInit()
        return Comrade.is_wrapped(process) ? UniformInit() : StationaryInit()
    end
    if spec isa AbstractString
        return _init_from_kind(String(spec), Dict{String, Any}(), where_)
    elseif spec isa AbstractDict
        haskey(spec, "kind") || error("$where_ 'init' table is missing 'kind': $spec")
        return _init_from_kind(String(spec["kind"]), spec, where_)
    else
        error("$where_ 'init' must be a string or a table, got: $(repr(spec))")
    end
end

function _init_from_kind(kind::AbstractString, spec::AbstractDict, where_::AbstractString)
    if kind == "stationary"
        check_config_keys(spec, ("kind",), "$where_ init")
        return StationaryInit()
    elseif kind == "uniform"
        check_config_keys(spec, ("kind",), "$where_ init")
        return UniformInit()
    elseif kind == "fixed"
        check_config_keys(spec, ("kind", "value"), "$where_ init")
        haskey(spec, "value") ||
            error("$where_ init kind=\"fixed\" requires a 'value' (use a table, not a bare string)")
        return FixedInit(Float64(spec["value"]))
    elseif kind == "gaussian"
        check_config_keys(spec, ("kind", "mu", "sigma"), "$where_ init")
        haskey(spec, "sigma") ||
            error("$where_ init kind=\"gaussian\" requires a 'sigma' (use a table, not a bare string)")
        return GaussianInit(Float64(get(spec, "mu", 0.0)), Float64(spec["sigma"]))
    else
        error(
            "$where_ has unknown init kind '$kind'. Allowed: stationary, uniform, fixed, gaussian"
        )
    end
end
