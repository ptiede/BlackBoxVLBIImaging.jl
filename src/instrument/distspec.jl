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
