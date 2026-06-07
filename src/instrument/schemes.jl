# Registry mapping a TOML scheme name to the closure (from parameterizations.jl) it
# selects and the exact set of parameter names that closure reads. The assembler uses the
# param set to validate that the TOML supplies a prior for every parameter the closure
# will destructure, and to build the matching `intprior` NamedTuple keys.
#
# Parameter names MUST match the fields accessed inside the closures in
# parameterizations.jl — they become the keys of the instrument prior NamedTuple.

const GAIN_SCHEMES = Dict{String, NamedTuple}(
    "gain" => (closure = gain, kind = :jones,
        params = (:lg1, :gp1, :lgratμ, :lgratσ, :lgrat, :gprat, :gpratμ)),
    "gain_centered" => (closure = gain_centered, kind = :jones,
        params = (:lg1, :gp1, :lgrat, :gprat)),
    "gain_hier" => (closure = gain_hier, kind = :jones,
        params = (:lg1μ, :lg1σ, :lg1, :gp1, :lgratμ, :lgratσ, :lgrat, :gpratμ, :gpratσ, :gprat)),
    "gain_noratio" => (closure = gain_noratio, kind = :jones,
        params = (:lg, :gp)),
    "singlegain" => (closure = singlegain, kind = :single,
        params = (:lg, :gp)),
)

const LEAKAGE_SCHEMES = Dict{String, NamedTuple}(
    "none" => (closure = nothing, params = ()),
    "leakage_simple" => (closure = leakage_simple,
        params = (:d1re, :d1im, :d2re, :d2im)),
    "leakage_hier" => (closure = leakage_hier,
        params = (:d1reμ, :d1reσ, :d1re, :d1imμ, :d1imσ, :d1im, :d2reμ, :d2reσ, :d2re, :d2imμ, :d2imσ, :d2im)),
)

const SEGMENTATIONS = Dict{String, Any}(
    "integ" => IntegSeg(),
    "scan" => ScanSeg(),
    "track" => TrackSeg(),
)
