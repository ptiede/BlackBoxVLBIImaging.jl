# The generic instrument-model assembler. Turns a parsed instrument TOML into a Comrade
# `InstrumentModel` by selecting a gain/leakage scheme (schemes.jl), building one
# `ArrayPrior` per required parameter from its prior table, and composing the Jones
# matrices. This replaces the ~20 hand-written builders with a single declarative path.

function _parse_refant(spec)
    isnothing(spec) && return NoReference()
    kind = String(get(spec, "kind", "None"))
    if kind == "None"
        return NoReference()
    elseif kind == "SEFD"
        return SEFDReference(Float64(get(spec, "val", 0.0)))
    elseif kind == "Single"
        haskey(spec, "site") || error("refant kind='Single' requires a 'site'")
        return SingleReference(Symbol(spec["site"]), Float64(get(spec, "val", 0.0)))
    else
        error("unknown refant kind '$kind'. Allowed: None, SEFD, Single")
    end
end

function _segmentation(name)
    haskey(SEGMENTATIONS, name) ||
        error("unknown segmentation '$name'. Allowed: $(sort(collect(keys(SEGMENTATIONS))))")
    return SEGMENTATIONS[name]
end

function _iid_site_prior(t::AbstractDict)
    haskey(t, "seg") || error("prior entry is missing 'seg': $t")
    haskey(t, "dist") || error("prior entry is missing 'dist': $t")
    return IIDSitePrior(_segmentation(String(t["seg"])), parse_dist(t["dist"]))
end

function _build_array_prior(pcfg::AbstractDict)
    default = _iid_site_prior(pcfg)
    phase = Bool(get(pcfg, "phase", false))
    refant = _parse_refant(get(pcfg, "refant", nothing))

    overrides = get(pcfg, "overrides", Dict{String, Any}())
    ovr_pairs = [Symbol(site) => _iid_site_prior(scfg) for (site, scfg) in overrides]
    ovr = NamedTuple(ovr_pairs)

    return ArrayPrior(default; refant = refant, phase = phase, ovr...)
end

"""
    assemble_instrument(cfg::AbstractDict) -> InstrumentModel

Build a Comrade `InstrumentModel` from a parsed instrument TOML. `cfg` must contain a
`[gain]` section with a `scheme`, an optional `[leakage]` section, an optional
`field_rotation` flag, and a `[priors]` table with one entry per parameter required by the
chosen gain (and leakage) scheme. Throws if any required prior is missing.
"""
function assemble_instrument(cfg::AbstractDict)
    haskey(cfg, "gain") || error("instrument config needs a [gain] section")
    gname = String(get(cfg["gain"], "scheme", "") )
    haskey(GAIN_SCHEMES, gname) ||
        error("unknown gain scheme '$gname'. Allowed: $(sort(collect(keys(GAIN_SCHEMES))))")
    gsch = GAIN_SCHEMES[gname]

    lname = haskey(cfg, "leakage") ? String(get(cfg["leakage"], "scheme", "none")) : "none"
    haskey(LEAKAGE_SCHEMES, lname) ||
        error("unknown leakage scheme '$lname'. Allowed: $(sort(collect(keys(LEAKAGE_SCHEMES))))")
    lsch = LEAKAGE_SCHEMES[lname]

    field_rotation = Bool(get(cfg, "field_rotation", false))

    required = Symbol[gsch.params..., lsch.params...]
    priors = get(cfg, "priors", Dict{String, Any}())
    missing_params = filter(p -> !haskey(priors, String(p)), required)
    isempty(missing_params) ||
        error("instrument config is missing priors for: $(missing_params)")

    intprior = NamedTuple([p => _build_array_prior(priors[String(p)]) for p in required])

    G = gsch.kind === :single ? SingleStokesGain(gsch.closure) : JonesG(gsch.closure)

    if lname == "none"
        J = G
    else
        D = JonesD(lsch.closure)
        R = JonesR(; add_fr = true)
        sw = field_rotation ? sandwich_withfrcal : sandwich
        J = JonesSandwich(sw, G, D, R)
    end

    @info "Instrument: gain=$gname leakage=$lname field_rotation=$field_rotation nparams=$(length(required))"
    return InstrumentModel(J, intprior)
end
