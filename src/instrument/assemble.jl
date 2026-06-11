# The generic instrument-model assembler. Turns a parsed instrument TOML into a Comrade
# `InstrumentModel` by selecting a gain/leakage scheme (schemes.jl), building one
# `ArrayPrior` per required parameter from its prior table, and composing the Jones
# matrices. This replaces the ~20 hand-written builders with a single declarative path.

function _parse_refant(spec)
    isnothing(spec) && return NoReference()
    check_config_keys(spec, ("kind", "val", "site"), "a refant spec")
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

function _iid_site_prior(t::AbstractDict, where_::AbstractString)
    haskey(t, "seg") || error("$where_ is missing 'seg': $t")
    haskey(t, "dist") || error("$where_ is missing 'dist': $t")
    return IIDSitePrior(_segmentation(String(t["seg"])), parse_dist(t["dist"]))
end

function _build_array_prior(pcfg::AbstractDict, name::AbstractString)
    check_config_keys(
        pcfg, ("seg", "dist", "phase", "refant", "overrides"), "[priors.$name]"
    )
    default = _iid_site_prior(pcfg, "[priors.$name]")
    phase = Bool(get(pcfg, "phase", false))
    refant = _parse_refant(get(pcfg, "refant", nothing))

    overrides = get(pcfg, "overrides", Dict{String, Any}())
    # Site overrides replace only the IIDSitePrior, so `phase`/`refant` are not accepted
    # here — they live on the parameter-level entry.
    ovr_pairs = map(collect(overrides)) do (site, scfg)
        check_config_keys(scfg, ("seg", "dist"), "[priors.$name.overrides.$site]")
        return Symbol(site) => _iid_site_prior(scfg, "[priors.$name.overrides.$site]")
    end
    ovr = NamedTuple(ovr_pairs)

    return ArrayPrior(default; refant = refant, phase = phase, ovr...)
end

"""
    assemble_instrument(cfg::AbstractDict) -> InstrumentModel

Build a Comrade `InstrumentModel` from a parsed instrument TOML. `cfg` must contain a
`[gain]` section with a `scheme`, an optional `[leakage]` section, an optional
`frcal` flag, and a `[priors]` table with one entry per parameter required by the
chosen gain (and leakage) scheme. Throws if any required prior is missing.
"""
function assemble_instrument(cfg::AbstractDict)
    check_config_keys(
        cfg, ("gain", "leakage", "frcal", "priors"), "the instrument config (top level)"
    )

    haskey(cfg, "gain") || error("instrument config needs a [gain] section")
    # frcal is top-level; a `frcal` line placed below a [gain]/[leakage] header parses as a
    # nested key and would otherwise be silently ignored.
    for sec in ("gain", "leakage")
        haskey(cfg, sec) && haskey(cfg[sec], "frcal") && error(
            "'frcal' was found inside [$sec] — it is a top-level key, so move it above the " *
                "first [section] header in the instrument TOML."
        )
        haskey(cfg, sec) && check_config_keys(cfg[sec], ("scheme",), "[$sec]")
    end

    gname = String(get(cfg["gain"], "scheme", ""))
    haskey(GAIN_SCHEMES, gname) ||
        error("unknown gain scheme '$gname'. Allowed: $(sort(collect(keys(GAIN_SCHEMES))))")
    gsch = GAIN_SCHEMES[gname]

    lname = haskey(cfg, "leakage") ? String(get(cfg["leakage"], "scheme", "none")) : "none"
    haskey(LEAKAGE_SCHEMES, lname) ||
        error("unknown leakage scheme '$lname'. Allowed: $(sort(collect(keys(LEAKAGE_SCHEMES))))")
    lsch = LEAKAGE_SCHEMES[lname]

    frcal = Bool(get(cfg, "frcal", false))

    required = Symbol[gsch.params..., lsch.params...]
    priors = get(cfg, "priors", Dict{String, Any}())
    missing_params = filter(p -> !haskey(priors, String(p)), required)
    isempty(missing_params) ||
        error("instrument config is missing priors for: $(missing_params)")

    # Priors for parameters the chosen schemes never read are legal (e.g. kept around while
    # switching schemes) but inert, so say so rather than silently skipping them.
    unused = sort!([k for k in keys(priors) if Symbol(k) ∉ required])
    isempty(unused) ||
        @warn "instrument [priors] entries unused by gain=$gname/leakage=$lname (ignored): $(unused)"

    intprior = NamedTuple([p => _build_array_prior(priors[String(p)], String(p)) for p in required])

    G = gsch.kind === :single ? SingleStokesGain(gsch.parameterization) : JonesG(gsch.parameterization)

    if lname == "none"
        J = G
    else
        D = JonesD(lsch.parameterization)
        R = JonesR(; add_fr = true)
        sw = frcal ? sandwich_withfrcal : sandwich
        J = JonesSandwich(sw, G, D, R)
    end

    @info "Instrument: gain=$gname leakage=$lname frcal=$frcal nparams=$(length(required))"
    return InstrumentModel(J, intprior)
end
