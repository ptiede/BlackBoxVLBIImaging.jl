# The generic instrument-model assembler. Turns a parsed instrument TOML into a Comrade
# `InstrumentModel` by selecting a gain/leakage `@instrument` scheme (instruments.jl),
# building one `ArrayPrior` per required parameter from its prior table, and composing the
# gain and leakage pieces into a `JonesSandwich` when leakage is requested.

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

# A single site prior, either the default `IIDSitePrior` (temporally independent, `dist`) or
# the temporally correlated `GaussMarkovSitePrior` (a Gauss-Markov `process` + optional
# `centered`/`anchored` flags). The kind is selected by `kind = "iid"` (default) /
# `"gaussmarkov"`; both are accepted by `ArrayPrior` as the default or as a per-site
# override, so this one builder serves both call sites. `anchored = true` pins each site's
# chain to zero at its first time (the phase-fluctuation-plus-circular-offset idiom).
function _site_prior(t::AbstractDict, where_::AbstractString)
    haskey(t, "seg") || error("$where_ is missing 'seg': $t")
    seg = _segmentation(String(t["seg"]))
    kind = String(get(t, "kind", "iid"))
    if kind == "iid"
        haskey(t, "dist") || error("$where_ is missing 'dist': $t")
        return IIDSitePrior(seg, parse_dist(t["dist"]))
    elseif kind == "gaussmarkov"
        haskey(t, "process") ||
            error("$where_ has kind=\"gaussmarkov\" but is missing 'process': $t")
        # GaussMarkovSitePrior requires a time segmentation; every registered segmentation
        # is one, but keep the invariant explicit for when a non-time seg is added.
        seg isa Comrade.TimeSegmentation || error(
            "$where_ uses kind=\"gaussmarkov\", which needs a time segmentation " *
                "(integ/scan/track), got seg=\"$(t["seg"])\""
        )
        centered = Bool(get(t, "centered", false))
        anchored = Bool(get(t, "anchored", false))
        return GaussMarkovSitePrior(
            seg, parse_process(t["process"]); centered = centered, anchored = anchored
        )
    else
        error("$where_ has unknown site-prior kind '$kind'. Allowed: iid, gaussmarkov")
    end
end

function _build_array_prior(pcfg::AbstractDict, name::AbstractString)
    check_config_keys(
        pcfg,
        ("kind", "seg", "dist", "process", "centered", "anchored", "phase", "refant", "overrides"),
        "[priors.$name]",
    )
    default = _site_prior(pcfg, "[priors.$name]")
    phase = Bool(get(pcfg, "phase", false))
    refant = _parse_refant(get(pcfg, "refant", nothing))

    overrides = get(pcfg, "overrides", Dict{String, Any}())
    # Site overrides replace only the site prior, so `phase`/`refant` are not accepted here
    # — they live on the parameter-level entry. Each override may itself be iid or gaussmarkov.
    ovr_pairs = map(collect(overrides)) do (site, scfg)
        check_config_keys(
            scfg, ("kind", "seg", "dist", "process", "centered", "anchored"),
            "[priors.$name.overrides.$site]",
        )
        return Symbol(site) => _site_prior(scfg, "[priors.$name.overrides.$site]")
    end
    ovr = NamedTuple(ovr_pairs)

    # GaussMarkov site priors are incompatible with a phase-wrapped ArrayPrior; catch it
    # here with a parameter-named error rather than letting ArrayPrior throw generically.
    if phase && any(Base.Fix2(isa, GaussMarkovSitePrior), (default, values(ovr)...))
        error(
            "[priors.$name] combines phase=true with a GaussMarkov site prior, which " *
                "ArrayPrior does not support — use kind=\"iid\" for phase parameters."
        )
    end

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
    gctor = GAIN_SCHEMES[gname]

    lname = haskey(cfg, "leakage") ? String(get(cfg["leakage"], "scheme", "none")) : "none"
    haskey(LEAKAGE_SCHEMES, lname) ||
        error("unknown leakage scheme '$lname'. Allowed: $(sort(collect(keys(LEAKAGE_SCHEMES))))")
    lctor = LEAKAGE_SCHEMES[lname]

    frcal = Bool(get(cfg, "frcal", false))

    required = Symbol[required_params(gctor)..., required_params(lctor)...]
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

    @info "Instrument: gain=$gname leakage=$lname frcal=$frcal nparams=$(length(required))"

    gm = gctor(; priors = intprior)
    lctor === nothing && return gm

    # Compose the gain and leakage pieces (each an InstrumentModel over its own tilde
    # parameters) into G*D*R with the feed-rotation term, merging their priors.
    dm = lctor(; priors = intprior)
    sw = frcal ? sandwich_withfrcal : sandwich
    J = JonesSandwich(sw, gm.jones, dm.jones, JonesR(; add_fr = true))
    return InstrumentModel(J, merge(gm.prior, dm.prior); refbasis = gm.refbasis)
end
