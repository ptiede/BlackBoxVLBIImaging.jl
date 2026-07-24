# Data flagging with a closed TOML schema. The accepted keys are fixed (deliberately no
# `@eval` of config strings); adding a new flag axis means extending `apply_flagtable`
# here, not authoring cleverness in the TOML.
#
# Supported keys (all optional):
#   corr_polbasis = [{site = "HAY", R = "Y", L = "X"}]  # feed-by-feed corrections
#   sites         = ["AP"]          # drop all baselines touching these sites
#   baselines     = [["AA","LM"]]   # drop these specific (order-independent) baselines
#   drop_tranges  = [[4.5, 5.2]]    # drop datums with Ti in these UT-decimal-hour ranges
#   uvranges      = [[0.0, 1.0e8]]  # drop datums with uvdist in these ranges (units: λ)
#
# (`drop_tranges` removes time windows; the load-time `[data] keep_trange` *selects* one.)

"""
    parse_flagtable(cfg::AbstractDict) -> NamedTuple

Parse and validate flag-table keys out of a TOML dict (`cfg`). Shared by
[`read_flagtable`](@ref) and the data-config parser so a single `data.toml` can both load
and flag the data.
"""
function _parse_corr_polbasis_feed(feed, field, index)
    String(feed) == "R" && return RPol()
    String(feed) == "L" && return LPol()
    String(feed) == "X" && return XPol()
    String(feed) == "Y" && return YPol()
    error("corr_polbasis[$index].$field must be one of R, L, X, or Y; got '$feed'")
end

function _parse_corr_polbasis(entries)
    corrections = NamedTuple{(:site, :mapping), Tuple{Symbol, Vector{Pair{Any, Any}}}}[]
    for (i, entry) in enumerate(entries)
        if entry isa AbstractString
            push!(corrections, (; site = Symbol(entry), mapping = Pair{Any, Any}[DEFAULT_CORPOL_MAPPING...]))
        elseif entry isa AbstractDict
            check_config_keys(entry, ("site", "R", "L", "X", "Y"), "corr_polbasis[$i]")
            haskey(entry, "site") || error("corr_polbasis[$i] needs a 'site'")
            source_labels = filter(label -> haskey(entry, label), ("R", "L", "X", "Y"))
            length(source_labels) == 2 || error(
                "corr_polbasis[$i] must map exactly two source feeds (e.g. R = \"Y\", L = \"X\")"
            )
            mapping = Pair{Any, Any}[
                _parse_corr_polbasis_feed(source, source, i) =>
                    _parse_corr_polbasis_feed(entry[source], source, i)
                for source in source_labels
            ]
            length(unique(last.(mapping))) == 2 || error("corr_polbasis[$i] target feeds must be distinct")
            push!(corrections, (; site = Symbol(entry["site"]), mapping))
        else
            error("corr_polbasis[$i] must be a site string or {site, R/L/X/Y mappings} table, got $entry")
        end
    end
    return corrections
end

function parse_flagtable(cfg::AbstractDict)
    check_config_keys(
        cfg, ("corr_polbasis", "sites", "baselines", "drop_tranges", "uvranges"),
        "the [flags] table"
    )
    sites_corr = _parse_corr_polbasis(get(cfg, "corr_polbasis", Any[]))
    sites = Symbol.(get(cfg, "sites", String[]))
    baselines = [Set(Symbol.(bl)) for bl in get(cfg, "baselines", Vector{String}[])]
    drop_tranges = [Tuple(Float64.(t)) for t in get(cfg, "drop_tranges", Vector{Float64}[])]
    uvranges = [Tuple(Float64.(u)) for u in get(cfg, "uvranges", Vector{Float64}[])]

    for (i, bl) in enumerate(baselines)
        length(bl) == 2 || error("flag table: baselines[$i] must name two distinct sites, got $bl")
    end
    for (i, t) in enumerate(drop_tranges)
        length(t) == 2 || error("flag table: drop_tranges[$i] must be [a, b], got $t")
    end
    for (i, u) in enumerate(uvranges)
        length(u) == 2 || error("flag table: uvranges[$i] must be [a, b], got $u")
    end

    @info "Flag table: corr_polbasis=$(length(sites_corr)) sites=$(length(sites)) baselines=$(length(baselines)) drop_tranges=$(length(drop_tranges)) uvranges=$(length(uvranges))"
    return (; corr_polbasis = sites_corr, sites, baselines, drop_tranges, uvranges)
end

"""
    read_flagtable(path::String) -> NamedTuple

Parse a standalone flag-table TOML file. See [`parse_flagtable`](@ref) for the schema.
"""
function read_flagtable(path::String)
    return parse_flagtable(TOML.parsefile(path))
end

"""
    apply_flagtable(dvis, cfg) -> dvis

Apply the parsed flag-table `cfg` to a coherency table `dvis`, running one
`corr_polbasis`/`flag` operation per entry and logging how many datums each drops.
"""
function apply_flagtable(dvis, cfg)
    for correction in cfg.corr_polbasis
        @info "corr_polbasis: $(correction.site) $(correction.mapping)"
        dvis = corr_polbasis(dvis, correction.site, correction.mapping)
    end
    for s in cfg.sites
        n0 = length(dvis)
        dvis = flag(x -> s ∈ x.baseline.sites, dvis)
        @info "  flag site=$s dropped $(n0 - length(dvis)) datums"
    end
    for bl in cfg.baselines
        n0 = length(dvis)
        dvis = flag(x -> Set(x.baseline.sites) == bl, dvis)
        @info "  flag baseline=$(collect(bl)) dropped $(n0 - length(dvis)) datums"
    end
    for (a, b) in cfg.drop_tranges
        n0 = length(dvis)
        dvis = flag(x -> a <= x.baseline.Ti <= b, dvis)
        @info "  flag drop_trange=[$a, $b] dropped $(n0 - length(dvis)) datums"
    end
    for (a, b) in cfg.uvranges
        n0 = length(dvis)
        dvis = flag(x -> a <= uvdist(x) <= b, dvis)
        @info "  flag uvrange=[$a, $b] dropped $(n0 - length(dvis)) datums"
    end
    return dvis
end
