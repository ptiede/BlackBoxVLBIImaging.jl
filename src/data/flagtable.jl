# Data flagging with a closed TOML schema. The accepted keys are fixed (deliberately no
# `@eval` of config strings); adding a new flag axis means extending `apply_flagtable`
# here, not authoring cleverness in the TOML.
#
# Supported keys (all optional):
#   corr_polbasis = ["HAY", "AA"]   # sites whose polbasis label is flipped
#   sites         = ["AP"]          # drop all baselines touching these sites
#   baselines     = [["AA","LM"]]   # drop these specific (order-independent) baselines
#   tranges       = [[4.5, 5.2]]    # drop datums with Ti in these UT-decimal-hour ranges
#   uvranges      = [[0.0, 1.0e8]]  # drop datums with uvdist in these ranges (units: λ)

"""
    parse_flagtable(cfg::AbstractDict) -> NamedTuple

Parse and validate flag-table keys out of a TOML dict (`cfg`). Shared by
[`read_flagtable`](@ref) and the data-config parser so a single `data.toml` can both load
and flag the data.
"""
function parse_flagtable(cfg::AbstractDict)
    sites_corr = Symbol.(get(cfg, "corr_polbasis", String[]))
    sites = Symbol.(get(cfg, "sites", String[]))
    baselines = [Set(Symbol.(bl)) for bl in get(cfg, "baselines", Vector{String}[])]
    tranges = [Tuple(Float64.(t)) for t in get(cfg, "tranges", Vector{Float64}[])]
    uvranges = [Tuple(Float64.(u)) for u in get(cfg, "uvranges", Vector{Float64}[])]

    for (i, bl) in enumerate(baselines)
        length(bl) == 2 || error("flag table: baselines[$i] must name two distinct sites, got $bl")
    end
    for (i, t) in enumerate(tranges)
        length(t) == 2 || error("flag table: tranges[$i] must be [a, b], got $t")
    end
    for (i, u) in enumerate(uvranges)
        length(u) == 2 || error("flag table: uvranges[$i] must be [a, b], got $u")
    end

    @info "Flag table: corr_polbasis=$(length(sites_corr)) sites=$(length(sites)) baselines=$(length(baselines)) tranges=$(length(tranges)) uvranges=$(length(uvranges))"
    return (; corr_polbasis = sites_corr, sites, baselines, tranges, uvranges)
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
    for s in cfg.corr_polbasis
        @info "corr_polbasis: $s"
        dvis = corr_polbasis(dvis, s)
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
    for (a, b) in cfg.tranges
        n0 = length(dvis)
        dvis = flag(x -> a <= x.baseline.Ti <= b, dvis)
        @info "  flag trange=[$a, $b] dropped $(n0 - length(dvis)) datums"
    end
    for (a, b) in cfg.uvranges
        n0 = length(dvis)
        dvis = flag(x -> a <= uvdist(x) <= b, dvis)
        @info "  flag uvrange=[$a, $b] dropped $(n0 - length(dvis)) datums"
    end
    return dvis
end
