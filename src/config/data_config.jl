# Parse the data TOML: load coherencies (uvfits or dlist) and apply the flag table. The
# flag-table keys live in the same file (see flagtable.jl for the schema).

function _infer_format(file::AbstractString)
    if endswith(file, ".uvfits") || endswith(file, ".uvf")
        return "uvfits"
    elseif endswith(file, ".dlist")
        return "dlist"
    else
        error("cannot infer data format from '$file'; set format = \"uvfits\" or \"dlist\"")
    end
end

"""
    build_data_config(cfg::AbstractDict) -> EHTObservationTable

Load and flag the coherency data described by a parsed data TOML. Required keys: `file`,
`array`. Optional: `format` (`"auto"`/`"uvfits"`/`"dlist"`), `avg`, `ferr`, `trange`, plus
the flag-table keys consumed by [`parse_flagtable`](@ref).
"""
function build_data_config(cfg::AbstractDict)
    haskey(cfg, "file") || error("data config needs a 'file'")
    haskey(cfg, "array") || error("data config needs an 'array'")
    file = String(cfg["file"])
    array = String(cfg["array"])

    fmt = String(get(cfg, "format", "auto"))
    fmt = fmt == "auto" ? _infer_format(file) : fmt
    avg = string(get(cfg, "avg", "scan"))
    ferr = Float64(get(cfg, "ferr", 0.005))
    tr = get(cfg, "trange", Float64[])
    trange = isempty(tr) ? nothing : Tuple(Float64.(tr))

    @info "Loading $fmt data: $file"
    if fmt == "uvfits"
        dcoh = build_data_uvfits(file, array; avg, ferr, trange)
    elseif fmt == "dlist"
        dcoh = build_data_dlist(file, array; avg, ferr, trange)
    else
        error("unknown data format '$fmt'. Allowed: uvfits, dlist (or 'auto')")
    end

    flag = parse_flagtable(cfg)
    dcoh = apply_flagtable(dcoh, flag)
    return dcoh
end
