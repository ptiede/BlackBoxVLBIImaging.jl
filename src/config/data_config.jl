# Parse the data TOML: load coherencies (uvfits or dlist) and apply the flag table. The TOML
# is organized into three sections — `[paths]` (file/array + path resolution), `[data]`
# (format/averaging/noise + a load-time keep window), and `[flags]` (post-load flagging; see
# flagtable.jl for the schema).

function _infer_format(file::AbstractString)
    if endswith(file, ".uvfits") || endswith(file, ".uvf")
        return "uvfits"
    elseif endswith(file, ".dlist")
        return "dlist"
    else
        error("cannot infer data format from '$file'; set [data] format = \"uvfits\" or \"dlist\"")
    end
end

"""
    build_data_config(cfg::AbstractDict; base_dir=pwd()) -> EHTObservationTable

Load and flag the coherency data described by a parsed data TOML. Sections:

- `[paths]` — `file`, `array`, and `path_mode` (`"toml"` resolves relative `file`/`array`
  against `base_dir` — the TOML's own directory when called via the driver — and `"cwd"`
  leaves them relative to the launch directory; absolute paths are used as-is either way).
- `[data]` — `format` (`"auto"`/`"uvfits"`/`"dlist"`), `avg`, `ferr`, and `keep_trange`
  (a single `[lo, hi]` UT-hour window the data is restricted to at load; `[]` = keep all).
- `[flags]` — the flag-table keys consumed by [`parse_flagtable`](@ref), including
  `drop_tranges` (UT-hour windows to remove). `keep_trange` selects; `drop_tranges` removes.
"""
function build_data_config(cfg::AbstractDict; base_dir::AbstractString = pwd())
    haskey(cfg, "paths") || error("data config needs a [paths] section with 'file' and 'array'")
    paths = cfg["paths"]
    haskey(paths, "file") || error("data config [paths] needs a 'file'")
    haskey(paths, "array") || error("data config [paths] needs an 'array'")

    path_mode = String(get(paths, "path_mode", "toml"))
    path_mode in ("toml", "cwd") || error(
        "paths.path_mode must be \"toml\" (resolve relative file/array against the TOML's " *
            "directory) or \"cwd\" (against the launch directory); got '$path_mode'"
    )
    _resolve(p) = (path_mode == "toml" && !isabspath(p)) ? abspath(joinpath(base_dir, p)) : p
    file = _resolve(String(paths["file"]))
    array = _resolve(String(paths["array"]))

    dat = get(cfg, "data", Dict{String, Any}())
    fmt = String(get(dat, "format", "auto"))
    fmt = fmt == "auto" ? _infer_format(file) : fmt
    avg = string(get(dat, "avg", "scan"))
    ferr = Float64(get(dat, "ferr", 0.005))
    kt = get(dat, "keep_trange", Float64[])
    keep_trange = isempty(kt) ? nothing : Tuple(Float64.(kt))

    @info "Loading $fmt data: $file"
    if fmt == "uvfits"
        dcoh = build_data_uvfits(file, array; avg, ferr, trange = keep_trange)
    elseif fmt == "dlist"
        dcoh = build_data_dlist(file, array; avg, ferr, trange = keep_trange)
    else
        error("unknown data format '$fmt'. Allowed: uvfits, dlist (or 'auto')")
    end

    dcoh = apply_flagtable(dcoh, parse_flagtable(get(cfg, "flags", Dict{String, Any}())))
    return dcoh
end
