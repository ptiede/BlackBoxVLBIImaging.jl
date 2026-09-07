# Parse the data TOML: load the data (uvfits or dlist) and apply the flag table. The TOML
# is organized into three sections — `[paths]` (file + optional array + path resolution),
# `[data]` (format/averaging/noise + a load-time keep window), and `[flags]` (post-load
# flagging; see flagtable.jl for the schema).

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
    build_data_config(cfg::AbstractDict; base_dir=pwd(), polrep=PolExp()) -> EHTObservationTable

Load and flag the data described by a parsed data TOML. `polrep` (the sky model's
polarization representation) selects the Comrade data product: `TotalIntensity` extracts
complex visibilities, any polarized representation extracts coherency matrices (the
default). Sections:

- `[paths]` — `file`, optional `array`, and `path_mode` (`"toml"` resolves relative
  `file`/`array` against `base_dir` — the TOML's own directory when called via the driver —
  and `"cwd"` leaves them relative to the launch directory; absolute paths are used as-is
  either way). `array` supplies feed-rotation/mount overrides; omit it for uvfits to keep the
  data file's own mount metadata. dlist requires it (it builds the antenna table from it).
- `[data]` — `format` (`"auto"`/`"uvfits"`/`"dlist"`), `avg`, `ferr`, `IF` (1-based single
  intermediate-frequency selection for a multi-IF UVFITS file; omit it for the frequency-
  averaged band), `keep_trange` (a single `[lo, hi]` UT-hour window the data is restricted
  to at load; `[]` = keep all), `conjugate` (uvfits only; `true` flips the phase-sign
  convention of the file, see [`conjugate_data`](@ref)), and `ignore_feed_labels` (uvfits
  only; `true` reads the RR/LL/RL/LR slots as circular for every station regardless of the
  antenna table, see [`force_circular_feed_labels!`](@ref); declare linear stations with
  `[flags] corr_polbasis` instead).
- `[flags]` — the flag-table keys consumed by [`parse_flagtable`](@ref), including
  `drop_tranges` (UT-hour windows to remove). `keep_trange` selects; `drop_tranges` removes.
"""
function build_data_config(
        cfg::AbstractDict; base_dir::AbstractString = pwd(), polrep::PolRep = PolExp()
    )
    check_config_keys(cfg, ("paths", "data", "flags"), "the data config (top level)")
    haskey(cfg, "paths") || error("data config needs a [paths] section with a 'file'")
    paths = cfg["paths"]
    check_config_keys(paths, ("file", "array", "path_mode"), "[paths]")
    haskey(paths, "file") || error("data config [paths] needs a 'file'")

    path_mode = String(get(paths, "path_mode", "toml"))
    path_mode in ("toml", "cwd") || error(
        "paths.path_mode must be \"toml\" (resolve relative file/array against the TOML's " *
            "directory) or \"cwd\" (against the launch directory); got '$path_mode'"
    )
    _resolve(p) = (path_mode == "toml" && !isabspath(p)) ? abspath(joinpath(base_dir, p)) : p
    file = _resolve(String(paths["file"]))
    # `array` is optional (feed-rotation/mount overrides): omit it for uvfits to keep the
    # data file's own mount metadata. dlist still requires it (build_data_dlist enforces this).
    array = haskey(paths, "array") ? _resolve(String(paths["array"])) : nothing

    dat = get(cfg, "data", Dict{String, Any}())
    check_config_keys(
        dat, ("format", "avg", "ferr", "IF", "keep_trange", "conjugate", "ignore_feed_labels"),
        "[data]"
    )
    fmt = String(get(dat, "format", "auto"))
    fmt = fmt == "auto" ? _infer_format(file) : fmt
    avg = string(get(dat, "avg", "scan"))
    ferr = Float64(get(dat, "ferr", 0.005))
    kt = get(dat, "keep_trange", Float64[])
    keep_trange = isempty(kt) ? nothing : Tuple(Float64.(kt))
    # `IF` (1-based, matching Julia indexing) selects a single intermediate frequency from a
    # multi-IF UVFITS file; omit it to keep the frequency-averaged band. dlist files have no IFs.
    IF = haskey(dat, "IF") ? Int(dat["IF"]) : nothing
    conjugate = Bool(get(dat, "conjugate", false))
    ignore_feed_labels = Bool(get(dat, "ignore_feed_labels", false))

    @info "Loading $fmt data: $file"
    if fmt == "uvfits"
        dcoh = build_data_uvfits(
            file, array; avg, ferr, trange = keep_trange, IF, polrep, conjugate, ignore_feed_labels
        )
    elseif fmt == "dlist"
        isnothing(IF) || @warn "data.IF is set but dlist files have no IFs; ignoring."
        conjugate && @warn "data.conjugate is set but only supported for uvfits; ignoring."
        ignore_feed_labels && @warn "data.ignore_feed_labels is set but only applies to uvfits; ignoring."
        dcoh = build_data_dlist(file, array; avg, ferr, trange = keep_trange, polrep)
    else
        error("unknown data format '$fmt'. Allowed: uvfits, dlist (or 'auto')")
    end

    dcoh = apply_flagtable(dcoh, parse_flagtable(get(cfg, "flags", Dict{String, Any}())))
    # After averaging + flagging, make sure every timestamp is covered by a scan; otherwise
    # any scan-segmented instrument parameter fails its segment lookup in `set_array`.
    # TODO upsteam to VLBIFiles?
    dcoh = repair_scan_coverage(dcoh)
    return dcoh
end
