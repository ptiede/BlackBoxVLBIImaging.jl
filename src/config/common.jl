# Small shared helpers for the TOML config layer.

"""
    check_config_keys(cfg, allowed, where_)

Error if `cfg` contains keys outside `allowed`. Every config table has a closed schema; a
key we don't recognize would otherwise be silently ignored (e.g. a typo, or a top-level key
accidentally nested under the preceding TOML [section]), which is far worse than failing.
"""
function check_config_keys(cfg::AbstractDict, allowed, where_::AbstractString)
    unknown = sort!([String(k) for k in keys(cfg) if String(k) ∉ allowed])
    isempty(unknown) && return nothing
    return error(
        "unknown key(s) $(unknown) in $where_ — these would be silently ignored. " *
            "Allowed: $(sort!(collect(String.(allowed))))"
    )
end

"""
    imaging_executor()

Pick the image-domain executor based on the number of Julia threads: a threaded executor
when run with `-t N` (N>1), otherwise serial. BLAS itself is pinned to one thread in the
module init so that thread parallelism comes from here and from the NUFFT.
"""
function imaging_executor()
    if Threads.nthreads() > 1
        @info "Using $(Threads.nthreads()) threads for imaging"
        return ThreadsEx()
    else
        @info "Using a single thread for imaging"
        return Serial()
    end
end
