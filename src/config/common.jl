# Small shared helpers for the TOML config layer.

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
