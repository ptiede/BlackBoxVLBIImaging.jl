# Plots-recipe plotting hooks for BlackBoxVLBIImaging, loaded automatically when Plots is
# available (the driver loads it). Implements the residual and caltable hooks declared in
# pipeline/output.jl via the `residual`/`residual!` user-recipes and the caltable recipe
# from Comrade. (Image PNGs are handled in the core via CairoMakie's `imageviz`.)

module BlackBoxVLBIImagingPlotsExt

using BlackBoxVLBIImaging
using Comrade
using Plots

# `path::AbstractString` makes these strictly more specific than the core's `(Any, Any...)`
# fallbacks, so they extend rather than overwrite them (the latter is illegal at precompile).
function BlackBoxVLBIImaging.plot_residuals_png(path::AbstractString, post, x)
    p = residual(post, x)
    Plots.savefig(p, path)
    return nothing
end

function BlackBoxVLBIImaging.plot_caltable_png(path::AbstractString, gtp)
    p = Plots.plot(gtp, layout = (4, 4), size = (800, 500))
    Plots.savefig(p, path)
    return nothing
end

function BlackBoxVLBIImaging.plot_chain_residuals_png(path::AbstractString, post, post_samples)
    isempty(post_samples) && return nothing
    p = residual(post, post_samples[begin])
    for s in post_samples[(begin + 1):end]
        residual!(p, post, s)
    end
    Plots.savefig(p, path)
    return nothing
end

end
