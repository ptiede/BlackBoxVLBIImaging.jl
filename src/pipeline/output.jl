# Output and plotting. The numeric/FITS/CSV writing lives here in the core. Image PNGs use
# CairoMakie's `imageviz` (CairoMakie is a hard dependency). The residual and caltable PNGs
# come from Plots recipes, which are gated behind the (weakdep) Plots extension via the
# hooks below — without Plots loaded those PNGs are skipped but the rest of the run proceeds.

_plot_backend_warn() = @warn "Plots not loaded; skipping residual/caltable PNGs. Load Plots (the driver does this) to enable them." maxlog = 1

# Image plotting via CairoMakie (always available).
function plot_image_png(path, img)
    p = imageviz(img)
    CairoMakie.save(path, p)
    return nothing
end

# --- Plots-backend hooks (overridden in BlackBoxVLBIImagingPlotsExt) --------------------
plot_residuals_png(path, post, x) = (_plot_backend_warn(); nothing)
plot_caltable_png(path, gtp) = nothing
plot_chain_residuals_png(path, post, post_samples) = nothing

# --- Core output -----------------------------------------------------------------------
"""
    save_optimal(out, post, xopt, gimg; label="optimal") -> IntensityMap

Render and save the MAP/optimal image as a FITS file (and, if a plotting backend is
loaded, a PNG). Returns the rendered image.
"""
function save_optimal(out, post, xopt, gimg; label = "optimal")
    img = intensitymap(skymodel(post, xopt), gimg)
    Comrade.save_fits(out * "_$(label).fits", img)
    plot_image_png(out * "_$(label).png", img)
    return img
end

"""
    write_caltables(out, xopt)

Write one CSV (and PNG via the plotting hook) calibration table per instrument quantity in
`xopt.instrument`. No-op if `xopt` has no instrument component.
"""
function write_caltables(out, xopt)
    hasproperty(xopt, :instrument) || return nothing
    for (ki, vi) in pairs(xopt.instrument)
        gtp = Comrade.caltable(vi)
        CSV.write(out * "_ctable_$(ki).csv", gtp)
        plot_caltable_png(out * "_ctable_$(ki).png", gtp)
    end
    return nothing
end

"""
    save_checkpoint(post, params, gimg, outbase, tag)

Render the current draw `params` (a constrained parameter NamedTuple) through `post`'s sky
model on grid `gimg` and write a checkpoint: a FITS image, an image PNG, and a residual PNG.
Host-side; used by both the Reactant optimizer (periodically) and the Reactant sampler
(per chunk, via `sample_callback`). `post` must be a CPU posterior and `params` host arrays.
"""
function save_checkpoint(post, params, gimg, outbase, tag)
    img = intensitymap(skymodel(post, params), gimg)
    Comrade.save_fits(outbase * "_$(tag).fits", img)
    plot_image_png(outbase * "_$(tag).png", img)
    plot_residuals_png(outbase * "_$(tag)_residuals.png", post, params)
    return nothing
end

"""
    save_posterior_draws(outdir, outbase, post, post_samples, gimg)

Save each posterior sky draw as a FITS file under `outdir`.
"""
function save_posterior_draws(outdir, outbase, post, post_samples, gimg)
    for (i, s) in enumerate(post_samples)
        f = joinpath(outdir, basename(outbase) * @sprintf("_draw_%03d.fits", i))
        Comrade.save_fits(f, intensitymap(s, gimg))
    end
    return nothing
end

# --- Analysis helpers (ported from the original utils.jl) ------------------------------
"""
    load_chain_and_post(file, nsamples=:) -> (chain, post)

Reload a saved chain and its posterior (from `<file>_optimum_allres.jls`).
"""
function load_chain_and_post(file, nsamples = Base.Colon())
    chain = load_samples(file, nsamples)
    post = deserialize(file * "_optimum_allres.jls")[:post]
    return chain, post
end

function saveimgs(imgs, outpath)
    for (i, img) in enumerate(imgs)
        Comrade.save_fits(joinpath(outpath, @sprintf("draw_img_%03d.fits", i)), img)
    end
    return nothing
end
