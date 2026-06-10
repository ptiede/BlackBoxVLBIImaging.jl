# Output and plotting. The numeric/FITS/CSV writing lives here in the core. All PNGs render
# through CairoMakie (a hard dependency): images via `imageviz`, residuals via the helpers
# below, and caltables via Comrade's `plotcaltable` — so every run writes them with no
# optional plotting backend to load.

# Image plotting via CairoMakie (always available).
function plot_image_png(path, img)
    p = imageviz(img)
    CairoMakie.save(path, p)
    return nothing
end

# --- Caltable PNG via Comrade's CairoMakie `plotcaltable` (one subplot per station) ------
function plot_caltable_png(path, gtp)
    CairoMakie.save(path, Comrade.plotcaltable(gtp))
    return nothing
end

# --- Residual PNGs via CairoMakie (hard dep, so these always render) ---------------------
# Mirrors Comrade's Plots `residual` recipe: normalized residuals (data−model)/σ scattered
# against uv-distance (Gλ). For polarized coherency data the four feed correlations are split
# into a 2×2 grid (RR/RL over LR/LL), each panel titled with its reduced χ². Multiple
# `post_samples` are overlaid (chain residuals).

_avg(f, xs) = sum(f, xs) / length(xs)
# reduced χ² of a set of normalized residuals (mean of squares, NaNs dropped).
_redchi2(v) = (w = filter(!isnan, v); isempty(w) ? NaN : sum(abs2, w) / length(w))

# Per-residual-table view: uv-distance (Gλ) + the normalized-residual columns, tagged by kind.
function _resid_components(rest)
    dt = Comrade.datatable(rest)
    uv = Comrade.uvdist.(dt) ./ 1.0e9
    res = map(d -> d.measurement ./ d.noise, dt)
    if rest isa Comrade.EHTObservationTable{<:Comrade.EHTCoherencyDatum}
        # 2×2 coherency, column-major: M11,M21,M12,M22, each (re,im) → 8 cols.
        m = collect(reinterpret(reshape, Float64, res)')
        return (; kind = :coh, uv, m)
    elseif eltype(res) <: Complex
        return (; kind = :cpx, uv, m = collect(reinterpret(reshape, Float64, res)'))
    else
        return (; kind = :real, uv, m = collect(res))
    end
end

const _RE_COL = (:dodgerblue, 0.5)
const _IM_COL = (:crimson, 0.5)

const _COH_PANELS = (("PP", 1:2, (1, 1)), ("PQ", 5:6, (1, 2)), ("QP", 3:4, (2, 1)), ("QQ", 7:8, (2, 2)))

function _draw_coherency!(fig, row, comps)
    gl = fig[row, 1] = CairoMakie.GridLayout()
    for (name, cols, (r, c)) in _COH_PANELS
        χ2 = _avg(cp -> _redchi2(vec(cp.m[:, cols])), comps)
        ax = CairoMakie.Axis(
            gl[r, c]; ylabel = name, xlabel = (r == 2 ? "uv-distance (Gλ)" : ""),
            title = @sprintf("%s   χ²ᵣ = %.2f", name, χ2)
        )
        CairoMakie.hlines!(ax, [0.0]; color = (:gray, 0.7), linewidth = 1)
        for cp in comps
            CairoMakie.scatter!(ax, cp.uv, cp.m[:, first(cols)]; markersize = 4, color = _RE_COL)
            CairoMakie.scatter!(ax, cp.uv, cp.m[:, last(cols)]; markersize = 4, color = _IM_COL)
        end
    end
    return nothing
end

function _draw_scalar!(fig, row, comps, label)
    iscpx = first(comps).kind == :cpx
    χ2 = _avg(cp -> _redchi2(vec(cp.m)), comps)
    ax = CairoMakie.Axis(
        fig[row, 1]; xlabel = "uv-distance (Gλ)", ylabel = "Norm. Res. $label",
        title = @sprintf("χ²ᵣ = %.2f", χ2)
    )
    CairoMakie.hlines!(ax, [0.0]; color = (:gray, 0.7), linewidth = 1)
    for cp in comps
        if iscpx
            CairoMakie.scatter!(ax, cp.uv, cp.m[:, 1]; markersize = 4, color = _RE_COL)
            CairoMakie.scatter!(ax, cp.uv, cp.m[:, 2]; markersize = 4, color = _IM_COL)
        else
            CairoMakie.scatter!(ax, cp.uv, cp.m; markersize = 4, color = _RE_COL)
        end
    end
    return nothing
end

# Build the residual Figure for one or more parameter draws (overlaid).
function _residual_figure(post, params_list)
    ress = [residuals(post, p) for p in params_list]
    nprod = length(first(ress))
    perprod = [[_resid_components(r[i]) for r in ress] for i in 1:nprod]
    heights = [first(cs).kind === :coh ? 620 : 300 for cs in perprod]
    fig = CairoMakie.Figure(size = (820, sum(heights) + 60))
    for (i, comps) in enumerate(perprod)
        if first(comps).kind === :coh
            _draw_coherency!(fig, i, comps)
        else
            ST = split("$(Comrade.datumtype(first(ress)[i]))", "Datum{")[1] |> x -> split(x, ".EHT")[end]
            _draw_scalar!(fig, i, comps, ST)
        end
    end
    # Re/Im colour key.
    CairoMakie.Legend(
        fig[nprod + 1, 1],
        [CairoMakie.MarkerElement(color = _RE_COL, marker = :circle),
            CairoMakie.MarkerElement(color = _IM_COL, marker = :circle)],
        ["Real", "Imag"]; orientation = :horizontal, framevisible = false
    )
    return fig
end

function plot_residuals_png(path, post, x)
    CairoMakie.save(path, _residual_figure(post, [x]))
    return nothing
end

function plot_chain_residuals_png(path, post, post_samples)
    isempty(post_samples) && return nothing
    CairoMakie.save(path, _residual_figure(post, collect(post_samples)))
    return nothing
end

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
    mkpath(dirname(out))
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
