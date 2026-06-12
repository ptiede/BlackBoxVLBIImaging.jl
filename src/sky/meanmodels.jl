# Mean-image models. Each model defines `make_mean(model, grid, θ)` (the deterministic
# mean image given hyperparameters θ) and `genmeanprior(model)` (the priors over those
# hyperparameters). `centerfix(::Type)` declares whether an `ImagingModel` using this
# mean should re-center the image by default.
#
# Each `register_mean_model!` call below binds the TOML name (`mean.type` in the image
# TOML) to a builder `(meancfg, grid, beam) -> model` that constructs the mean from the
# parsed `[mean]` table, the image grid, and the observation beam.

const fwhmfac = 2 * sqrt(2 * log(2))

# Mean Gaussian FWHM in radians. Data-driven by default: `fwhm_beams` × the observation beam
# (default 1× beam, matching MixedPolPaper's `Stretch(beamsize(dcoh)/fwhmfac)`). An explicit
# `fwhm` (μas) overrides. `default_μas` is only used when the sky is built without a beam
# (standalone/no data).
function _mean_fwhm_rad(meancfg::AbstractDict, beam, default_μas)
    if haskey(meancfg, "fwhm")
        return μas2rad(Float64(meancfg["fwhm"]))
    elseif !isnothing(beam)
        return Float64(get(meancfg, "fwhm_beams", 1.0)) * beam
    else
        return μas2rad(default_μas)
    end
end

# --- Fixed mean image ------------------------------------------------------------------
function make_mean(mimg::IntensityMap, grid, θ)
    return mimg
end

function genmeanprior(::IntensityMap)
    return Dict()
end

# --- Fixed mean image blended with a disk background -----------------------------------
struct MimgPlusBkg{M}
    mimg::M
    bkgd::M
    function MimgPlusBkg(mimg::IntensityMap)
        grid = axisdims(mimg)
        x0, y0 = phasecenter(grid)
        fovx, fovy = fieldofview(grid)
        pa = posang(grid)
        bkgd = intensitymap(modify(VLBISkyModels.GaussDisk(0.3), Stretch(fovx / 2, fovy / 2), Shift(-x0, -y0), Rotate(pa)), grid)
        return new{typeof(mimg)}(mimg ./ Comrade.flux(mimg), bkgd ./ Comrade.flux(bkgd))
    end
end

function make_mean(mimg::MimgPlusBkg, grid, θ)
    (; fb) = θ
    return mimg.mimg .* ((1 - fb)) .+ fb .* mimg.bkgd
end

function genmeanprior(::MimgPlusBkg)
    # `lower = 0.0` must be explicit: VLBITruncated's flat transform is built from the
    # truncation bounds only, so a one-sided `upper` maps ℝ → (-∞, 1) and lets the
    # optimizer/sampler walk into fb < 0 (negative background flux) where logpdf = -Inf.
    return Dict(:fb => VLBITruncated(VLBIExponential(0.1); lower = 0.0, upper = 1.0))
end

register_mean_model!("Bkgd") do meancfg, g, beam
    fwhm = _mean_fwhm_rad(meancfg, beam, 50.0)
    @info "Using a background mean for the sky model (Gaussian FWHM = $(round(rad2μas(fwhm), digits = 1)) μas)"
    mimg = intensitymap(modify(Gaussian(), Stretch(fwhm / fwhmfac)), g)
    return MimgPlusBkg(mimg ./ sum(mimg))
end

# --- Gaussian mean ---------------------------------------------------------------------
struct GaussMean end
centerfix(::Type{<:GaussMean}) = true

function make_mean(::GaussMean, grid, θ)
    (; fwhm) = θ
    m = modify(Gaussian(), Stretch(fwhm / fwhmfac))
    mimg = intensitymap(m, grid)
    pmimg = baseimage(mimg)
    pmimg ./= sum(pmimg)
    return mimg
end

function genmeanprior(::GaussMean)
    return Dict(
        :fwhm => VLBITruncated(VLBIGaussian(μas2rad(50.0), μas2rad(20.0)); lower = μas2rad(2.0), upper = μas2rad(100.0)),
    )
end

register_mean_model!("Gauss") do meancfg, g, beam
    @info "Using a Gaussian mean for the sky model"
    return GaussMean()
end

# --- Double power-law ring mean --------------------------------------------------------
struct DblRingMean end
centerfix(::Type{<:DblRingMean}) = false

function make_mean(::DblRingMean, grid, θ)
    (; r0, ain, aout) = θ
    m = modify(RingTemplate(RadialDblPower(ain, aout), AzimuthalUniform()), Stretch(r0))
    mimg = intensitymap(m, grid)
    pmimg = baseimage(mimg)
    pmimg .= pmimg ./ sum(pmimg)
    return mimg
end

function genmeanprior(::DblRingMean)
    return Dict(
        :r0 => VLBIUniform(μas2rad(0.1), μas2rad(25.0)),
        :ain => VLBIUniform(0.0, 10.0),
        :aout => VLBIUniform(1.0, 10.0)
    )
end

register_mean_model!("Ring") do meancfg, g, beam
    @info "Using a ring mean for the sky model"
    return DblRingMean()
end

# --- Double power-law ring with background --------------------------------------------
struct DblRingWBkgd end
centerfix(::Type{<:DblRingWBkgd}) = false

function make_mean(::DblRingWBkgd, grid, θ)
    (; r0, ain, aout, fb) = θ
    m = modify(RingTemplate(RadialDblPower(ain, 1 + aout), AzimuthalUniform()), Stretch(r0))
    mimg = intensitymap(m, grid)
    pmimg = baseimage(mimg)
    fbn = fb / (prod(size(grid)))
    pmimg .= pmimg ./ sum(pmimg) * ((1 - fb)) .+ fbn
    return mimg
end

function genmeanprior(::DblRingWBkgd)
    return Dict(
        :r0 => VLBIUniform(μas2rad(10.0), μas2rad(25.0)),
        :ain => VLBIExponential(5.0),
        :aout => VLBIExponential(5.0),
        :fb => VLBITruncated(VLBIExponential(0.1); lower = 0.0, upper = 1.0)
    )
end

register_mean_model!("RingBkgd") do meancfg, g, beam
    @info "Using a ring+background mean for the sky model"
    return DblRingWBkgd()
end

# --- Student-t blob mean ---------------------------------------------------------------
struct TBlobMean end
centerfix(::Type{<:TBlobMean}) = true

function make_mean(::TBlobMean, grid, θ)
    (; fwhm, s) = θ
    m = modify(TBlob(s), Stretch(fwhm / fwhmfac))
    mimg = intensitymap(m, grid)
    pmimg = baseimage(mimg)
    pmimg ./= sum(pmimg)
    return mimg
end

function genmeanprior(::TBlobMean)
    return Dict(
        :fwhm => VLBITruncated(VLBIGaussian(μas2rad(50.0), μas2rad(20.0)); lower = μas2rad(10.0), upper = μas2rad(100.0)),
        :s => VLBIUniform(1.0, 10.0)
    )
end

register_mean_model!("TBlob") do meancfg, g, beam
    @info "Using a Student-t blob mean for the sky model"
    return TBlobMean()
end

# --- Core image plus a jet Gaussian ----------------------------------------------------
struct JetGauss{M}
    core::M
end
centerfix(::Type{<:JetGauss}) = true

function make_mean(mimg::JetGauss, grid, θ)
    (; r, τ, ξτ, x, y, fj) = θ
    img = intensitymap(modify(Gaussian(), Stretch(r, r * (1 + τ)), Rotate(ξτ / 2), Shift(x, y)), grid)
    fl = sum(img)
    pimg = baseimage(img)
    pcore = baseimage(mimg.core)
    pimg .= pcore .* (1 - fj) .+ pimg .* (fj / fl)
    return img
end

function genmeanprior(m::JetGauss)
    fovx, fovy = fieldofview(m.core)
    x0, y0 = phasecenter(m.core)
    dx, dy = pixelsizes(m.core)
    return Dict(
        :r => VLBIUniform(dx * 4, min(fovx, fovy) / 3),
        :τ => VLBIUniform(0.0, 10.0),
        :ξτ => DiagonalVonMises(0.0, inv(π^2)),
        :x => VLBIUniform(-fovx / 4 - x0, fovx / 4 - x0),
        :y => VLBIUniform(-fovy / 4 - y0, fovy / 4 - y0),
        :fj => VLBITruncated(VLBIExponential(0.1); lower = 0.0, upper = 1.0)
    )
end

register_mean_model!("JetGauss") do meancfg, g, beam
    fwhm = _mean_fwhm_rad(meancfg, beam, 30.0)
    @info "Using a jet+Gaussian mean for the sky model (Gaussian FWHM = $(round(rad2μas(fwhm), digits = 1)) μas)"
    mimg = intensitymap(modify(Gaussian(), Stretch(fwhm / fwhmfac)), g)
    return JetGauss(mimg ./ sum(mimg))
end

# --- Gaussian blended with a disk background ------------------------------------------
struct GaussBkgdMean{M}
    bkgd::M
    function GaussBkgdMean(grid::RectiGrid)
        x0, y0 = phasecenter(grid)
        fovx, fovy = fieldofview(grid)
        pa = posang(grid)
        bkgd = intensitymap(modify(VLBISkyModels.GaussDisk(0.3), Stretch(fovx / 2, fovy / 2), Shift(-x0, -y0), Rotate(pa)), grid)
        return new{typeof(bkgd)}(bkgd ./ Comrade.flux(bkgd))
    end
end
centerfix(::Type{<:GaussBkgdMean}) = true

function make_mean(p::GaussBkgdMean, grid, θ)
    (; fwhm, fb) = θ
    m = modify(Gaussian(), Stretch(fwhm / fwhmfac))
    mimg = intensitymap(m, grid)
    pmimg = baseimage(mimg)
    pf = sum(pmimg)
    pmimg .= pmimg ./ pf * (1 - fb) .+ p.bkgd ./ sum(p.bkgd) * fb
    return mimg
end

function genmeanprior(::GaussBkgdMean)
    return Dict(
        :fwhm => VLBITruncated(VLBIGaussian(μas2rad(50.0), μas2rad(20.0)); lower = μas2rad(20.0), upper = μas2rad(100.0)),
        :fb => VLBIUniform(0.0, 1.0)
    )
end

register_mean_model!("GaussBkgd") do meancfg, g, beam
    @info "Using a Gaussian background mean for the sky model"
    return GaussBkgdMean(g)
end
