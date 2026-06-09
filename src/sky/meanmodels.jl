# Mean-image models. Each model defines `make_mean(model, grid, θ)` (the deterministic
# mean image given hyperparameters θ) and `genmeanprior(model)` (the priors over those
# hyperparameters). `centerfix(::Type)` declares whether an `ImagingModel` using this
# mean should re-center the image by default.

const fwhmfac = 2 * sqrt(2 * log(2))

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
    return Dict(:fb => VLBITruncated(VLBIExponential(0.1); upper = 1.0))
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
        :aout => VLBIUniform(0.0, 10.0)
    )
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
        :ain => VLBIExponential(3.0),
        :aout => VLBIExponential(3.0),
        :fb => VLBITruncated(Exponential(0.1); upper = 1.0)
    )
end

# --- Student-t blob mean ---------------------------------------------------------------
struct TBlobMean end
centerfix(::Type{<:TBlobMean}) = true

function make_mean(::TBlobMean, grid, θ)
    (; fwhm, s) = θ
    m = modify(TBlobNN(s), Stretch(fwhm / fwhmfac))
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
        :fj => VLBITruncated(Exponential(0.1); upper = 1.0)
    )
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
