# Mean-image models. Each model defines `make_mean(model, grid, θ)` (the deterministic
# mean image given hyperparameters θ) and `genmeanprior(model)` (the priors over those
# hyperparameters). `centerfix(::Type)` declares whether an `ImagingModel` using this
# mean should re-center the image by default.

const fwhmfac = 2 * sqrt(2 * log(2))

centerfix(::Type{<:Any}) = true

# --- Fixed mean image ------------------------------------------------------------------
function make_mean(mimg::IntensityMap, grid, θ)
    return mimg
end

function genmeanprior(::IntensityMap)
    return NamedTuple()
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
    return (fb = VLBITruncated(VLBIExponential(0.1); lower = 0.0, upper = 1.0),)
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
    return (
        fwhm = VLBITruncated(VLBIGaussian(μas2rad(50.0), μas2rad(20.0)); lower = μas2rad(2.0), upper = μas2rad(100.0)),
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
    return (
        r0 = VLBIUniform(μas2rad(0.1), μas2rad(25.0)),
        ain = VLBIUniform(0.0, 10.0),
        aout = VLBIUniform(1.0, 10.0),
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
    return (
        r0 = VLBIUniform(μas2rad(10.0), μas2rad(25.0)),
        ain = VLBIExponential(5.0),
        aout = VLBIExponential(5.0),
        fb = VLBITruncated(VLBIExponential(0.1); lower = 0.0, upper = 1.0),
    )
end

# --- Lyapunov double ring --------------------------------------------------------------
# Photon-ring mean: an elliptical double-power-law n=0 ring plus an unresolved Gaussian
# n=1 ring whose FWHM is the n=0 ring FWHM demagnified by exp(-γ). Each ring is
# normalized to unit flux and mixed linearly, `(1 - f1) * m0 + f1 * m1`, so `f1` is
# exactly the n=1 flux fraction; its flat prior includes 0 so the posterior can express
# both "no n=1 ring" and "f1 unconstrained (posterior ≈ prior)". The n=1 radius is
# parameterized relative to n=0 (`r1 = r0 * (1 + δr)`), keeping the n=1 component
# anchored to the ring. See LyapunovRingJSUMean for a JohnsonSU n=0 variant with a
# radially skewed profile (sharp inner edge, extended outer falloff).
struct LyapunovRingMean end
centerfix(::Type{<:LyapunovRingMean}) = false

# Fractional FWHM of the n=0 double-power-law ring `r^a / (1 + r^(a+b+1))`, from the
# exact power-law-tail half-max crossings (inner tail r^a, outer tail r^-(b+1)).
# Closed form so it stays Reactant-traceable; overestimates by ~3-7% for typical slopes
# (up to ~15% for very shallow aout ≈ 1 tails).
function _dblpower_fwhm(a, b)
    s = a + b + 1
    rpk = (a / (b + 1))^(1 / s)
    h = rpk^a * (b + 1) / s / 2
    return h^(-1 / (b + 1)) - h^(1 / a)
end

function make_mean(::LyapunovRingMean, grid, θ)
    (; r0, ain, aout, τ0, ξτ0, δr, γ, f1, τ1, ξτ1, x1, y1) = θ
    r1 = r0 * (1 + δr)
    # n=1 width: the n=0 ring width demagnified by exp(-γ), converted to a Gaussian σ.
    σ1 = _dblpower_fwhm(ain, aout) * exp(-γ) / fwhmfac
    m0 = modify(
        RingTemplate(RadialDblPower(ain, aout), AzimuthalUniform()),
        Stretch(r0, r0 * (1 + τ0)), Rotate(ξτ0 / 2)
    )
    m1 = modify(
        RingTemplate(RadialGaussian(σ1), AzimuthalUniform()),
        Stretch(r1, r1 * (1 + τ1)), Rotate(ξτ1 / 2), Shift(x1, y1)
    )
    mimg = intensitymap(m0, grid)
    mimg1 = intensitymap(m1, grid)
    pmimg = baseimage(mimg)
    pmimg .= (1 - f1) .* pmimg ./ sum(pmimg) .+ f1 .* baseimage(mimg1) ./ sum(mimg1)
    return mimg
end

function genmeanprior(::LyapunovRingMean)
    return (
        r0 = VLBIUniform(μas2rad(5.0), μas2rad(30.0)),
        # Slope caps floor the n=0 fractional FWHM at ~0.5 r0 (10 μas at r0 = 20 μas):
        # a narrower n=0 ring would impersonate photon-ring sharpness the field cannot
        # produce (its correlation length is likewise floored), breaking identifiability.
        ain = VLBIUniform(0.0, 5.0),
        aout = VLBIUniform(1.0, 5.0),
        τ0 = VLBIExponential(0.25),
        ξτ0 = DiagonalVonMises(0.0, inv(π^2)),
        δr = VLBIGaussian(0.0, 0.1),
        γ = VLBIUniform(0.1, 1π),
        f1 = VLBIUniform(0.0, 0.5),
        τ1 = VLBIExponential(0.05),
        ξτ1 = DiagonalVonMises(0.0, inv(π^2)),
        x1 = VLBIUniform(-μas2rad(8.0), μas2rad(8.0)),
        y1 = VLBIUniform(-μas2rad(8.0), μas2rad(8.0)),
    )
end

# --- Lyapunov double ring, JohnsonSU n=0 variant ---------------------------------------
# Photon-ring mean: an elliptical JohnsonSU n=0 ring plus an unresolved Gaussian n=1
# ring whose width is the n=0 width demagnified by exp(-γ). The JohnsonSU skew α0 is
# physical — a lensed ring has a sharp inner edge and an extended outer falloff — but is
# bounded to ±3: beyond that the profile bulk shifts by sinh(α)·σ ≳ 10σ and the ring
# degenerates into a hard-edged disk. Each ring is normalized to unit flux and mixed
# linearly, `(1 - f1) * m0 + f1 * m1`, so `f1` is exactly the n=1 flux fraction; its flat
# prior includes 0 so the posterior can express "no n=1 ring" and "unconstrained" alike.
# The n=1 radius is parameterized relative to n=0 (`r1 = r0 * (1 + δr)`), keeping the
# n=1 component anchored to the ring. `w ≥ 0.2` floors the n=0 FWHM at ≈ 0.47 r0
# (~9 μas at r0 = 20 μas) so the mean cannot impersonate photon-ring sharpness that the
# field (correlation length likewise floored) cannot produce.
struct LyapunovRingJSUMean end
centerfix(::Type{<:LyapunovRingJSUMean}) = false

function make_mean(::LyapunovRingJSUMean, grid, θ)
    (; r0, w, α0, τ0, ξτ0, δr, γ, f1, τ1, ξτ1, x1, y1) = θ
    r1 = r0 * (1 + δr)
    m0 = modify(
        RingTemplate(RadialJohnsonSU(w, α0), AzimuthalUniform()),
        Stretch(r0, r0 * (1 + τ0)), Rotate(ξτ0 / 2)
    )
    m1 = modify(
        RingTemplate(RadialGaussian(w * exp(-γ)), AzimuthalUniform()),
        Stretch(r1, r1 * (1 + τ1)), Rotate(ξτ1 / 2), Shift(x1, y1)
    )
    mimg = intensitymap(m0, grid)
    mimg1 = intensitymap(m1, grid)
    pmimg = baseimage(mimg)
    pmimg .= (1 - f1) .* pmimg ./ sum(pmimg) .+ f1 .* baseimage(mimg1) ./ sum(mimg1)
    return mimg
end

function genmeanprior(::LyapunovRingJSUMean)
    return (
        r0 = VLBIUniform(μas2rad(5.0), μas2rad(30.0)),
        w = VLBIUniform(0.2, 1.0),
        α0 = VLBIUniform(-3.0, 3.0),
        τ0 = VLBIExponential(0.25),
        ξτ0 = DiagonalVonMises(0.0, inv(π^2)),
        δr = VLBIGaussian(0.0, 0.1),
        γ = VLBIUniform(0.1, 1π),
        f1 = VLBIUniform(0.0, 0.5),
        τ1 = VLBIExponential(0.05),
        ξτ1 = DiagonalVonMises(0.0, inv(π^2)),
        x1 = VLBIUniform(-μas2rad(8.0), μas2rad(8.0)),
        y1 = VLBIUniform(-μas2rad(8.0), μas2rad(8.0)),
    )
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
    return (
        fwhm = VLBITruncated(VLBIGaussian(μas2rad(50.0), μas2rad(20.0)); lower = μas2rad(10.0), upper = μas2rad(100.0)),
        s = VLBIUniform(1.0, 10.0),
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
    return (
        r = VLBIUniform(dx * 4, min(fovx, fovy) / 3),
        τ = VLBIUniform(0.0, 10.0),
        ξτ = DiagonalVonMises(0.0, inv(π^2)),
        x = VLBIUniform(-fovx / 4 - x0, fovx / 4 - x0),
        y = VLBIUniform(-fovy / 4 - y0, fovy / 4 - y0),
        fj = VLBITruncated(VLBIExponential(0.1); lower = 0.0, upper = 1.0),
    )
end

# --- Gaussian blended with a disk background ------------------------------------------
# `fwhm0` (radians) is the nominal core size: the FWHM prior is centered on it and its
# upper bound scales with it. `beam` (radians) is the observation resolution and sets the
# lower bound alone, so rescaling `fwhm0` never lets the core shrink below what the data
# can resolve.
struct GaussBkgdMean{M, T}
    bkgd::M
    fwhm0::T
    beam::T
    function GaussBkgdMean(grid::RectiGrid, fwhm0, beam)
        x0, y0 = phasecenter(grid)
        fovx, fovy = fieldofview(grid)
        pa = posang(grid)
        bkgd = intensitymap(modify(VLBISkyModels.GaussDisk(0.3), Stretch(fovx / 2, fovy / 2), Shift(-x0, -y0), Rotate(pa)), grid)
        f0, b = promote(fwhm0, beam)
        return new{typeof(bkgd), typeof(f0)}(bkgd ./ Comrade.flux(bkgd), f0, b)
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

function genmeanprior(p::GaussBkgdMean)
    f0 = p.fwhm0
    return (
        fwhm = VLBITruncated(VLBIGaussian(f0, 0.4 * f0); lower = 0.25 * p.beam, upper = 3.0 * f0),
        fb = VLBITruncated(VLBIExponential(0.25); lower = 0.0, upper = 1.0),
    )
end
