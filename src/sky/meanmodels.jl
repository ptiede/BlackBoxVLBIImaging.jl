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
# A `TBlob(ν)` core, I ∝ (1 + r²/ν)^(-(ν+2)/2), whose tail falls as a power law. The tail is
# sampled as `logη = -log ν`, so logη → -∞ is a Gaussian, and `fwhm` is the core's actual FWHM
# at every ν. `fwhm0` and `beam` (radians) set the FWHM prior as in `GaussBkgdMean`.
struct TBlobMean{T}
    fwhm0::T
    beam::T
    TBlobMean(fwhm0, beam) = (p = promote(fwhm0, beam); new{eltype(p)}(p...))
end
centerfix(::Type{<:TBlobMean}) = true

# Half-maximum radius of the unit-scale `TBlob(ν)`.
_tblob_halfwidth(ν) = sqrt(ν * (2^(2 / (ν + 2)) - 1))

# ν from 0.1 to 100; `TBlob`'s normalization overflows past ν ≈ 340.
_tblob_tail_prior() = VLBIUniform(log(0.01), log(10.0))

function _tblob_core(fwhm, logη)
    ν = exp(-logη)
    return modify(TBlob(ν), Stretch(fwhm / (2 * _tblob_halfwidth(ν))))
end

function make_mean(::TBlobMean, grid, θ)
    mimg = intensitymap(_tblob_core(θ.fwhm, θ.logη), grid)
    pmimg = baseimage(mimg)
    pmimg ./= sum(pmimg)
    return mimg
end

genmeanprior(p::TBlobMean) = (fwhm = _core_fwhm_prior(p.fwhm0, p.beam), logη = _tblob_tail_prior())

# --- Anisotropic power law -------------------------------------------------------------
# VLBISkyModels' `AnisotropicPowerLaw` with core radius `rcore` (radians), its n = 1, 2 slope
# modes (α1, β1, α2, β2) and the taper written as `taper r²/r_t²`, with r_t a quarter of the
# smaller field of view, normalized to unit flux. log m is linear in every parameter, so its
# curvature along them does not change across the posterior.
struct AnisotropicPowerLawMean{T}
    rcore::T
    rtaper::T
end
AnisotropicPowerLawMean(grid::RectiGrid, rcore) = AnisotropicPowerLawMean(promote(rcore, minimum(fieldofview(grid)) / 4)...)

function make_mean(p::AnisotropicPowerLawMean, grid, θ)
    (; slope, α1, β1, α2, β2, taper) = θ
    apl = AnisotropicPowerLaw(slope, (α1, α2), (β1, β2), taper * (p.rcore / p.rtaper)^2)
    mimg = intensitymap(modify(apl, Stretch(p.rcore)), grid)
    pmimg = baseimage(mimg)
    pmimg ./= sum(pmimg)
    return mimg
end

function genmeanprior(::AnisotropicPowerLawMean)
    harmonic = VLBIGaussian(0.0, 0.4)
    return (
        slope = VLBIGaussian(1.0, 0.5), α1 = harmonic, β1 = harmonic, α2 = harmonic, β2 = harmonic,
        taper = VLBIGaussian(1.0, 0.5),
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
        bkgd = _disk_background(grid)
        f0, b = promote(fwhm0, beam)
        return new{typeof(bkgd), typeof(f0)}(bkgd, f0, b)
    end
end
centerfix(::Type{<:GaussBkgdMean}) = true

# The unit-flux disk background of the Gaussian background means, filling the field of view.
function _disk_background(grid::RectiGrid)
    x0, y0 = phasecenter(grid)
    fovx, fovy = fieldofview(grid)
    pa = posang(grid)
    bkgd = intensitymap(modify(VLBISkyModels.GaussDisk(0.3), Stretch(fovx / 2, fovy / 2), Shift(-x0, -y0), Rotate(pa)), grid)
    return bkgd ./ Comrade.flux(bkgd)
end

function _gauss_bkgd_image(bkgd, grid, fwhm, fb)
    mimg = intensitymap(modify(Gaussian(), Stretch(fwhm / fwhmfac)), grid)
    pmimg = baseimage(mimg)
    pf = sum(pmimg)
    pmimg .= pmimg ./ pf * (1 - fb) .+ bkgd ./ sum(bkgd) * fb
    return mimg
end

make_mean(p::GaussBkgdMean, grid, θ) = _gauss_bkgd_image(p.bkgd, grid, θ.fwhm, θ.fb)

_bkgd_fraction_prior() = VLBITruncated(VLBIExponential(0.25); lower = 0.0, upper = 1.0)

_core_fwhm_prior(fwhm0, beam) = VLBITruncated(VLBIGaussian(fwhm0, 0.4 * fwhm0); lower = 0.25 * beam, upper = 3.0 * fwhm0)

genmeanprior(p::GaussBkgdMean) = (fwhm = _core_fwhm_prior(p.fwhm0, p.beam), fb = _bkgd_fraction_prior())

# --- Gaussian of fixed FWHM blended with a disk background ------------------------------
# `GaussBkgdMean` with the core FWHM held at `fwhm` (radians); only the background fraction
# is sampled, with the same prior.
struct FixedGaussBkgdMean{M, T}
    bkgd::M
    fwhm::T
    FixedGaussBkgdMean(grid::RectiGrid, fwhm) = (b = _disk_background(grid); new{typeof(b), typeof(fwhm)}(b, fwhm))
end
centerfix(::Type{<:FixedGaussBkgdMean}) = true

make_mean(p::FixedGaussBkgdMean, grid, θ) = _gauss_bkgd_image(p.bkgd, grid, p.fwhm, θ.fb)

genmeanprior(::FixedGaussBkgdMean) = (fb = _bkgd_fraction_prior(),)
