# The sky models, written with Comrade's `@sky` macro: one definition per
# (polarization representation × random-field base) combination, so each model's priors
# (the `~` lines) sit directly next to the code that consumes them. The config layer
# (`build_sky_config`) picks the constructor from the polrep and the `order`-derived base
# marker (see `_order_to_base` / `sky_constructor`).
#
# Shared structure of every definition:
#   - image-fluctuation priors are polrep/base specific and flat (`c`, `σ`, `σa`, ...),
#   - `mean ~ genmeanprior(meanmodel)` is the mean-model group (empty for a fixed image),
#   - `flux ~ _flux_prior(ftot)` samples the total flux only when `ftot` is a distribution,
#   - `gauss ~ gaussprior` is the optional extra-Gaussian group (empty NamedTuple = off),
#   - `center` is `Val(true/false)` so re-centering stays a compile-time choice, and
#     `center_power` is the intensity power the centroid it re-centers on is weighted by.
#
# NOTE: the definitions carry plain comments instead of docstrings — the doc system cannot
# attach a docstring to the multi-definition block `@sky`/`@instrument` expand to.

# --- total-flux handling -----------------------------------------------------------------
# `ftot` is either a fixed `Real` or a distribution; only the latter contributes a sampled
# parameter (under the `flux` group).
_flux_prior(::Real) = NamedTuple()
_flux_prior(d) = (ftot = d,)
@inline _get_ftot(f::Real, ::NamedTuple{()}) = f
@inline _get_ftot(_, flux::NamedTuple) = flux.ftot

# --- optional extra Gaussian -------------------------------------------------------------
function gengaussprior(::PolModel)
    return (
        fg = VLBIUniform(0.0, 1.0),
        σg = VLBIUniform(μas2rad(250.0), μas2rad(1000.0)),
        τg = VLBIUniform(0.0, 7.0),
        ξg = DiagonalVonMises(0.0, inv(1π^2)),
        xg = VLBIUniform(-μas2rad(10_000.0), μas2rad(10_000)),
        yg = VLBIUniform(-μas2rad(10_000.0), μas2rad(10_000.0)),
        pg = VLBIUniform(0.0, 1.0),
        pxg = VLBITruncated(VLBIGaussian(0.0, 1.0); lower = 0.0),
        pyg = VLBITruncated(VLBIGaussian(0.0, 1.0); lower = 0.0),
        pzg = VLBITruncated(VLBIGaussian(0.0, 1.0); lower = 0.0),
    )
end

function gengaussprior(::TotalIntensity)
    return (
        fg = VLBIUniform(0.0, 1.0),
        σg = VLBIUniform(μas2rad(250.0), μas2rad(1000.0)),
        τg = VLBIUniform(0.0, 7.0),
        ξg = DiagonalVonMises(0.0, inv(1π^2)),
        xg = VLBIUniform(-μas2rad(10_000.0), μas2rad(10_000)),
        yg = VLBIUniform(-μas2rad(10_000.0), μas2rad(10_000.0)),
    )
end

# The fraction of the total flux left for the image once the Gaussian takes its share.
@inline _img_flux(f, ::NamedTuple{()}) = f
@inline _img_flux(f, gauss::NamedTuple) = f * (1 - gauss.fg)

@inline _add_gauss(ms, ftot, ::NamedTuple{()}) = ms

@inline function _add_gauss(ms, ftot, θ::NamedTuple{(:fg, :σg, :τg, :ξg, :xg, :yg)})
    (; fg, σg, τg, ξg, xg, yg) = θ
    g = modify(Gaussian(), Stretch(σg, σg * (1 + τg)), Rotate(ξg / 2), Shift(xg, yg), Renormalize(ftot * fg))
    return ms + g
end

@inline function _add_gauss(ms, ftot, θ::NamedTuple{(:fg, :σg, :τg, :ξg, :xg, :yg, :pg, :pxg, :pyg, :pzg)})
    (; fg, σg, τg, ξg, xg, yg, pg, pxg, pyg, pzg) = θ
    g = modify(Gaussian(), Stretch(σg, σg * (1 + τg)), Rotate(ξg / 2), Shift(xg, yg), Renormalize(ftot * fg))
    pr = sqrt(pxg^2 + pyg^2 + pzg^2) + 1.0e-6
    polg = PolarizedModel(g, (pg * pxg / pr) * g, (pg * pyg / pr) * g, (pg * pzg / pr) * g)
    return ms + polg
end

# --- centering ---------------------------------------------------------------------------
"""
    power_centroid(img, p)

The `Iᵖ`-weighted centroid of an intensity map, in the coordinates of its grid. `p = 1` is the
center of light, [`centroid`](@ref); a larger `p` weights the bright pixels more heavily, so
the result follows the compact structure and is nearly blind to faint extended flux. Polarized
maps use Stokes `I`.
"""
function power_centroid(img, p)
    isone(p) && return centroid(img)
    # The Iᵖ centroid is the center of light of the raised map, so it inherits whichever
    # `centroid` the array type brings with it (CPU, GPU, Reactant).
    return centroid(IntensityMap(baseimage(img) .^ p, axisdims(img)))
end

power_centroid(img::IntensityMap{<:StokesParams}, p) = power_centroid(stokes(img, :I), p)

@inline _center_model(pmap, ::Val{false}, pulse, power) = ContinuousImage(pmap, pulse)
@inline function _center_model(pmap, ::Val{true}, pulse, power)
    x0, y0 = power_centroid(pmap, power)
    return shifted(ContinuousImage(pmap, pulse), -x0, -y0)
end

# --- random-field plan preparation --------------------------------------------------------
# Maps the base *marker* (polreps.jl) to the concrete transform/plan object handed to the
# `@sky` constructors as their `base` keyword. `GMRF` (order 1) needs no plan — the prior
# carries it — so it has no method here.
prepare_base(::NonCenteredMRF, grid, order) = standardize(MarkovRandomFieldGraph(grid; order); flag = Comrade.VLBISkyModels.FFTW.EXHAUSTIVE)
prepare_base(ps::Union{Matern, MarkovRF}, grid, order) = SRF(ps, StationaryRandomFieldPlan(grid))

markov_order(::MarkovRF{N}) where {N} = N

"""
    MaternSlopePS(ℓ, α)

The Matérn power spectrum `S(k) ∝ (1 + ℓ² k²)^(-α/2)` written by its outer scale `ℓ` (in
pixels) and its high-`k` slope `α`: `MaternPS(ρ, ν)` with `α = 2(ν + 1)` and
`ℓ = ρ / √(8ν)`. `α = 2` gives equal power per log `k` below the outer scale, the limit
`ν → 0` that the `(ρ, ν)` form reaches only at its edge.
"""
struct MaternSlopePS{T} <: VLBIImagePriors.AbstractPowerSpectrum
    ℓ::T
    α::T
end

@inline function VLBIImagePriors.ampspectrum(ps::MaternSlopePS, ks)
    kx, ky = ks
    return (1 + ps.ℓ^2 * (kx^2 + ky^2))^(-ps.α / 4)
end

# The power spectrum of a field from its spectral parameters: the `N` correlation lengths of
# an order-`N` Markov RF, or the outer scale and slope `(ℓ, α)` of a Matérn field.
field_spectrum(::MarkovRF, ρs) = MarkovPS(ρs)
field_spectrum(::Matern, ρs) = MaternSlopePS(ρs[1], ρs[2])

# --- Markov-field correlation-length priors -----------------------------------------------
# A `MarkovRF` field of order `N` has amplitude spectrum `1/sqrt(1 + Σₙ (ρₙ² k²)ⁿ)`, so each
# term `n = 1 … N` carries its own correlation length `ρₙ` in pixels. The markers below pick
# the prior family; `markov_rho_prior` turns one into the `NTuple` of per-term priors the
# `@sky` bodies sample. `build_sky_config` selects the marker from `[model] rho_prior`.

"""Correlation-length prior that is flat in `ρ` across the grid."""
struct UniformRhoPrior end

"""
Log-normal correlation-length prior: the unconstrained (flat) coordinate is `log ρ`, so the
sampler sees a Gaussian.
"""
struct LogNormalRhoPrior end

# `LogNormal(log(med), logsd)` truncated to `[lower, upper]` pixels, built as the `exp`
# pushforward of a truncated `VLBIGaussian` rather than from `Distributions.LogNormal`, so the
# density and the transform are branchless and trace under Reactant.
function _lognormal(med, logsd, lower, upper)
    lower < med < upper || error(
        "the log-normal prior median $med is not inside [$lower, $upper]"
    )
    return PT.PushforwardDistribution(
        exp, VLBITruncated(VLBIGaussian(log(med), logsd); lower = log(lower), upper = log(upper))
    )
end

"""
    markov_rho_prior(kind, grid, beamsize, order::Int; kwargs...) -> NTuple{order}

The per-term correlation-length priors of an order-`order` Markov random field on `grid`, in
pixels. `kind` is the family marker:

  - [`UniformRhoPrior`](@ref): every term is uniform on `[lower, max(size(grid)...)]`.
  - [`LogNormalRhoPrior`](@ref): term 1 is log-normal with median `median_first` — half the
    larger grid dimension, the largest structure the field can carry — and log-sd
    `logsd_first`; terms `n ≥ 2` are log-normal with median `median_rest`, the data beam
    `beamsize` in pixels, and the tighter log-sd `logsd_rest`. The unconstrained coordinate
    of every term is `log ρ`. Every term is truncated to `[lower, upper]` pixels (default 1
    pixel to the larger grid dimension): a correlation length under a pixel is not resolved
    by the grid and makes the white coefficients of the data-constrained modes stiff, and one
    longer than the grid leaves the field nearly constant across it.
"""
function markov_rho_prior(::UniformRhoPrior, grid, beamsize, order::Int; lower = 0.1)
    return ntuple(Returns(VLBIUniform(lower, 1.0 * max(size(grid)...))), order)
end

function markov_rho_prior(
        ::LogNormalRhoPrior, grid, beamsize, order::Int;
        median_first = max(size(grid)...) / 2, logsd_first = 1.0,
        median_rest = beamsize / step(grid.X), logsd_rest = 0.7, lower = 1.0,
        upper = 1.0 * max(size(grid)...)
    )
    return ntuple(order) do n
        return n == 1 ? _lognormal(median_first, logsd_first, lower, upper) :
            _lognormal(median_rest, logsd_rest, lower, upper)
    end
end

"""
    spectrum_prior(base, kind, grid, beamsize) -> Tuple

The priors of a field's spectral parameters. For `MarkovRF{N}`, the `N` correlation lengths of
[`markov_rho_prior`](@ref). For `Matern`, the outer scale `ℓ` and slope `α` of
[`MaternSlopePS`](@ref): with [`LogNormalRhoPrior`](@ref), `ℓ` takes the first Markov term's
prior and `α` is log-normal with median 2.5 and log-sd 0.4 on `[1, 8]`; with
[`UniformRhoPrior`](@ref), `ℓ` is uniform on `[0.1, max(size(grid)...)]` pixels and `α`
uniform on `[1, 8]`.
"""
spectrum_prior(::MarkovRF{N}, kind, grid, beamsize) where {N} = markov_rho_prior(kind, grid, beamsize, N)

function spectrum_prior(::Matern, kind::LogNormalRhoPrior, grid, beamsize)
    return (first(markov_rho_prior(kind, grid, beamsize, 1)), _lognormal(2.5, 0.4, 1.0, 8.0))
end

function spectrum_prior(::Matern, kind::UniformRhoPrior, grid, beamsize)
    return (first(markov_rho_prior(kind, grid, beamsize, 1)), VLBIUniform(1.0, 8.0))
end

# --- shared image builders (unchanged math) ------------------------------------------------
@inline function make_stokesi(ftot, mimg, δ)
    stokesi = apply_fluctuations(CenteredLR(), mimg, δ)
    pstokesi = baseimage(stokesi)
    pstokesi .*= ftot
    return stokesi
end

function make_poincare(ftot, mimg, δ, p0, pσ, pδ, angparams)
    stokesi = apply_fluctuations(CenteredLR(), mimg, δ)
    pstokesi = parent(stokesi)
    pstokesi .*= ftot
    ptotim = logistic.(p0 .+ pσ .* pδ)
    pmap = PoincareSphere2Map(stokesi, ptotim, angparams)
    return pmap
end

function make_pol2expimage(ftot, a, b, c, d, mimg)
    # this allocates a whole new map so we can do things in place after
    pmap = VLBISkyModels.PolExp2Map(a, b, c, d, axisdims(mimg))
    bpmap = baseimage(pmap)
    bpmapI = stokes(bpmap, :I)
    bmimg = baseimage(mimg)
    bpmapI .*= bmimg
    ft = sum(bpmapI)
    bpmapI .*= ftot ./ ft
    map((:Q, :U, :V)) do s
        bpmapS = stokes(bpmap, s)
        bpmapS .*= bmimg
        bpmapS .*= ftot ./ ft
    end
    return pmap
end

# =========================================================================================
# Total intensity (Stokes I)
# =========================================================================================

# Stokes-I imaging with a first-order GMRF fluctuation field (`order == 1`).
@sky function stokesi_gmrf(grid; meanmodel, ftot, beamsize, order = 1, gaussprior = NamedTuple(), center = Val(true), center_power = 1, pulse = DeltaPulse())
    c ~ corr_image_prior(grid, beamsize; base = GMRF, order = order, lower = 4.0)
    σ ~ VLBIExponential(1.0)
    mean ~ genmeanprior(meanmodel)
    flux ~ _flux_prior(ftot)
    gauss ~ gaussprior
    mimg = make_mean(meanmodel, grid, mean)
    f = _get_ftot(ftot, flux)
    pmap = make_stokesi(_img_flux(f, gauss), mimg, σ .* c.params)
    ms = _center_model(pmap, center, pulse, center_power)
    return _add_gauss(ms, f, gauss)
end

# Stokes-I imaging with a non-centered Markov transform of the GMRF (`order > 1`).
@sky function stokesi_ncmrf(grid; base, meanmodel, ftot, beamsize, gaussprior = NamedTuple(), center = Val(true), center_power = 1, pulse = DeltaPulse())
    c ~ (
        hyperparams = VLBITruncated(
            VLBIInverseGamma(1.0, -log(0.01) * beamsize / pixelsizes(grid).X);
            lower = 1.0, upper = 2 * max(size(grid)...)
        ),
        params = VLBIImagePriors.StdNormal(size(grid)),
    )
    σ ~ VLBIExponential(1.0)
    mean ~ genmeanprior(meanmodel)
    flux ~ _flux_prior(ftot)
    gauss ~ gaussprior
    mimg = make_mean(meanmodel, grid, mean)
    f = _get_ftot(ftot, flux)
    δ = centerdist(base, c.hyperparams, c.params)
    δ .*= σ
    img = IntensityMap(δ, axisdims(mimg))
    apply_fluctuations!(CenteredLR(), img, mimg, δ)
    bimg = baseimage(img)
    bimg .*= _img_flux(f, gauss)
    ms = _center_model(img, center, pulse, center_power)
    return _add_gauss(ms, f, gauss)
end

# Stokes-I imaging with a stationary random field: an order-`N` Markov power spectrum
# (`order < 0`) or a Matérn one (`order == 0`); `base.ps` selects it.
@sky function stokesi_srf(grid; base, meanmodel, ftot, beamsize, rhoprior = UniformRhoPrior(), gaussprior = NamedTuple(), center = Val(true), center_power = 1, pulse = DeltaPulse())
    c ~ VLBIImagePriors.std_dist(base.plan)
    σ ~ VLBIExponential(1.0)
    ρs ~ spectrum_prior(base.ps, rhoprior, grid, beamsize)
    mean ~ genmeanprior(meanmodel)
    flux ~ _flux_prior(ftot)
    gauss ~ gaussprior
    mimg = make_mean(meanmodel, grid, mean)
    f = _get_ftot(ftot, flux)
    δ = genfield(StationaryRandomField(field_spectrum(base.ps, ρs), base.plan), c)
    δ .*= σ
    pmap = make_stokesi(_img_flux(f, gauss), mimg, δ)
    ms = _center_model(pmap, center, pulse, center_power)
    return _add_gauss(ms, f, gauss)
end

# =========================================================================================
# Poincaré-sphere polarized imaging
# =========================================================================================

# Poincaré-sphere polarized imaging with a first-order GMRF field (`order == 1`).
@sky function poincare_gmrf(grid; meanmodel, ftot, beamsize, order = 1, gaussprior = NamedTuple(), center = Val(true), center_power = 1, pulse = DeltaPulse())
    c ~ corr_image_prior(grid, beamsize; base = GMRF, order = order, lower = 4.0)
    σ ~ VLBIExponential(1.0)
    p ~ corr_image_prior(grid, beamsize; base = GMRF, order = order, lower = 4.0)
    p0 ~ VLBIGaussian(-1.0, 2.0)
    pσ ~ VLBIExponential(0.5)
    angparams ~ ImageSphericalUniform(size(grid)...)
    mean ~ genmeanprior(meanmodel)
    flux ~ _flux_prior(ftot)
    gauss ~ gaussprior
    mimg = make_mean(meanmodel, grid, mean)
    f = _get_ftot(ftot, flux)
    pmap = make_poincare(_img_flux(f, gauss), mimg, σ .* c.params, p0, pσ, p.params, angparams)
    ms = _center_model(pmap, center, pulse, center_power)
    return _add_gauss(ms, f, gauss)
end

# Poincaré-sphere polarized imaging with stationary random fields: order-`N` Markov power
# spectra (`order < 0`) or Matérn ones (`order == 0`); `base.ps` selects them.
@sky function poincare_srf(grid; base, meanmodel, ftot, beamsize, rhoprior = UniformRhoPrior(), gaussprior = NamedTuple(), center = Val(true), center_power = 1, pulse = DeltaPulse())
    c ~ VLBIImagePriors.std_dist(base.plan)
    σ ~ VLBIExponential(1.0)
    ρ ~ spectrum_prior(base.ps, rhoprior, grid, beamsize)
    p ~ VLBIImagePriors.std_dist(base.plan)
    pρ ~ spectrum_prior(base.ps, rhoprior, grid, beamsize)
    p0 ~ VLBIGaussian(-1.0, 2.0)
    pσ ~ VLBIExponential(0.5)
    angparams ~ ImageSphericalUniform(size(grid)...)
    mean ~ genmeanprior(meanmodel)
    flux ~ _flux_prior(ftot)
    gauss ~ gaussprior
    mimg = make_mean(meanmodel, grid, mean)
    f = _get_ftot(ftot, flux)
    δ = genfield(StationaryRandomField(field_spectrum(base.ps, ρ), base.plan), c)
    pδ = genfield(StationaryRandomField(field_spectrum(base.ps, pρ), base.plan), p)
    δ .*= σ
    pmap = make_poincare(_img_flux(f, gauss), mimg, δ, p0, pσ, pδ, angparams)
    ms = _center_model(pmap, center, pulse, center_power)
    return _add_gauss(ms, f, gauss)
end

# =========================================================================================
# Matrix-exponential (PolExp) polarized imaging
# =========================================================================================

# PolExp polarized imaging with first-order GMRF fields (`order == 1`).
@sky function polexp_gmrf(grid; meanmodel, ftot, beamsize, order = 1, gaussprior = NamedTuple(), center = Val(true), center_power = 1, pulse = DeltaPulse())
    a ~ corr_image_prior(grid, beamsize; base = GMRF, order = order, lower = 4.0)
    b ~ corr_image_prior(grid, beamsize; base = GMRF, order = order, lower = 4.0)
    c ~ corr_image_prior(grid, beamsize; base = GMRF, order = order, lower = 4.0)
    d ~ corr_image_prior(grid, beamsize; base = GMRF, order = order, lower = 4.0)
    σa ~ VLBIExponential(1.0)
    σb ~ VLBIExponential(0.5)
    σc ~ VLBIExponential(0.5)
    σd ~ VLBIExponential(0.05)
    mean ~ genmeanprior(meanmodel)
    flux ~ _flux_prior(ftot)
    gauss ~ gaussprior
    mimg = make_mean(meanmodel, grid, mean)
    f = _get_ftot(ftot, flux)
    pmap = make_pol2expimage(_img_flux(f, gauss), σa .* a.params, σb .* b.params, σc .* c.params, σd .* d.params, mimg)
    ms = _center_model(pmap, center, pulse, center_power)
    return _add_gauss(ms, f, gauss)
end

# PolExp polarized imaging with non-centered Markov transforms (`order > 1`).
@sky function polexp_ncmrf(grid; base, meanmodel, ftot, beamsize, gaussprior = NamedTuple(), center = Val(true), center_power = 1, pulse = DeltaPulse())
    a ~ (
        hyperparams = VLBITruncated(
            VLBIInverseGamma(1.0, -log(0.01) * beamsize / pixelsizes(grid).X);
            lower = 1.0, upper = 2 * max(size(grid)...)
        ),
        params = VLBIImagePriors.StdNormal(size(grid)),
    )
    b ~ (
        hyperparams = VLBITruncated(
            VLBIInverseGamma(1.0, -log(0.01) * beamsize / pixelsizes(grid).X);
            lower = 1.0, upper = 2 * max(size(grid)...)
        ),
        params = VLBIImagePriors.StdNormal(size(grid)),
    )
    c ~ (
        hyperparams = VLBITruncated(
            VLBIInverseGamma(1.0, -log(0.01) * beamsize / pixelsizes(grid).X);
            lower = 1.0, upper = 2 * max(size(grid)...)
        ),
        params = VLBIImagePriors.StdNormal(size(grid)),
    )
    d ~ (
        hyperparams = VLBITruncated(
            VLBIInverseGamma(1.0, -log(0.01) * beamsize / pixelsizes(grid).X);
            lower = 1.0, upper = 2 * max(size(grid)...)
        ),
        params = VLBIImagePriors.StdNormal(size(grid)),
    )
    σa ~ VLBIExponential(1.0)
    σb ~ VLBIExponential(0.5)
    σc ~ VLBIExponential(0.5)
    σd ~ VLBIExponential(0.05)
    mean ~ genmeanprior(meanmodel)
    flux ~ _flux_prior(ftot)
    gauss ~ gaussprior
    mimg = make_mean(meanmodel, grid, mean)
    f = _get_ftot(ftot, flux)
    δa = centerdist(base, a.hyperparams, a.params)
    δb = centerdist(base, b.hyperparams, b.params)
    δc = centerdist(base, c.hyperparams, c.params)
    δd = centerdist(base, d.hyperparams, d.params)
    δa .*= σa
    δb .*= σb
    δc .*= σc
    δd .*= σd
    pmap = make_pol2expimage(_img_flux(f, gauss), δa, δb, δc, δd, mimg)
    ms = _center_model(pmap, center, pulse, center_power)
    return _add_gauss(ms, f, gauss)
end

# PolExp polarized imaging with stationary random fields: order-`N` Markov power spectra
# (`order < 0`) or Matérn ones (`order == 0`); `base.ps` selects them.
@sky function polexp_srf(grid; base, meanmodel, ftot, beamsize, rhoprior = UniformRhoPrior(), gaussprior = NamedTuple(), center = Val(true), center_power = 1, pulse = DeltaPulse())
    a ~ VLBIImagePriors.std_dist(base.plan)
    b ~ VLBIImagePriors.std_dist(base.plan)
    c ~ VLBIImagePriors.std_dist(base.plan)
    d ~ VLBIImagePriors.std_dist(base.plan)
    σa ~ VLBIExponential(1.0)
    σb ~ VLBIExponential(0.5)
    σc ~ VLBIExponential(0.5)
    σd ~ VLBIExponential(0.05)
    ρa ~ spectrum_prior(base.ps, rhoprior, grid, beamsize)
    ρb ~ spectrum_prior(base.ps, rhoprior, grid, beamsize)
    ρc ~ spectrum_prior(base.ps, rhoprior, grid, beamsize)
    ρd ~ spectrum_prior(base.ps, rhoprior, grid, beamsize)
    mean ~ genmeanprior(meanmodel)
    flux ~ _flux_prior(ftot)
    gauss ~ gaussprior
    mimg = make_mean(meanmodel, grid, mean)
    f = _get_ftot(ftot, flux)
    δa = genfield(StationaryRandomField(field_spectrum(base.ps, ρa), base.plan), a)
    δb = genfield(StationaryRandomField(field_spectrum(base.ps, ρb), base.plan), b)
    δc = genfield(StationaryRandomField(field_spectrum(base.ps, ρc), base.plan), c)
    δd = genfield(StationaryRandomField(field_spectrum(base.ps, ρd), base.plan), d)
    δa .*= σa
    δb .*= σb
    δc .*= σc
    δd .*= σd
    pmap = make_pol2expimage(_img_flux(f, gauss), δa, δb, δc, δd, mimg)
    ms = _center_model(pmap, center, pulse, center_power)
    return _add_gauss(ms, f, gauss)
end

# =========================================================================================
# Constructor selection + prior overrides
# =========================================================================================

"""
    sky_constructor(polrep::PolRep, base) -> @sky constructor

Map a polarization representation and a random-field base *marker* (from
`_order_to_base`) to the matching `@sky` constructor.
"""
sky_constructor(::TotalIntensity, ::Type{<:VLBIImagePriors.MarkovRandomField}) = stokesi_gmrf
sky_constructor(::TotalIntensity, ::NonCenteredMRF) = stokesi_ncmrf
sky_constructor(::TotalIntensity, ::Union{Matern, MarkovRF}) = stokesi_srf
sky_constructor(::Poincare, ::Type{<:VLBIImagePriors.MarkovRandomField}) = poincare_gmrf
sky_constructor(::Poincare, ::Union{Matern, MarkovRF}) = poincare_srf
sky_constructor(::PolExp, ::Type{<:VLBIImagePriors.MarkovRandomField}) = polexp_gmrf
sky_constructor(::PolExp, ::NonCenteredMRF) = polexp_ncmrf
sky_constructor(::PolExp, ::Union{Matern, MarkovRF}) = polexp_srf
sky_constructor(p::PolRep, base) =
    error("no sky model for polrep $(typeof(p)) with random-field base $(base); see sky_constructor methods for the supported combinations")

"""
    apply_sky_overrides(prior::NamedTuple, overrides::Dict{Symbol}) -> NamedTuple

Replace individual sky-prior entries by key. Keys are matched against the flat image
parameters first, then inside the `mean`, `gauss`, and `flux` groups (so the TOML
`[overrides]` table keeps working with the same flat keys it used before the `@sky`
rewrite). An override key that matches nothing is an error.
"""
function apply_sky_overrides(prior::NamedTuple, overrides::Dict{Symbol})
    for (k, v) in overrides
        if haskey(prior, k) && k ∉ (:mean, :gauss, :flux)
            prior = merge(prior, NamedTuple{(k,)}((v,)))
        elseif haskey(prior, :mean) && haskey(prior.mean, k)
            prior = merge(prior, (mean = merge(prior.mean, NamedTuple{(k,)}((v,))),))
        elseif haskey(prior, :gauss) && haskey(prior.gauss, k)
            prior = merge(prior, (gauss = merge(prior.gauss, NamedTuple{(k,)}((v,))),))
        elseif k === :ftot && haskey(prior, :flux) && haskey(prior.flux, :ftot)
            prior = merge(prior, (flux = merge(prior.flux, NamedTuple{(k,)}((v,))),))
        else
            error(
                "sky prior override '$k' does not match any prior entry. Available: " *
                    "$(collect(keys(prior))) plus the mean/gauss group parameters."
            )
        end
    end
    return prior
end
