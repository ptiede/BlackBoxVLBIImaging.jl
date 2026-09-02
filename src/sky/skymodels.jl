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
#   - `center` is `Val(true/false)` so re-centering stays a compile-time choice.
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
@inline _center_model(pmap, ::Val{false}, pulse) = ContinuousImage(pmap, pulse)
@inline function _center_model(pmap, ::Val{true}, pulse)
    x0, y0 = centroid(pmap)
    return shifted(ContinuousImage(pmap, pulse), -x0, -y0)
end

# --- random-field plan preparation --------------------------------------------------------
# Maps the base *marker* (polreps.jl) to the concrete transform/plan object handed to the
# `@sky` constructors as their `base` keyword. `GMRF` (order 1) needs no plan — the prior
# carries it — so it has no method here.
prepare_base(::NonCenteredMRF, grid, order) = standardize(MarkovRandomFieldGraph(grid; order); flag = Comrade.VLBISkyModels.FFTW.EXHAUSTIVE)
prepare_base(::Matern, grid, order) = first(matern(size(grid)))
prepare_base(ps::MarkovRF, grid, order) = SRF(ps, StationaryRandomFieldPlan(grid))

markov_order(::MarkovRF{N}) where {N} = N

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
@sky function stokesi_gmrf(grid; meanmodel, ftot, beamsize, order = 1, gaussprior = NamedTuple(), center = Val(true), pulse = DeltaPulse())
    c ~ corr_image_prior(grid, beamsize; base = GMRF, order = order, lower = 4.0)
    σ ~ VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0)
    mean ~ genmeanprior(meanmodel)
    flux ~ _flux_prior(ftot)
    gauss ~ gaussprior
    mimg = make_mean(meanmodel, grid, mean)
    f = _get_ftot(ftot, flux)
    pmap = make_stokesi(_img_flux(f, gauss), mimg, σ .* c.params)
    ms = _center_model(pmap, center, pulse)
    return _add_gauss(ms, f, gauss)
end

# Stokes-I imaging with a non-centered Markov transform of the GMRF (`order > 1`).
@sky function stokesi_ncmrf(grid; base, meanmodel, ftot, beamsize, gaussprior = NamedTuple(), center = Val(true), pulse = DeltaPulse())
    c ~ (
        hyperparams = VLBITruncated(
            VLBIInverseGamma(1.0, -log(0.01) * beamsize / pixelsizes(grid).X);
            lower = 1.0, upper = 2 * max(size(grid)...)
        ),
        params = VLBIImagePriors.StdNormal(size(grid)),
    )
    σ ~ VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0)
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
    ms = _center_model(img, center, pulse)
    return _add_gauss(ms, f, gauss)
end

# Stokes-I imaging with a stationary Matérn fluctuation field (`order == 0`).
@sky function stokesi_matern(grid; base, meanmodel, ftot, beamsize, gaussprior = NamedTuple(), center = Val(true), pulse = DeltaPulse())
    c ~ VLBIImagePriors.std_dist(base)
    σ ~ VLBITruncated(VLBIGaussian(0.0, 1.0); lower = 0.0)
    ρ ~ VLBITruncated(
        VLBIInverseGamma(1.0, -log(0.1) * beamsize / step(grid.X));
        lower = 4.0, upper = 2 * max(size(grid)...)
    )
    ν ~ VLBITruncated(VLBIInverseGamma(5.0, 9.0); lower = 0.1)
    mean ~ genmeanprior(meanmodel)
    flux ~ _flux_prior(ftot)
    gauss ~ gaussprior
    mimg = make_mean(meanmodel, grid, mean)
    f = _get_ftot(ftot, flux)
    # NOTE: σ is sampled but not applied to δ — this reproduces the pre-macro behavior
    # (make_image(TotalIntensity, StationaryMatern) never multiplied by σ).
    δ = base(c, ρ, ν)
    pmap = make_stokesi(_img_flux(f, gauss), mimg, δ)
    ms = _center_model(pmap, center, pulse)
    return _add_gauss(ms, f, gauss)
end

# Stokes-I imaging with an order-`N` Markov power-spectrum stationary field (`order < 0`).
@sky function stokesi_markovrf(grid; base, meanmodel, ftot, beamsize, gaussprior = NamedTuple(), center = Val(true), pulse = DeltaPulse())
    c ~ VLBIImagePriors.std_dist(base.plan)
    σ ~ VLBITruncated(VLBIGaussian(0.0, 1.0); lower = 0.0)
    ρs ~ ntuple(Returns(VLBIUniform(0.1, 1.0 * max(size(grid)...))), markov_order(base.ps))
    mean ~ genmeanprior(meanmodel)
    flux ~ _flux_prior(ftot)
    gauss ~ gaussprior
    mimg = make_mean(meanmodel, grid, mean)
    f = _get_ftot(ftot, flux)
    δ = genfield(StationaryRandomField(MarkovPS(ρs), base.plan), c)
    δ .*= σ
    pmap = make_stokesi(_img_flux(f, gauss), mimg, δ)
    ms = _center_model(pmap, center, pulse)
    return _add_gauss(ms, f, gauss)
end

# =========================================================================================
# Poincaré-sphere polarized imaging
# =========================================================================================

# Poincaré-sphere polarized imaging with a first-order GMRF field (`order == 1`).
@sky function poincare_gmrf(grid; meanmodel, ftot, beamsize, order = 1, gaussprior = NamedTuple(), center = Val(true), pulse = DeltaPulse())
    c ~ corr_image_prior(grid, beamsize; base = GMRF, order = order, lower = 4.0)
    σ ~ VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0)
    p ~ corr_image_prior(grid, beamsize; base = GMRF, order = order, lower = 4.0)
    p0 ~ VLBIGaussian(-1.0, 2.0)
    pσ ~ VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0)
    angparams ~ ImageSphericalUniform(size(grid)...)
    mean ~ genmeanprior(meanmodel)
    flux ~ _flux_prior(ftot)
    gauss ~ gaussprior
    mimg = make_mean(meanmodel, grid, mean)
    f = _get_ftot(ftot, flux)
    pmap = make_poincare(_img_flux(f, gauss), mimg, σ .* c.params, p0, pσ, p.params, angparams)
    ms = _center_model(pmap, center, pulse)
    return _add_gauss(ms, f, gauss)
end

# Poincaré-sphere polarized imaging with a stationary Matérn field (`order == 0`).
# The polarized-field hyperparameters are named `pρ`/`pν` (matching the `p0`/`pσ` style);
# the pre-macro code disagreed with itself (prior `ρp`/`νp` vs body `pρ`/`pν`) and errored.
@sky function poincare_matern(grid; base, meanmodel, ftot, beamsize, gaussprior = NamedTuple(), center = Val(true), pulse = DeltaPulse())
    c ~ VLBIImagePriors.std_dist(base)
    σ ~ VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0)
    ρ ~ VLBITruncated(
        VLBIInverseGamma(1.0, -log(0.1) * beamsize / step(grid.X));
        lower = 4.0, upper = 2 * max(size(grid)...)
    )
    ν ~ VLBITruncated(VLBIInverseGamma(5.0, 9.0); lower = 0.1)
    p ~ VLBIImagePriors.std_dist(base)
    pρ ~ VLBITruncated(
        VLBIInverseGamma(1.0, -log(0.1) * beamsize / step(grid.X));
        lower = 4.0, upper = 2 * max(size(grid)...)
    )
    pν ~ VLBITruncated(VLBIInverseGamma(5.0, 9.0); lower = 0.1)
    p0 ~ VLBIGaussian(-1.0, 2.0)
    pσ ~ VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0)
    angparams ~ ImageSphericalUniform(size(grid)...)
    mean ~ genmeanprior(meanmodel)
    flux ~ _flux_prior(ftot)
    gauss ~ gaussprior
    mimg = make_mean(meanmodel, grid, mean)
    f = _get_ftot(ftot, flux)
    δ = base(c, ρ, ν)
    pδ = base(p, pρ, pν)
    δ .*= σ
    pmap = make_poincare(_img_flux(f, gauss), mimg, δ, p0, pσ, pδ, angparams)
    ms = _center_model(pmap, center, pulse)
    return _add_gauss(ms, f, gauss)
end

# =========================================================================================
# Matrix-exponential (PolExp) polarized imaging
# =========================================================================================

# PolExp polarized imaging with first-order GMRF fields (`order == 1`).
@sky function polexp_gmrf(grid; meanmodel, ftot, beamsize, order = 1, gaussprior = NamedTuple(), center = Val(true), pulse = DeltaPulse())
    a ~ corr_image_prior(grid, beamsize; base = GMRF, order = order, lower = 4.0)
    b ~ corr_image_prior(grid, beamsize; base = GMRF, order = order, lower = 4.0)
    c ~ corr_image_prior(grid, beamsize; base = GMRF, order = order, lower = 4.0)
    d ~ corr_image_prior(grid, beamsize; base = GMRF, order = order, lower = 4.0)
    σa ~ VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0)
    σb ~ VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0)
    σc ~ VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0)
    σd ~ VLBITruncated(VLBIGaussian(0.0, 0.05); lower = 0.0)
    mean ~ genmeanprior(meanmodel)
    flux ~ _flux_prior(ftot)
    gauss ~ gaussprior
    mimg = make_mean(meanmodel, grid, mean)
    f = _get_ftot(ftot, flux)
    pmap = make_pol2expimage(_img_flux(f, gauss), σa .* a.params, σb .* b.params, σc .* c.params, σd .* d.params, mimg)
    ms = _center_model(pmap, center, pulse)
    return _add_gauss(ms, f, gauss)
end

# PolExp polarized imaging with non-centered Markov transforms (`order > 1`).
@sky function polexp_ncmrf(grid; base, meanmodel, ftot, beamsize, gaussprior = NamedTuple(), center = Val(true), pulse = DeltaPulse())
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
    σa ~ VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0)
    σb ~ VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0)
    σc ~ VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0)
    σd ~ VLBITruncated(VLBIGaussian(0.0, 0.05); lower = 0.0)
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
    ms = _center_model(pmap, center, pulse)
    return _add_gauss(ms, f, gauss)
end

# PolExp polarized imaging with stationary Matérn fields (`order == 0`).
@sky function polexp_matern(grid; base, meanmodel, ftot, beamsize, gaussprior = NamedTuple(), center = Val(true), pulse = DeltaPulse())
    a ~ VLBIImagePriors.std_dist(base)
    b ~ VLBIImagePriors.std_dist(base)
    c ~ VLBIImagePriors.std_dist(base)
    d ~ VLBIImagePriors.std_dist(base)
    σa ~ VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0)
    σb ~ VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0)
    σc ~ VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0)
    σd ~ VLBITruncated(VLBIGaussian(0.0, 0.1); lower = 0.0)
    ρa ~ VLBITruncated(
        VLBIInverseGamma(1.0, -log(0.1) * beamsize / step(grid.X));
        lower = 4.0, upper = 2 * max(size(grid)...)
    )
    νa ~ VLBITruncated(VLBIInverseGamma(5.0, 9.0); lower = 0.1)
    ρb ~ VLBITruncated(
        VLBIInverseGamma(1.0, -log(0.1) * beamsize / step(grid.X));
        lower = 4.0, upper = 2 * max(size(grid)...)
    )
    νb ~ VLBITruncated(VLBIInverseGamma(5.0, 9.0); lower = 0.1)
    ρc ~ VLBITruncated(
        VLBIInverseGamma(1.0, -log(0.1) * beamsize / step(grid.X));
        lower = 4.0, upper = 2 * max(size(grid)...)
    )
    νc ~ VLBITruncated(VLBIInverseGamma(5.0, 9.0); lower = 0.1)
    ρd ~ VLBITruncated(
        VLBIInverseGamma(1.0, -log(0.1) * beamsize / step(grid.X));
        lower = 4.0, upper = 2 * max(size(grid)...)
    )
    νd ~ VLBITruncated(VLBIInverseGamma(5.0, 9.0); lower = 0.1)
    mean ~ genmeanprior(meanmodel)
    flux ~ _flux_prior(ftot)
    gauss ~ gaussprior
    mimg = make_mean(meanmodel, grid, mean)
    f = _get_ftot(ftot, flux)
    δa = base(a, ρa, νa)
    δb = base(b, ρb, νb)
    δc = base(c, ρc, νc)
    δd = base(d, ρd, νd)
    δa .*= σa
    δb .*= σb
    δc .*= σc
    δd .*= σd
    pmap = make_pol2expimage(_img_flux(f, gauss), δa, δb, δc, δd, mimg)
    ms = _center_model(pmap, center, pulse)
    return _add_gauss(ms, f, gauss)
end

# PolExp polarized imaging with order-`N` Markov power-spectrum stationary fields (`order < 0`).
@sky function polexp_markovrf(grid; base, meanmodel, ftot, beamsize, gaussprior = NamedTuple(), center = Val(true), pulse = DeltaPulse())
    a ~ VLBIImagePriors.std_dist(base.plan)
    b ~ VLBIImagePriors.std_dist(base.plan)
    c ~ VLBIImagePriors.std_dist(base.plan)
    d ~ VLBIImagePriors.std_dist(base.plan)
    σa ~ VLBITruncated(VLBIGaussian(0.0, 1.0); lower = 0.0)
    σb ~ VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0)
    σc ~ VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0)
    σd ~ VLBITruncated(VLBIGaussian(0.0, 0.1); lower = 0.0)
    ρa ~ ntuple(Returns(VLBIUniform(0.1, max(size(grid)...))), markov_order(base.ps))
    ρb ~ ntuple(Returns(VLBIUniform(0.1, max(size(grid)...))), markov_order(base.ps))
    ρc ~ ntuple(Returns(VLBIUniform(0.1, max(size(grid)...))), markov_order(base.ps))
    ρd ~ ntuple(Returns(VLBIUniform(0.1, max(size(grid)...))), markov_order(base.ps))
    mean ~ genmeanprior(meanmodel)
    flux ~ _flux_prior(ftot)
    gauss ~ gaussprior
    mimg = make_mean(meanmodel, grid, mean)
    f = _get_ftot(ftot, flux)
    δa = genfield(StationaryRandomField(MarkovPS(ρa), base.plan), a)
    δb = genfield(StationaryRandomField(MarkovPS(ρb), base.plan), b)
    δc = genfield(StationaryRandomField(MarkovPS(ρc), base.plan), c)
    δd = genfield(StationaryRandomField(MarkovPS(ρd), base.plan), d)
    δa .*= σa
    δb .*= σb
    δc .*= σc
    δd .*= σd
    pmap = make_pol2expimage(_img_flux(f, gauss), δa, δb, δc, δd, mimg)
    ms = _center_model(pmap, center, pulse)
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
sky_constructor(::TotalIntensity, ::Matern) = stokesi_matern
sky_constructor(::TotalIntensity, ::MarkovRF) = stokesi_markovrf
sky_constructor(::Poincare, ::Type{<:VLBIImagePriors.MarkovRandomField}) = poincare_gmrf
sky_constructor(::Poincare, ::Matern) = poincare_matern
sky_constructor(::PolExp, ::Type{<:VLBIImagePriors.MarkovRandomField}) = polexp_gmrf
sky_constructor(::PolExp, ::NonCenteredMRF) = polexp_ncmrf
sky_constructor(::PolExp, ::Matern) = polexp_matern
sky_constructor(::PolExp, ::MarkovRF) = polexp_markovrf
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
