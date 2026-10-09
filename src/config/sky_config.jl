# Parse the image/sky TOML into a Comrade `SkyModel` (plus optional centroid-regularizer
# image data). Mirrors the grid/mean/flux/order logic that used to live in the CLI driver.

function _order_to_base(order::Int)
    if order == 1
        return GMRF
    elseif order > 1
        return NonCenteredMRF(GMRF)
    elseif order < 0
        @info "Using a Markov RF expansion of order $(abs(order))"
        return MarkovRF(abs(order))
    else # order == 0
        @info "Using a Matern covariance for the random field"
        return Matern()
    end
end

# NonCenteredMRF (order > 1) prefers an image size whose value+1 is a product of small
# primes; the other bases prefer the size itself to be such a product.
function _snap_noncentered(n::Int)
    if n == nextprod((2, 3, 5, 7), n)
        return n - 1
    elseif n + 1 == nextprod((2, 3, 5, 7), n + 1)
        return n
    else
        return nextprod((2, 3, 5, 7), n + 1) - 1
    end
end

"""
    snap_grid_size(base, order, nx, ny) -> (nx, ny)

Adjust pixel counts to FFT-friendly sizes for the chosen random-field base, warning when a
change is made. A warning here is expected, not an error.
"""
function snap_grid_size(base, order::Int, nx::Int, ny::Int)
    if base isa Matern || base isa MarkovRF || (base === GMRF && order == 1)
        nx2 = nextprod((2, 3, 5, 7), nx)
        ny2 = nextprod((2, 3, 5, 7), ny)
        if (nx2 != nx) || (ny2 != ny)
            @warn "Image size ($nx, $ny) is not optimal for $base; using ($nx2, $ny2) instead."
        end
        return nx2, ny2
    elseif base isa NonCenteredMRF && order > 1
        nx2 = _snap_noncentered(nx)
        ny2 = _snap_noncentered(ny)
        if (nx2 != nx) || (ny2 != ny)
            @warn "Image size ($nx, $ny) is not optimal for $base; using ($nx2, $ny2) instead."
        end
        return nx2, ny2
    end
    return nx, ny
end

"""
    _parse_pulse(s) -> Pulse

Parse `[model] pulse`: the interpolation kernel that turns the pixel raster into a
continuous sky model. Allowed: `delta` (default), `bspline0`, `bspline1`, `bspline3`,
`bicubic`, `gaussian`, `raised_cosine`.
"""
function _parse_pulse(s::AbstractString)
    s == "delta" && return DeltaPulse()
    s == "bspline0" && return BSplinePulse{0}()
    s == "bspline1" && return BSplinePulse{1}()
    s == "bspline3" && return BSplinePulse{3}()
    s == "bicubic" && return BicubicPulse()
    s == "gaussian" && return Gaussian()
    s == "raised_cosine" && return RaisedCosinePulse()
    error("unknown pulse '$s'. Allowed: delta, bspline0, bspline1, bspline3, bicubic, gaussian, raised_cosine")
end

"""
    _parse_rho_prior(s) -> rho-prior marker

Parse `[model] rho_prior`: the prior family for the spectral parameters of a Markov RF
expansion (`order < 0`) or a Matérn field (`order = 0`). Allowed: `uniform` (default),
`lognormal`. See [`spectrum_prior`](@ref) for what each family is.
"""
function _parse_rho_prior(s::AbstractString)
    s == "uniform" && return UniformRhoPrior()
    s == "lognormal" && return LogNormalRhoPrior()
    error("unknown rho_prior '$s'. Allowed: uniform, lognormal")
end

function _parse_polrep(s::AbstractString)
    s == "PolExp" && return PolExp()
    s == "Poincare" && return Poincare()
    s == "TotalIntensity" && return TotalIntensity()
    error("unknown polrep '$s'. Allowed: PolExp, Poincare, TotalIntensity")
end

"""
    sky_polrep(cfg::AbstractDict) -> PolRep

Parse the polarization representation from a parsed image/sky TOML (`[model] polrep`,
default `PolExp`). This is the single parse point shared by `build_sky_config` and the data
loader (which derives its data product — coherencies vs complex visibilities — from it).
"""
sky_polrep(cfg::AbstractDict) =
    _parse_polrep(String(get(get(cfg, "model", Dict{String, Any}()), "polrep", "PolExp")))

function _parse_ftot(ftot)
    fs = Float64.(ftot)
    if length(fs) == 1
        @info "Using a fixed total flux of $(fs[1])"
        return fs[1]
    elseif length(fs) == 2
        @info "Fitting the total flux between $(fs[1]) and $(fs[2])"
        # VLBIUniform (not Distributions.Uniform) so the fit-flux prior traces under Reactant
        # as well as on CPU.
        return VLBIImagePriors.VLBIUniform(fs[1], fs[2])
    else
        error("flux.ftot must have 1 (fixed) or 2 (range) values, got $(fs)")
    end
end

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

function _build_mean_model(meancfg::AbstractDict, g, beam)
    mtype = String(get(meancfg, "type", "Bkgd"))
    if mtype == "GaussBkgd"
        fwhm0 = _mean_fwhm_rad(meancfg, beam, 50.0)
        # Lower bound of the core-size prior comes from the data beam itself, not the
        # (possibly rescaled) fwhm0; without data, fall back to fwhm0.
        beam0 = isnothing(beam) ? fwhm0 : beam
        @info "Using a Gaussian background mean for the sky model (core FWHM prior centered at $(round(rad2μas(fwhm0), digits = 1)) μas, lower bound $(round(rad2μas(0.25 * beam0), digits = 1)) μas)"
        return GaussBkgdMean(g, fwhm0, beam0)
    elseif mtype == "Bkgd"
        fwhm = _mean_fwhm_rad(meancfg, beam, 50.0)
        @info "Using a background mean for the sky model (Gaussian FWHM = $(round(rad2μas(fwhm), digits = 1)) μas)"
        mimg = intensitymap(modify(Gaussian(), Stretch(fwhm / fwhmfac)), g)
        return MimgPlusBkg(mimg ./ sum(mimg))
    elseif mtype == "Gauss"
        @info "Using a Gaussian mean for the sky model"
        return GaussMean()
    elseif mtype == "Ring"
        @info "Using a ring mean for the sky model"
        return DblRingMean()
    elseif mtype == "LyapunovRing"
        @info "Using a Lyapunov double-ring mean for the sky model"
        return LyapunovRingMean()
    elseif mtype == "LyapunovRingJSU"
        @info "Using a Lyapunov double-ring mean (JohnsonSU n=0) for the sky model"
        return LyapunovRingJSUMean()
    elseif mtype == "TBlob"
        @info "Using a Student-t blob mean for the sky model"
        return TBlobMean()
    elseif mtype == "JetGauss"
        fwhm = _mean_fwhm_rad(meancfg, beam, 30.0)
        @info "Using a jet+Gaussian mean for the sky model (Gaussian FWHM = $(round(rad2μas(fwhm), digits = 1)) μas)"
        mimg = intensitymap(modify(Gaussian(), Stretch(fwhm / fwhmfac)), g)
        return JetGauss(mimg ./ sum(mimg))
    else
        error("unknown mean type '$mtype'. Allowed: GaussBkgd, Bkgd, Gauss, Ring, LyapunovRing, LyapunovRingJSU, TBlob, JetGauss")
    end
end

_parse_sky_override_value(v::AbstractDict) = parse_dist(v)
_parse_sky_override_value(v::AbstractVector) = Tuple(parse_dist(vi) for vi in v)

# A sky prior entry is either a single distribution (e.g. `σa`) or, for a Markov RF term
# (`ρa`, ...), an `NTuple` of one distribution per order — so an override is either a
# distribution spec table or an array of one spec per term.
function _parse_sky_overrides(ocfg::AbstractDict)
    d = Dict{Symbol, Any}()
    for (k, v) in ocfg
        d[Symbol(k)] = _parse_sky_override_value(v)
    end
    return d
end

"""
    build_sky_config(cfg::AbstractDict; beam=nothing) -> (SkyModel, imgdata)

Construct the `SkyModel` and (optional) centroid-regularizer image data from a parsed
image/sky TOML. `imgdata` is `nothing` unless `model.creg = true` or `model.partial_centering`
is set.

`beam` is the observation's nominal resolution in radians (`beamsize(dcoh)`); the driver
passes it so the random-field correlation length and the mean-Gaussian width are set from the
data rather than hand-tuned. The random-field correlation length defaults to `model.beamsize_beams`
× `beam` (1× by default); an explicit `model.beamsize` (μas) overrides. When `beam` is `nothing`
(building the sky standalone, without data) the legacy 20 μas / `fwhm` defaults are used.

`model.rho_prior` picks the prior family for the per-term correlation lengths of a Markov RF
expansion (`order < 0`): `"uniform"` (default) or `"lognormal"`. It applies to no other base,
so setting it alongside a different `order` is an error.

`model.center` sets whether the sky model re-centers its raster on its own centroid. Without
the key the choice comes from `centerfix` of the mean model, except under
`model.creg = true`, which turns re-centering off. `center = false` turns it off explicitly
(with or without `creg`), `center = true` turns it on; `center = true` alongside
`creg = true` is an error, since both pin the image position.

`model.center_power` is the intensity power `p ≥ 1` of the [`power_centroid`](@ref) that both
mechanisms use, `1` (the center of light) by default. `p = 2` follows the bright compact
structure and barely responds to faint diffuse flux. The power reaches the model only through
re-centering or the centroid regularizer, so setting it while neither is in effect is an error.

`model.partial_centering` is the path of a serialized `(λ, ref)` named tuple of
`nx × ny` arrays that partially centers the field-`a` coefficients of a PolExp model with a
stationary random-field base (see [`PartialCentering`](@ref)); its density correction joins
`imgdata` as a `Comrade.ParamLogDensity`. Moves that rescale field `a` are not invariant
under it.

`[overrides]` replaces individual sky-prior entries by name (see [`apply_sky_overrides`](@ref)
for the matching rules). A value is a distribution spec table (`{ dist = "Normal", args =
[...] }`, parsed by [`parse_dist`](@ref)) for a scalar prior entry (`σa`, ...), or an array of
one such table per term for a Markov RF correlation length (`ρa`, ...), whose prior is an
`NTuple` of one distribution per order.
"""
function build_sky_config(cfg::AbstractDict; beam = nothing)
    check_config_keys(
        cfg, ("grid", "model", "mean", "flux", "overrides"), "the image config (top level)"
    )
    grid = get(cfg, "grid", Dict{String, Any}())
    model = get(cfg, "model", Dict{String, Any}())
    check_config_keys(grid, ("fovx", "fovy", "nx", "ny", "pa", "x0", "y0"), "[grid]")
    check_config_keys(
        model,
        (
            "polrep", "order", "addgauss", "creg", "center", "center_power",
            "beamsize_beams", "beamsize", "pulse", "rho_prior", "partial_centering",
        ),
        "[model]"
    )
    check_config_keys(
        get(cfg, "mean", Dict{String, Any}()), ("type", "fwhm", "fwhm_beams"), "[mean]"
    )
    check_config_keys(get(cfg, "flux", Dict{String, Any}()), ("ftot",), "[flux]")

    fovx = Float64(get(grid, "fovx", 200.0))
    fovy = Float64(get(grid, "fovy", 200.0))
    nx = Int(get(grid, "nx", 63))
    ny = Int(get(grid, "ny", 63))
    pa = Float64(get(grid, "pa", 0.0))     # degrees
    x0 = Float64(get(grid, "x0", 0.0))     # μas
    y0 = Float64(get(grid, "y0", 0.0))     # μas

    order = Int(get(model, "order", 1))
    base = _order_to_base(order)
    nx, ny = snap_grid_size(base, order, nx, ny)
    @info "Number of pixels: ($nx, $ny)"

    polrep = sky_polrep(cfg)
    addg = Bool(get(model, "addgauss", false))
    creg = Bool(get(model, "creg", false))
    pulse = _parse_pulse(String(get(model, "pulse", "delta")))

    # `rho_prior` only reaches the stationary-random-field constructors; with any other base
    # it would be accepted and then silently ignored.
    issrf = base isa Union{MarkovRF, Matern}
    if haskey(model, "rho_prior") && !issrf
        error(
            "[model] rho_prior sets the spectral-parameter prior of a Markov RF expansion " *
                "(order < 0) or a Matérn field (order = 0), but order = $order selects $base, " *
                "which has no such parameter"
        )
    end
    rhoprior = _parse_rho_prior(String(get(model, "rho_prior", "uniform")))
    issrf && @info "Random-field spectral-parameter prior: $(nameof(typeof(rhoprior)))"

    # Random-field correlation length: from the data beam by default (× `beamsize_beams`),
    # `model.beamsize` (μas) overrides, 20 μas fallback only when no beam is available.
    corr_beam = if haskey(model, "beamsize")
        μas2rad(Float64(model["beamsize"]))
    elseif !isnothing(beam)
        Float64(get(model, "beamsize_beams", 1.0)) * beam
    else
        μas2rad(20.0)
    end
    isnothing(beam) || @info "Data beam = $(round(rad2μas(beam), digits = 1)) μas; random-field correlation = $(round(rad2μas(corr_beam), digits = 1)) μas"

    ex = imaging_executor()
    g = imagepixels(
        μas2rad(fovx), μas2rad(fovy), nx, ny,
        μas2rad(x0), μas2rad(y0), posang = deg2rad(pa), executor = ex
    )

    # Resolution check: how finely the grid samples the beam. Aim for ≳ 3-4 pixels/beam;
    # fewer means the grid under-resolves the data and the reconstruction will be pixelated.
    px = pixelsizes(g)
    if !isnothing(beam)
        ppbx, ppby = beam / px.X, beam / px.Y
        msg = "Pixel size = $(round(rad2μas(px.X), digits = 2)) × $(round(rad2μas(px.Y), digits = 2)) μas " *
            "→ $(round(ppbx, digits = 1)) × $(round(ppby, digits = 1)) pixels/beam"
        min(ppbx, ppby) < 3 ? (@warn "$msg (under 3 px/beam — consider more pixels / smaller FOV)") : (@info msg)
    else
        @info "Pixel size = $(round(rad2μas(px.X), digits = 2)) × $(round(rad2μas(px.Y), digits = 2)) μas"
    end

    mmodel = _build_mean_model(get(cfg, "mean", Dict{String, Any}()), g, beam)
    ftotpr = _parse_ftot(get(get(cfg, "flux", Dict{String, Any}()), "ftot", [1.0]))
    overrides = _parse_sky_overrides(get(cfg, "overrides", Dict{String, Any}()))

    gaussp = addg ? gengaussprior(polrep) : NamedTuple()

    # Intensity power of the centroid both position-pinning mechanisms use; see
    # `power_centroid`. p = 1 is the center of light, larger p tracks the compact structure.
    cpower = get(model, "center_power", 1)
    (cpower isa Real && cpower >= 1) ||
        error("[model] center_power must be a real number >= 1, got $(repr(cpower))")
    centmsg = isone(cpower) ? "centroid" : "I^$(cpower)-weighted centroid"

    # The image position is pinned either by re-centering the raster on its centroid or by the
    # centroid regularizer, never by both. `[model] center` states the choice outright;
    # otherwise the mean model decides through `centerfix`, and `creg` turns re-centering off.
    if creg
        if Bool(get(model, "center", false))
            error(
                "[model] center = true re-centers the image on its own centroid, but " *
                    "creg = true already pins the centroid through a regularization term; " *
                    "set only one of them"
            )
        end
        @info "Using a $centmsg regularization"
        docenter = false
        cfunc = rad2μas ∘ SVector ∘ Base.Fix2(power_centroid, cpower)
        imgdata = (Comrade.ImgNormalData(cfunc, SVector(0.0, 0.0), 1.0),)
    else
        docenter = Bool(get(model, "center", centerfix(typeof(mmodel))))
        imgdata = nothing
    end

    # Partial centering of field a: a serialized `(λ, ref)` named tuple, one entry per
    # coefficient of the field (see `PartialCentering`).
    pcenter = nothing
    if haskey(model, "partial_centering")
        (polrep isa PolExp && issrf) || error(
            "[model] partial_centering applies to field a of a PolExp model with a stationary " *
                "random-field base (order <= 0), not polrep $(typeof(polrep)) with $base"
        )
        pcfile = String(model["partial_centering"])
        pcnt = deserialize(pcfile)
        pcenter = PartialCentering(pcnt.λ, pcnt.ref)
        size(pcenter.λ) == (nx, ny) || throw(
            DimensionMismatch("partial_centering arrays in $pcfile are $(size(pcenter.λ)), the grid is $((nx, ny))")
        )
        @info "Partially centering field a from $pcfile (mean λ = $(round(sum(pcenter.λ) / length(pcenter.λ), digits = 3)))"
    end

    if haskey(model, "center_power") && !docenter && !creg
        error(
            "[model] center_power sets the intensity weighting of the centroid the image " *
                "position is pinned to, but nothing pins it: set center = true or " *
                "creg = true, or drop center_power"
        )
    end
    @info docenter ? "Re-centering the image on its $centmsg" : "Image re-centering is off"

    ctor = sky_constructor(polrep, base)
    skym = if base === GMRF
        ctor(
            g; meanmodel = mmodel, ftot = ftotpr, beamsize = corr_beam, order = order,
            gaussprior = gaussp, center = Val(docenter), center_power = cpower, pulse
        )
    elseif issrf
        pbase = prepare_base(base, g, order)
        if !isnothing(pcenter)
            pbase = PartiallyCenteredSRF(pbase.ps, pbase.plan, pcenter)
            pcterm = partial_centering_logdensity(pbase)
            imgdata = isnothing(imgdata) ? (pcterm,) : (imgdata..., pcterm)
        end
        ctor(
            g; base = pbase, meanmodel = mmodel, ftot = ftotpr,
            beamsize = corr_beam, gaussprior = gaussp, center = Val(docenter),
            center_power = cpower, pulse, rhoprior
        )
    else
        ctor(
            g; base = prepare_base(base, g, order), meanmodel = mmodel, ftot = ftotpr,
            beamsize = corr_beam, gaussprior = gaussp, center = Val(docenter),
            center_power = cpower, pulse
        )
    end
    skym = @set skym.prior = apply_sky_overrides(skym.prior, overrides)
    # The @sky constructor cannot pass a Fourier algorithm, so swap in the threaded NUFFT.
    skym = @set skym.algorithm = FINUFFTAlg(; threads = Threads.nthreads())
    return (skym, imgdata)
end
