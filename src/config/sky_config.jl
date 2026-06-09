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

function _parse_polrep(s::AbstractString)
    s == "PolExp" && return PolExp()
    s == "Poincare" && return Poincare()
    s == "TotalIntensity" && return TotalIntensity()
    error("unknown polrep '$s'. Allowed: PolExp, Poincare, TotalIntensity")
end

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
        @info "Using a Gaussian background mean for the sky model"
        return GaussBkgdMean(g)
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
    elseif mtype == "TBlob"
        @info "Using a Student-t blob mean for the sky model"
        return TBlobMean()
    elseif mtype == "JetGauss"
        fwhm = _mean_fwhm_rad(meancfg, beam, 30.0)
        @info "Using a jet+Gaussian mean for the sky model (Gaussian FWHM = $(round(rad2μas(fwhm), digits = 1)) μas)"
        mimg = intensitymap(modify(Gaussian(), Stretch(fwhm / fwhmfac)), g)
        return JetGauss(mimg ./ sum(mimg))
    else
        error("unknown mean type '$mtype'. Allowed: GaussBkgd, Bkgd, Gauss, Ring, TBlob, JetGauss")
    end
end

function _parse_sky_overrides(ocfg::AbstractDict)
    d = Dict{Symbol, Any}()
    for (k, v) in ocfg
        d[Symbol(k)] = parse_dist(v)
    end
    return d
end

"""
    build_sky_config(cfg::AbstractDict; beam=nothing) -> (SkyModel, imgdata)

Construct the `SkyModel` and (optional) centroid-regularizer image data from a parsed
image/sky TOML. `imgdata` is `nothing` unless `model.creg = true`.

`beam` is the observation's nominal resolution in radians (`beamsize(dcoh)`); the driver
passes it so the random-field correlation length and the mean-Gaussian width are set from the
data rather than hand-tuned. The random-field correlation length defaults to `model.beamsize_beams`
× `beam` (1× by default); an explicit `model.beamsize` (μas) overrides. When `beam` is `nothing`
(building the sky standalone, without data) the legacy 20 μas / `fwhm` defaults are used.
"""
function build_sky_config(cfg::AbstractDict; beam = nothing)
    grid = get(cfg, "grid", Dict{String, Any}())
    model = get(cfg, "model", Dict{String, Any}())

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

    polrep = _parse_polrep(String(get(model, "polrep", "PolExp")))
    addg = Bool(get(model, "addgauss", false))
    creg = Bool(get(model, "creg", false))

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

    if !creg
        msky = ImagingModel(polrep, mmodel, g, ftotpr; addgauss = addg, base = base, order = order)
        imgdata = nothing
    else
        @info "Using a centroid regularization"
        msky = ImagingModel(polrep, mmodel, g, ftotpr; addgauss = addg, base = base, order = order, center = false)
        imgdata = (Comrade.ImgNormalData(rad2μas ∘ centroid, SVector(0.0, 0.0), 1.0),)
    end

    pr = skyprior(msky; beamsize = corr_beam, overrides = overrides)
    skym = SkyModel(msky, pr, g; algorithm = FINUFFTAlg(; threads = Threads.nthreads()))
    return (skym, imgdata)
end
