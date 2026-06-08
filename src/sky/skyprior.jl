# Prior construction for `ImagingModel`. `skyprior` merges the image-fluctuation priors
# (`genimgprior`, dispatched on polarization rep × random-field base), the mean-model
# priors (`genmeanprior`), and—if requested—the extra-Gaussian priors (`gengaussprior`),
# applying any user `overrides` last.

"""
    skyprior(m::ImagingModel; beamsize=μas2rad(20.0), overrides=Dict())

Build the full sky prior as a `NamedTuple`. `beamsize` sets the correlation-length scale
of the random field; `overrides` replaces individual prior entries by key.
"""
function skyprior(m::ImagingModel{P}; beamsize = μas2rad(20.0), overrides::Dict = Dict()) where {P}
    imgprior = genimgprior(P, m.base, m.grid, beamsize, m.order)
    mprior = genmeanprior(m.mimg)
    if addgauss(m)
        gprior = gengaussprior(P)
    else
        gprior = Dict()
    end

    if !(m.ftot isa Real)
        imgprior[:ftot] = m.ftot
    end

    prior = merge(imgprior, mprior, gprior)

    for k in keys(overrides)
        prior[k] = overrides[k]
    end

    return NamedTuple(prior)
end

function genimgprior(::Type{<:TotalIntensity}, base::VLBIImagePriors.NonCenteredMarkovTransform, grid, beamsize, order)
    cprior = VLBIImagePriors.StdNormal(size(grid))
    bs = beamsize / pixelsizes(grid).X
    dρ = VLBITruncated(VLBIInverseGamma(1.0, -log(0.01) * bs); lower = 1.0, upper = 2 * max(size(grid)...))

    default = Dict(
        :c => (hyperparams = dρ, params = cprior),
        :σ => VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0),
    )
    return default
end

function genimgprior(::Type{<:TotalIntensity}, base::SRF{<:MarkovRF{N}}, grid, beamsize, order) where {N}
    bs = beamsize / step(grid.X)
    cprior = VLBIImagePriors.std_dist(base.plan)
    ρs = ntuple(Returns(VLBITruncated(VLBIUniform(0.1, 1.0*max(size(grid)...)); lower = 0.1, upper = 1.0)), N)
    default = Dict(
        :c => cprior,
        :σ => VLBITruncated(VLBIGaussian(0.0, 1.0); lower = 0.0),
        :ρs => ρs,
    )
    return default
end

function genimgprior(::Type{<:Poincare}, base::Type{<:VLBIImagePriors.MarkovRandomField}, grid, beamsize, order)
    cprior = corr_image_prior(grid, beamsize; base = base, order = order, lower = 4.0)
    default = Dict(
        :c => cprior,
        :σ => VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0),
        :p => cprior,
        :p0 => VLBIGaussian(-1.0, 2.0),
        :pσ => VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0),
        :angparams => ImageSphericalUniform(size(cprior.priormap.cache)...)
    )
    return default
end

function genimgprior(::Type{<:PolExp}, base::Type{<:VLBIImagePriors.MarkovRandomField}, grid, beamsize, order)
    cprior = corr_image_prior(grid, beamsize; base = base, order = order, lower = 4.0)
    default = Dict(
        :a => cprior,
        :b => cprior,
        :c => cprior,
        :d => cprior,
        :σa => VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0),
        :σb => VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0),
        :σc => VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0),
        :σd => VLBITruncated(VLBIGaussian(0.0, 0.05); lower = 0.0),
    )
    return default
end

function genimgprior(::Type{<:PolExp}, base::VLBIImagePriors.NonCenteredMarkovTransform, grid, beamsize, order)
    cprior = VLBIImagePriors.StdNormal(size(grid))
    bs = beamsize / pixelsizes(grid).X
    dρ = VLBITruncated(VLBIInverseGamma(1.0, -log(0.01) * bs); lower = 1.0, upper = 2 * max(size(grid)...))

    default = Dict(
        :a => (hyperparams = dρ, params = cprior),
        :b => (hyperparams = dρ, params = cprior),
        :c => (hyperparams = dρ, params = cprior),
        :d => (hyperparams = dρ, params = cprior),
        :σa => VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0),
        :σb => VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0),
        :σc => VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0),
        :σd => VLBITruncated(VLBIGaussian(0.0, 0.05); lower = 0.0),
    )
    return default
end

function genimgprior(::Type{<:Poincare}, base::VLBIImagePriors.StationaryMatern, grid, beamsize, order)
    bs = beamsize / step(grid.XL)
    cprior = VLBIImagePriors.std_dist(base)
    ρpr = VLBITruncated(VLBIInverseGamma(1.0, -log(0.1) * bs); lower = 4.0, upper = 2 * max(size(grid)...))
    νpr = VLBITruncated(VLBIInverseGamma(5.0, 9.0); lower = 0.1)

    default = Dict(
        :c => cprior,
        :σ => VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0),
        :ρ => ρpr,
        :ν => νpr,
        :p => cprior,
        :ρp => ρpr,
        :νp => νpr,
        :p0 => VLBIGaussian(-1.0, 2.0),
        :pσ => VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0),
        :angparams => ImageSphericalUniform(size(cprior.priormap.cache)...)
    )
    return default
end

function genimgprior(::Type{<:PolExp}, base::VLBIImagePriors.StationaryMatern, grid, beamsize, order)
    bs = beamsize / step(grid.X)
    cprior = VLBIImagePriors.std_dist(base)
    ρpr = VLBITruncated(VLBIInverseGamma(1.0, -log(0.1) * bs); lower = 4.0, upper = 2 * max(size(grid)...))
    νpr = VLBITruncated(VLBIInverseGamma(5.0, 9.0); lower = 0.1)

    default = Dict(
        :a => cprior,
        :b => cprior,
        :c => cprior,
        :d => cprior,
        :σa => VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0),
        :σb => VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0),
        :σc => VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0),
        :σd => VLBITruncated(VLBIGaussian(0.0, 0.1); lower = 0.0),
        :ρa => ρpr,
        :νa => νpr,
        :ρb => ρpr,
        :νb => νpr,
        :ρc => ρpr,
        :νc => νpr,
        :ρd => ρpr,
        :νd => νpr
    )
    return default
end

function genimgprior(::Type{<:PolExp}, base::SRF{<:MarkovRF{N}}, grid, beamsize, order) where {N}
    cprior = VLBIImagePriors.std_dist(base.plan)
    ρs = ntuple(Returns(VLBIUniform(0.1, max(size(grid)...))), N)
    default = Dict(
        :a => cprior,
        :b => cprior,
        :c => cprior,
        :d => cprior,
        :σa => VLBITruncated(VLBIGaussian(0.0, 1.0); lower = 0.0),
        :σb => VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0),
        :σc => VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0),
        :σd => VLBITruncated(VLBIGaussian(0.0, 0.1); lower = 0.0),
        :ρa => ρs,
        :ρb => ρs,
        :ρc => ρs,
        :ρd => ρs,
    )
    return default
end

function genimgprior(::Type{<:TotalIntensity}, base::Type{<:VLBIImagePriors.MarkovRandomField}, grid, beamsize, order)
    cprior = corr_image_prior(grid, beamsize; base = base, order = order, lower = 4.0)
    default = Dict(
        :c => cprior,
        :σ => VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0)
    )
    return default
end

function genimgprior(::Type{<:TotalIntensity}, base::VLBIImagePriors.StationaryMatern, grid, beamsize, order)
    bs = beamsize / step(grid.X)
    cprior = VLBIImagePriors.std_dist(base)
    ρpr = VLBITruncated(VLBIInverseGamma(1.0, -log(0.1) * bs); lower = 4.0, upper = 2 * max(size(grid)...))
    νpr = VLBITruncated(VLBIInverseGamma(5.0, 9.0); lower = 0.1)

    default = Dict(
        :c => cprior,
        :σ => VLBITruncated(VLBIGaussian(0.0, 1.0); lower = 0.0),
        :ρ => ρpr,
        :ν => νpr
    )
    return default
end

function gengaussprior(::Type{<:PolModel})
    default = Dict(
        :fg => VLBIUniform(0.0, 1.0),
        :σg => VLBIUniform(μas2rad(250.0), μas2rad(1000.0)),
        :τg => VLBIUniform(0.0, 7.0),
        :ξg => DiagonalVonMises(0.0, inv(1π^2)),
        :xg => VLBIUniform(-μas2rad(10_000.0), μas2rad(10_000)),
        :yg => VLBIUniform(-μas2rad(10_000.0), μas2rad(10_000.0)),
        :pg => VLBIUniform(0.0, 1.0),
        :pxg => VLBITruncated(VLBIGaussian(0.0, 1.0); lower = 0.0),
        :pyg => VLBITruncated(VLBIGaussian(0.0, 1.0); lower = 0.0),
        :pzg => VLBITruncated(VLBIGaussian(0.0, 1.0); lower = 0.0)
    )
    return default
end

function gengaussprior(::Type{<:TotalIntensity})
    default = Dict(
        :fg => VLBIUniform(0.0, 1.0),
        :σg => VLBIUniform(μas2rad(250.0), μas2rad(1000.0)),
        :τg => VLBIUniform(0.0, 7.0),
        :ξg => DiagonalVonMises(0.0, inv(1π^2)),
        :xg => VLBIUniform(-μas2rad(10_000.0), μas2rad(10_000)),
        :yg => VLBIUniform(-μas2rad(10_000.0), μas2rad(10_000.0)),
    )
    return default
end
