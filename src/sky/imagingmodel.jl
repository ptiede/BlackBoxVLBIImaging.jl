# The central sky model. `ImagingModel` encodes every structural choice in its type
# parameters so that dispatch is static and Enzyme-friendly:
#   P  - polarization representation (Poincare / PolExp / TotalIntensity)
#   M  - mean-image model type
#   G  - grid type
#   F  - flux (a Real for fixed flux, or a distribution for a fit flux)
#   B  - random-field base representation
#   AG - whether an extra Gaussian component is added
#   C  - whether the image is re-centered on its centroid
struct ImagingModel{P, M, G, F, B, AG, C}
    mimg::M
    grid::G
    ftot::F
    base::B
    order::Int
end
Enzyme.EnzymeRules.inactive_type(::Type{<:ImagingModel}) = true

centerfix(::Type{<:Any}) = true

function ImagingModel(p::PolRep, mimg::M, grid, ftot; order = 1, base = GMRF, center = centerfix(M), addgauss = false) where {M}
    b = prepare_base(base, grid, order)
    bt = typeof(b) === UnionAll ? Type{b} : typeof(b)
    return ImagingModel{typeof(p), M, typeof(grid), typeof(ftot), bt, addgauss, center}(mimg, grid, ftot, b, order)
end

function ImagingModel(p::PolRep, mimg::IntensityMap, ftot; order = 1, base = GMRF, addgauss = false)
    return ImagingModel(p, mimg ./ sum(mimg), axisdims(mimg), ftot; order = order, base = base, addgauss = addgauss)
end

prepare_base(b::Type{<:VLBIImagePriors.MarkovRandomField}, grid, order) = b
prepare_base(::NonCenteredMRF, grid, order) = (standardize(MarkovRandomFieldGraph(grid; order); flag = Comrade.VLBISkyModels.FFTW.EXHAUSTIVE))
prepare_base(::Matern, grid, order) = first(matern(size(grid)))
@inline prepare_base(ps::MarkovRF{N}, grid, order) where {N} = SRF(ps, StationaryRandomFieldPlan(grid))

@inline addgauss(::ImagingModel{P, M, G, F, B, AG}) where {P, M, G, F, B, AG} = AG
@inline center(::ImagingModel{P, M, G, F, B, AG, C}) where {P, M, G, F, B, AG, C} = C

getftot(m::ImagingModel{P, M, G, <:Real}, _) where {P, M, G} = m.ftot
getftot(::ImagingModel{P, M, G}, θ) where {P, M, G} = θ.ftot

function (m::ImagingModel{P})(θ, meta) where {P}
    mimg = make_mean(m.mimg, m.grid, θ)
    ftot = getftot(m, θ)
    if addgauss(m)
        fimg = ftot * (1 - θ.fg)
    else
        fimg = ftot
    end

    pmap = make_image(P, m.base, fimg, mimg, θ)
    if center(m)
        x0, y0 = centroid(pmap)
        ms = shifted(ContinuousImage(pmap, DeltaPulse()), -x0, -y0)
    else
        ms = ContinuousImage(pmap, DeltaPulse())
    end

    model = addgauss(m, ftot, ms, θ)
    return model
end

@inline function addgauss(m::ImagingModel{<:PolModel}, ftot, ms, θ)
    if addgauss(m)
        (; fg, σg, τg, ξg, xg, yg, pg, pxg, pyg, pzg) = θ
        g = modify(Gaussian(), Stretch(σg, σg * (1 + τg)), Rotate(ξg / 2), Shift(xg, yg), Renormalize(ftot * fg))
        pr = sqrt(pxg^2 + pyg^2 + pzg^2) + 1.0e-6
        polg = PolarizedModel(g, (pg * pxg / pr) * g, (pg * pyg / pr) * g, (pg * pzg / pr) * g)
        return ms + polg
    else
        return ms
    end
end

@inline function addgauss(m::ImagingModel{<:TotalIntensity}, ftot, ms, θ)
    if addgauss(m)
        (; fg, σg, τg, ξg, xg, yg) = θ
        g = modify(Gaussian(), Stretch(σg, σg * (1 + τg)), Rotate(ξg / 2), Shift(xg, yg), Renormalize(ftot * fg))
        return ms + g
    else
        return ms
    end
end

function make_image(::Type{<:TotalIntensity}, t::VLBIImagePriors.NonCenteredMarkovTransform, ftot, mimg, θ)
    (; c, σ) = θ
    δ = centerdist(t, c.hyperparams, c.params)
    δ .*= σ
    img = IntensityMap(δ, axisdims(mimg))
    apply_fluctuations!(CenteredLR(), img, mimg, δ)
    bimg = baseimage(img)
    bimg .*= ftot
    return img
end

@inline function make_image(::Type{<:Poincare}, ::Type{<:VLBIImagePriors.MarkovRandomField}, ftot, mimg, θ)
    (; c, σ, p, p0, pσ, angparams) = θ
    return make_poincare(ftot, mimg, σ .* c.params, p0, pσ, p.params, angparams)
end

@inline function make_image(::Type{<:PolExp}, ::Type{<:VLBIImagePriors.MarkovRandomField}, ftot, mimg, θ)
    (; a, b, c, d, σa, σb, σc, σd) = θ
    δa = σa .* a.params
    δb = σb .* b.params
    δc = σc .* c.params
    δd = σd .* d.params
    return make_pol2expimage(ftot, δa, δb, δc, δd, mimg)
end

@inline function make_image(::Type{<:PolExp}, t::VLBIImagePriors.NonCenteredMarkovTransform, ftot, mimg, θ)
    (; a, b, c, d, σa, σb, σc, σd) = θ
    δa = centerdist(t, a.hyperparams, a.params)
    δb = centerdist(t, b.hyperparams, b.params)
    δc = centerdist(t, c.hyperparams, c.params)
    δd = centerdist(t, d.hyperparams, d.params)

    δa .*= σa
    δb .*= σb
    δc .*= σc
    δd .*= σd

    return make_pol2expimage(ftot, δa, δb, δc, δd, mimg)
end

@inline function make_image(::Type{<:Poincare}, trf::VLBIImagePriors.StationaryMatern, ftot, mimg, θ)
    (; c, σ, ρ, ν, p, p0, pσ, pν, pρ, angparams) = θ
    δ = trf(c, ρ, ν)
    pδ = trf(p, pρ, pν)

    δ .*= σ
    return make_poincare(ftot, mimg, δ, p0, pσ, pδ, angparams)
end

@inline function make_image(::Type{<:PolExp}, trf::VLBIImagePriors.StationaryMatern, ftot, mimg, θ)
    (; a, b, c, d, ρa, ρb, ρc, ρd, νa, νb, νc, νd, σa, σb, σc, σd) = θ
    δa = trf(a, ρa, νa)
    δb = trf(b, ρb, νb)
    δc = trf(c, ρc, νc)
    δd = trf(d, ρd, νd)

    δa .*= σa
    δb .*= σb
    δc .*= σc
    δd .*= σd
    return make_pol2expimage(ftot, δa, δb, δc, δd, mimg)
end

@inline function make_image(::Type{<:PolExp}, trf::SRF{<:MarkovRF{N}}, ftot, mimg, θ) where {N}
    (; a, b, c, d, ρa, ρb, ρc, ρd, σa, σb, σc, σd) = θ
    δa = genfield(StationaryRandomField(MarkovPS(ρa), trf.plan), a)
    δb = genfield(StationaryRandomField(MarkovPS(ρb), trf.plan), b)
    δc = genfield(StationaryRandomField(MarkovPS(ρc), trf.plan), c)
    δd = genfield(StationaryRandomField(MarkovPS(ρd), trf.plan), d)

    δa .*= σa
    δb .*= σb
    δc .*= σc
    δd .*= σd

    return make_pol2expimage(ftot, δa, δb, δc, δd, mimg)
end

@inline function make_image(::Type{<:TotalIntensity}, trf::VLBIImagePriors.StationaryMatern, ftot, mimg, θ)
    (; c, σ, ρ, ν) = θ
    δ = trf(c, ρ, ν)
    return make_stokesi(ftot, mimg, δ)
end

@inline function make_image(::Type{<:TotalIntensity}, trf::SRF{<:MarkovRF}, ftot, mimg, θ)
    (; c, σ, ρs) = θ
    ps = MarkovPS(ρs)
    δ = genfield(StationaryRandomField(ps, trf.plan), c)
    δ .*= σ
    return make_stokesi(ftot, mimg, δ)
end

@inline function make_image(::Type{<:TotalIntensity}, ::Type{<:VLBIImagePriors.MarkovRandomField}, ftot, mimg, θ)
    return make_stokesi(ftot, mimg, θ.σ .* θ.c.params)
end

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
    # this allocated a whole new map so we can do things in place after
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
