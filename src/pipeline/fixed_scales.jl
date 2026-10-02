# Holding sky scale parameters fixed during optimization (`[optimizer] fix_scales`) and
# moving the optimum onto the typical-set radius of its non-centered fields.

"""
    held_scale_values(post::VLBIPosterior, names) -> Dict{Symbol, Float64}

The value each scalar sky parameter in `names` is held at during optimization: the median
of its prior. Errors if a name is not a scalar sky parameter of `post`.
"""
function held_scale_values(post::VLBIPosterior, names)
    skyprior = post.prior.sky
    x = prior_sample(Random.default_rng(), post)
    scalars = sort!([String(k) for k in keys(x.sky) if x.sky[k] isa Real])
    return Dict{Symbol, Float64}(
        map(names) do k
            (haskey(x.sky, k) && x.sky[k] isa Real) || error(
                "fix_scales entry '$k' is not a scalar sky parameter of this model. " *
                    "Scalar sky parameters: $scalars"
            )
            k => Float64(quantile(getproperty(skyprior, k), 0.5))
        end
    )
end

_hold_scales(x, held::AbstractDict) =
    isempty(held) ? x : merge(x, (sky = merge(x.sky, NamedTuple(held)),))

# Latent-space mask, `false` at the coordinate of each held parameter. Found by moving one
# held parameter between two in-support values and recording which latent coordinate moves.
function _free_coordinates(post::VLBIPosterior, x, held::AbstractDict; space = nothing)
    tpost = Comrade.maybe_transport(post, space)
    free = trues(dimension(tpost))
    for k in keys(held)
        d = getproperty(post.prior.sky, k)
        y1 = Comrade.inverse(tpost, _hold_scales(x, Dict(k => quantile(d, 0.25))))
        y2 = Comrade.inverse(tpost, _hold_scales(x, Dict(k => quantile(d, 0.75))))
        moved = findall(y1 .!= y2)
        length(moved) == 1 || error(
            "fix_scales entry '$k' maps to $(length(moved)) latent coordinates; only a " *
                "scalar parameter with its own latent coordinate can be held fixed"
        )
        free[only(moved)] = false
    end
    return free
end

function _check_held(x, held::AbstractDict)
    for (k, v) in held
        isapprox(x.sky[k], v; rtol = 1.0e-12) || error(
            "held sky parameter '$k' moved during optimization: $(x.sky[k]) instead of $v"
        )
    end
    return nothing
end

# `comrade_opt` in the latent space `space` with the `held` parameters fixed: the optimizer
# sees only the free latent coordinates.
function _comrade_opt_held(
        post::VLBIPosterior, opt, held::AbstractDict; initial_params, space = nothing, kwargs...
    )
    isempty(held) && return comrade_opt(
        post, opt; initial_params, transform = p -> Comrade.maybe_transport(p, space), kwargs...
    )
    tpost = Comrade.maybe_transport(post, space)
    x0 = _hold_scales(initial_params, held)
    y0 = Comrade.inverse(tpost, x0)
    idx = findall(_free_coordinates(post, x0, held; space))
    embed(u) = (y = copy(y0); y[idx] .= u; y)
    ℓ(u, p = tpost) = -logdensityof(p, embed(u))
    function grad!(G, u, p)
        _, dy = LogDensityProblems.logdensity_and_gradient(p, embed(u))
        G .= .-view(dy, idx)
        return G
    end
    f = Optimization.OptimizationFunction(ℓ; grad = grad!)
    sol = Optimization.solve(Optimization.OptimizationProblem(f, y0[idx], tpost), opt; kwargs...)
    return Comrade.transform(tpost, embed(sol.u)), sol
end

"""
    rescale_fields(post::VLBIPosterior, x) -> NamedTuple

Move `x` onto the typical-set radius of every non-centered sky field without changing the
image. A field is a sky array `X` with a scalar scale `σX` entering the image as `σX * X`;
each is replaced by `σX * rms(X)` and `X / rms(X)`, so `rms(X) = 1` and the product is
unchanged. A field that is all zeros stays zero and its scale is set to its prior median.

Errors if the sky has no such pair, or if the rescaled image differs from the original by
more than `rtol = 1e-10` (the model does not use the scale as a pure multiplier).
"""
function rescale_fields(post::VLBIPosterior, x; rtol = 1.0e-10)
    sky = x.sky
    fields = [
        k for k in keys(sky)
            if sky[k] isa AbstractArray && get(sky, Symbol(:σ, k), nothing) isa Real
    ]
    isempty(fields) && error(
        "no non-centered field (an array `X` with a scalar scale `σX`) among the sky " *
            "parameters $(collect(keys(sky)))"
    )
    new = mapreduce(vcat, fields) do k
        s = Symbol(:σ, k)
        z = sky[k]
        r = sqrt(mean(abs2, z))
        iszero(r) && return [k => sky[k], s => Float64(quantile(getproperty(post.prior.sky, s), 0.5))]
        σ = sky[s] * r
        return [k => z ./ r, s => σ]
    end
    x1 = merge(x, (sky = merge(sky, NamedTuple(new)),))
    grid = post.skymodel.grid.imgdomain
    img0 = baseimage(intensitymap(skymodel(post, x), grid))
    img1 = baseimage(intensitymap(skymodel(post, x1), grid))
    isapprox(img1, img0; rtol) || error(
        "rescaling the fields $fields changed the image (relative difference " *
            "$(norm(img1 - img0) / norm(img0)) > $rtol)"
    )
    return x1
end
