# Kernels of the dimension-independent likelihood-informed (DILI) sampler. The sampled
# coordinates `u` have the reference measure N(0, I); the target is
# `exp(-Φ̃(u)) N(u; 0, I)` with `Φ̃ = Φ + Ψ`, `Φ` the negative log-likelihood (up to a
# constant) and `Ψ` the remainder of the prior and log-Jacobian not captured by N(0, I).

# --- standardization of Gaussian prior blocks ---------------------------------------------

"""
    flat_layout(tpost)

The flat coordinates of each parameter of the flat posterior `tpost`, as a nested
`NamedTuple` of index ranges mirroring the prior (a `Tuple` for tuple-valued parameters).
"""
flat_layout(tpost) = _layout(_flat_root(tpost), 0)

function _layout(t, offset::Int)
    t isa TV.TransformTuple || return (offset + 1):(offset + TV.dimension(t))
    children = getfield(t, :inner)
    ranges = Any[]
    for c in children
        push!(ranges, _layout(c, offset))
        offset += TV.dimension(c)
    end
    return children isa NamedTuple ? NamedTuple{keys(children)}(Tuple(ranges)) : Tuple(ranges)
end

"""
    gaussian_standardization(post::VLBIPosterior)

The diagonal affine map `x = μ .+ s .* z` of the flat coordinates of `post` that makes the
prior of every independent-Gaussian block (a non-phase `Normal(μ, s)` leaf whose flat
transform is the identity, e.g. leakage terms and `lgratμ`) exactly N(0, I) in `z`. Other
coordinates pass through unchanged. Returned as a rank-0 `LowRankPreconditioner`; see
[`dili_posterior`](@ref).
"""
function gaussian_standardization(post::VLBIPosterior)
    n = dimension(asflat(post))
    b = zeros(n)
    s = ones(n)
    _standardize!(b, s, post.prior, flat_layout(asflat(post)))
    return LowRankPreconditioner(b, s, zeros(n, 0), Float64[])
end

function _standardize!(b, s, d::PT.NamedDist, layout::NamedTuple)
    for k in keys(layout)
        _standardize!(b, s, getproperty(d, k), layout[k])
    end
    return nothing
end

function _standardize!(b, s, d, r)
    μs = _gaussian_affine(d)
    μs === nothing && return nothing
    b[r] .= first(μs)
    s[r] .= last(μs)
    return nothing
end

_gaussian_affine(d) = nothing
_gaussian_affine(d::Comrade.ObservedArrayPrior) = d.phase ? nothing : _gaussian_affine(d.dists)
_gaussian_affine(d::Comrade.PartiallyConditionedDist) = _gaussian_affine(d.dist)
function _gaussian_affine(d::PT.PushforwardDistribution{<:PT.ScaleShift, <:PT.StdNormal})
    _identity_flat(PT.transport_node(d, PT.TVFlat())) || return nothing
    return d.f.μ, d.f.s
end

_identity_flat(t) = t isa TV.Identity || t isa TV.ArrayTransformation{TV.Identity}

"""
    dili_posterior(post::VLBIPosterior)

The flat posterior of `post` with [`gaussian_standardization`](@ref) composed in front of
the flat transform: the coordinates the DILI kernels act on.
"""
dili_posterior(post::VLBIPosterior) = PT.transport_to(post, gaussian_standardization(post))

# --- potentials --------------------------------------------------------------------------

"""
    ResidualMap(post::VLBIPosterior)
    ResidualMap(measurement::NTuple{4}, noise::NTuple{4})

The noise-whitened residuals of the coherency data of `post`, restricted to the entries
whose measurement and noise are both finite (mixed-basis data can carry NaN products).
The mask is fixed at construction, so a non-finite *model* value propagates into the
residuals rather than being dropped. The four components are the correlation products in
the column-major order of the coherency matrices.
"""
struct ResidualMap{I, M, N}
    finite::I
    measurement::M
    noise::N
end

function ResidualMap(measurement::NTuple{4, AbstractVector}, noise::NTuple{4, AbstractVector})
    finite = map((m, σ) -> findall(isfinite.(m) .& isfinite.(σ)), measurement, noise)
    M = map((m, i) -> m[i], measurement, finite)
    N = map((σ, i) -> σ[i], noise, finite)
    all(σ -> all(>(0), σ), N) ||
        throw(ArgumentError("finite data carry a non-positive noise value"))
    return ResidualMap(finite, M, N)
end

function ResidualMap(post::VLBIPosterior)
    d = only(post.data)
    M = StructArrays.components(StructArray(Comrade.measurement(d)))
    N = StructArrays.components(StructArray(Comrade.noise(d)))
    return ResidualMap(Tuple(vec.(M)), Tuple(vec.(N)))
end

Base.length(rm::ResidualMap) = 2 * sum(length, rm.finite)

"""
    dili_resid(rm::ResidualMap, tpost, u)

The whitened residuals at `u`, real parts then imaginary parts, so that the potential is
`Φ(u) = ½‖r‖²`.
"""
function dili_resid(rm::ResidualMap, tpost, u)
    x = Comrade.transform(tpost, u)
    V = StructArrays.components(baseimage(last(Comrade.forward_model(tpost.lpost, x))))
    # ntuple, not a map over a range: Reactant's mapreduce overlay does not trace this
    z = ntuple(k -> (V[k][rm.finite[k]] .- rm.measurement[k]) ./ rm.noise[k], Val(4))
    return vcat(real.(z)..., imag.(z)...)
end

"""
    dili_potential(rm::ResidualMap, tpost, u)

`Φ(u) = ½‖r(u)‖²`, the negative log-likelihood up to a constant.
"""
dili_potential(rm::ResidualMap, tpost, u) = sum(abs2, dili_resid(rm, tpost, u)) / 2

"""
    dili_prior_remainder(tpost, u)

`Ψ(u) = -ℓ(u) - ½‖u‖²`, with `ℓ` the log prior plus log-Jacobian of the flat transform:
the part of the prior not captured by the N(0, I) reference. Zero on blocks whose prior
is exactly N(0, I) in `u`.
"""
dili_prior_remainder(tpost, u) =
    -last(PT.latent_pfwd_and_logdensity(tpost.transform, u)) - sum(abs2, u) / 2

"""
    dili_reference_potential(rm::ResidualMap, tpost, u)

`Φ̃(u) = Φ(u) + Ψ(u)`: the target is `exp(-Φ̃(u)) N(u; 0, I)`.
"""
dili_reference_potential(rm::ResidualMap, tpost, u) =
    dili_potential(rm, tpost, u) + dili_prior_remainder(tpost, u)

# --- derivatives -------------------------------------------------------------------------

function _value_and_gradient(f, rm, tpost, u)
    g = zero(u)
    _, v = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal, f, Enzyme.Active,
        Enzyme.Const(rm), Enzyme.Const(tpost), Enzyme.Duplicated(u, g)
    )
    return v, g
end

dili_potential_gradient(rm, tpost, u) = _value_and_gradient(dili_potential, rm, tpost, u)
dili_reference_gradient(rm, tpost, u) = _value_and_gradient(dili_reference_potential, rm, tpost, u)

_resid_jvp(rm, tpost, u, v) = only(
    Enzyme.autodiff(
        Enzyme.Forward, dili_resid, Enzyme.Const(rm), Enzyme.Const(tpost), Enzyme.Duplicated(u, v)
    )
)

_weighted_resid(rm, tpost, u, w) = sum(dili_resid(rm, tpost, u) .* w)

function _resid_vjp(rm, tpost, u, w)
    g = zero(u)
    Enzyme.autodiff(
        Enzyme.Reverse, _weighted_resid, Enzyme.Active,
        Enzyme.Const(rm), Enzyme.Const(tpost), Enzyme.Duplicated(u, g), Enzyme.Const(w)
    )
    return g
end

"""
    dili_gauss_newton(rm::ResidualMap, tpost, u, v)

The Gauss–Newton product `Jᵀ(J v)` of the potential `Φ`, with `J` the Jacobian of
[`dili_resid`](@ref) at `u`.
"""
dili_gauss_newton(rm, tpost, u, v) = _resid_vjp(rm, tpost, u, _resid_jvp(rm, tpost, u, v))

function _prior_remainder_gradient(tpost, u)
    g = zero(u)
    Enzyme.autodiff(Enzyme.Reverse, dili_prior_remainder, Enzyme.Active, Enzyme.Const(tpost), Enzyme.Duplicated(u, g))
    return g
end

"""
    dili_prior_hvp(tpost, u, v)

The exact Hessian-vector product `∇²Ψ(u) v` of [`dili_prior_remainder`](@ref), by forward
differentiation of its reverse-mode gradient.
"""
dili_prior_hvp(tpost, u, v) = only(
    Enzyme.autodiff(Enzyme.Forward, _prior_remainder_gradient, Enzyme.Const(tpost), Enzyme.Duplicated(u, v))
)

"""
    dili_curvature(rm::ResidualMap, tpost, u, v)

The product of the likelihood-informed curvature operator with `v`: the Gauss–Newton
product of `Φ` plus the exact Hessian of `Ψ`.
"""
dili_curvature(rm, tpost, u, v) = dili_gauss_newton(rm, tpost, u, v) .+ dili_prior_hvp(tpost, u, v)

# --- compiled device kernels -------------------------------------------------------------

"""
    DILIKernels(post::VLBIPosterior)

The DILI potentials and their derivatives on [`dili_posterior`](@ref)`(post)`, compiled
with Reactant for the device. Each kernel compiles on its first call and is reused; all
positions and directions are runtime inputs, so new values (including pinned
hyperparameters held in `u`) never recompile.

Callable kernels (host or device vectors in, device vectors out):
- `potential(k, u)`: `Φ(u)` as a `Float64`
- `potential_gradient(k, u)`: `(Φ(u), ∇Φ(u))`
- `reference_gradient(k, u)`: `(Φ̃(u), ∇Φ̃(u))`
- `gauss_newton(k, u, v)`, `prior_hvp(k, u, v)`, `curvature(k, u, v)`

`k.tpost` is the device posterior and `k.host` the matching host posterior.
"""
struct DILIKernels{TP, TH, R}
    tpost::TP
    host::TH
    resid::R
    compiled::Dict{Symbol, Any}
end

function DILIKernels(post::VLBIPosterior)
    pre = gaussian_standardization(post)
    postc = ConstructionBase.setproperties(post, (; admode = nothing))
    rpost = Comrade.prepare_device(postc, Comrade.ComradeBase.ReactantEx())
    tpost = PT.transport_to(rpost, Comrade._device_pre(pre))
    return DILIKernels(tpost, PT.transport_to(post, pre), ResidualMap(post), Dict{Symbol, Any}())
end

_device(x) = x isa Reactant.ConcreteRArray ? x : Reactant.to_rarray(collect(Float64, x))

function _kernel(f, k::DILIKernels, name::Symbol, args...)
    c = get!(k.compiled, name) do
        Reactant.@compile sync = true f(k.resid, k.tpost, args...)
    end
    return c(k.resid, k.tpost, args...)
end

potential(k::DILIKernels, u) = _host_scalar(_kernel(dili_potential, k, :potential, _device(u)))

function potential_gradient(k::DILIKernels, u)
    v, g = _kernel(dili_potential_gradient, k, :potential_gradient, _device(u))
    return _host_scalar(v), g
end

function reference_gradient(k::DILIKernels, u)
    v, g = _kernel(dili_reference_gradient, k, :reference_gradient, _device(u))
    return _host_scalar(v), g
end

gauss_newton(k::DILIKernels, u, v) = _kernel(dili_gauss_newton, k, :gauss_newton, _device(u), _device(v))

_prior_hvp(rm, tpost, u, v) = dili_prior_hvp(tpost, u, v)
prior_hvp(k::DILIKernels, u, v) = _kernel(_prior_hvp, k, :prior_hvp, _device(u), _device(v))

# Two compiled calls: a fused kernel costs as much compile time as both together.
curvature(k::DILIKernels, u, v) = gauss_newton(k, u, v) .+ prior_hvp(k, u, v)
