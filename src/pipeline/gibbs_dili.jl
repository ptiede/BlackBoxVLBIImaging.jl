# Kernels of the dimension-independent likelihood-informed (DILI) sampler. The sampled
# coordinates `u` are the `StdNormal` latent space of the posterior, where the prior is
# exactly N(0, I), so the target is `exp(-Φ(u)) N(u; 0, I)` with `Φ` the negative
# log-likelihood up to a constant.

"""
    dili_posterior(post::VLBIPosterior)

`post` transported to the `StdNormal` latent space: the coordinates the DILI kernels act
on. Every prior block must have an exact transport to N(0, I); circular priors need
`AngularProjectedNormal` and wrapped Gauss–Markov chains are not supported.
"""
dili_posterior(post::VLBIPosterior) = PT.transport_to(post, PT.StdNormal())

"""
    latent_layout(tpost)

The latent coordinates of each parameter of the transported posterior `tpost`, as a nested
`NamedTuple` of index ranges mirroring the prior (a `Tuple` for tuple-valued parameters).
"""
latent_layout(tpost) = _layout(PT.transport_node(tpost.transform), 0)

function _layout(t, offset::Int)
    t isa PT.TupleTransport || return (offset + 1):(offset + PT.dimension(t))
    children = getfield(t, :transports)
    ranges = Any[]
    for c in children
        push!(ranges, _layout(c, offset))
        offset += PT.dimension(c)
    end
    return children isa NamedTuple ? NamedTuple{keys(children)}(Tuple(ranges)) : Tuple(ranges)
end

# --- potential ---------------------------------------------------------------------------

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

# --- derivatives -------------------------------------------------------------------------

function dili_potential_gradient(rm, tpost, u)
    g = zero(u)
    _, v = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal, dili_potential, Enzyme.Active,
        Enzyme.Const(rm), Enzyme.Const(tpost), Enzyme.Duplicated(u, g)
    )
    return v, g
end

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

# --- compiled device kernels -------------------------------------------------------------

"""
    DILIKernels(post::VLBIPosterior)

The DILI potential and its derivatives on [`dili_posterior`](@ref)`(post)`, compiled with
Reactant for the device. Each kernel compiles on its first call and is reused; positions
and directions are runtime inputs, so new values (including pinned hyperparameters held in
`u`) never recompile.

Callable kernels (host or device vectors in, device vectors out):
- `potential(k, u)`: `Φ(u)` as a `Float64`
- `potential_gradient(k, u)`: `(Φ(u), ∇Φ(u))`
- `gauss_newton(k, u, v)`

`k.tpost` is the device posterior and `k.host` the matching host posterior.
"""
struct DILIKernels{TP, TH, R}
    tpost::TP
    host::TH
    resid::R
    compiled::Dict{Any, Any}
end

function DILIKernels(post::VLBIPosterior)
    postc = ConstructionBase.setproperties(post, (; admode = nothing))
    rpost = Comrade.prepare_device(postc, Comrade.ComradeBase.ReactantEx())
    return DILIKernels(dili_posterior(rpost), dili_posterior(post), ResidualMap(post), Dict{Any, Any}())
end

_device(x) = x isa Reactant.ConcreteRArray ? x : Reactant.to_rarray(collect(Float64, x))

function _kernel(f, k::DILIKernels, name, args...)
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

gauss_newton(k::DILIKernels, u, v) = _kernel(dili_gauss_newton, k, :gauss_newton, _device(u), _device(v))
