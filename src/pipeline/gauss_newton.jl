# The Gauss–Newton operator of the likelihood in the `StdNormal` latent space of the
# posterior, where the prior is exactly N(0, I), so the target is `exp(-Φ(u)) N(u; 0, I)`
# with `Φ(u) = ½‖r(u)‖²` the negative log-likelihood up to a constant; and its leading
# eigenpairs averaged over posterior draws.

"""
    stdnormal_posterior(post::VLBIPosterior)

`post` transported to the `StdNormal` latent space, where the prior is exactly N(0, I).
Every prior block must have an exact transport to N(0, I); circular priors need
`AngularProjectedNormal` or `WrappedNormal`, and wrapped Gauss–Markov chains are not
supported.
"""
stdnormal_posterior(post::VLBIPosterior) = PT.transport_to(post, PT.StdNormal())

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

# --- residuals ---------------------------------------------------------------------------

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
    whitened_residuals(rm::ResidualMap, tpost, u)

The whitened residuals at `u`, real parts then imaginary parts, so that the negative
log-likelihood is `Φ(u) = ½‖r‖²` up to a constant.
"""
function whitened_residuals(rm::ResidualMap, tpost, u)
    x = Comrade.transform(tpost, u)
    V = StructArrays.components(baseimage(last(Comrade.forward_model(tpost.lpost, x))))
    # ntuple, not a map over a range: Reactant's mapreduce overlay does not trace this
    z = ntuple(k -> (V[k][rm.finite[k]] .- rm.measurement[k]) ./ rm.noise[k], Val(4))
    return vcat(real.(z)..., imag.(z)...)
end

# --- Gauss–Newton product ----------------------------------------------------------------

_resid_jvp(rm, tpost, u, v) = only(
    Enzyme.autodiff(
        Enzyme.Forward, whitened_residuals, Enzyme.Const(rm), Enzyme.Const(tpost), Enzyme.Duplicated(u, v)
    )
)

_weighted_resid(rm, tpost, u, w) = sum(whitened_residuals(rm, tpost, u) .* w)

function _resid_vjp(rm, tpost, u, w)
    g = zero(u)
    Enzyme.autodiff(
        Enzyme.Reverse, _weighted_resid, Enzyme.Active,
        Enzyme.Const(rm), Enzyme.Const(tpost), Enzyme.Duplicated(u, g), Enzyme.Const(w)
    )
    return g
end

"""
    gauss_newton_product(rm::ResidualMap, tpost, u, v)

The Gauss–Newton product `Jᵀ(J v)` of the potential `Φ`, with `J` the Jacobian of
[`whitened_residuals`](@ref) at `u`.
"""
gauss_newton_product(rm, tpost, u, v) = _resid_vjp(rm, tpost, u, _resid_jvp(rm, tpost, u, v))

"""
    GaussNewtonKernels(post::VLBIPosterior)

The Gauss–Newton product on [`stdnormal_posterior`](@ref)`(post)`, compiled with Reactant
for the device. It compiles on its first call and is reused; positions and directions are
runtime inputs, so new values never recompile. `gauss_newton(k, u, v)` takes host or device
vectors and returns a device vector.

`k.tpost` is the device posterior and `k.host` the matching host posterior.
"""
struct GaussNewtonKernels{TP, TH, R}
    tpost::TP
    host::TH
    resid::R
    compiled::Dict{Any, Any}
end

function GaussNewtonKernels(post::VLBIPosterior)
    postc = ConstructionBase.setproperties(post, (; admode = nothing))
    return GaussNewtonKernels(post, Comrade.prepare_device(postc, Comrade.ComradeBase.ReactantEx()))
end

"""
    GaussNewtonKernels(post::VLBIPosterior, rpost::VLBIPosterior)

The kernels on `rpost`, an existing device copy of `post` (from `Comrade.prepare_device`).
"""
GaussNewtonKernels(post::VLBIPosterior, rpost::VLBIPosterior) = GaussNewtonKernels(
    stdnormal_posterior(rpost), stdnormal_posterior(post), ResidualMap(post), Dict{Any, Any}()
)

_device(x) = x isa Reactant.ConcreteRArray ? x : Reactant.to_rarray(collect(Float64, x))

function _kernel(f, k::GaussNewtonKernels, name, args...)
    c = get!(k.compiled, name) do
        Reactant.@compile sync = true f(k.resid, k.tpost, args...)
    end
    return c(k.resid, k.tpost, args...)
end

gauss_newton(k::GaussNewtonKernels, u, v) =
    _kernel(gauss_newton_product, k, :gauss_newton, _device(u), _device(v))

"""
    gauss_newton_curvature(k::GaussNewtonKernels)

The function `(x, W) -> H(x) W` of the Gauss–Newton operator `H` of `k` at the StdNormal
latent point `x`, applied to each column of the host matrix `W`; the curvature argument of
`Comrade.GaussNewtonLowRank`.
"""
function gauss_newton_curvature(k::GaussNewtonKernels)
    return function (x, W)
        xd = _device(x)
        HW = similar(W, Float64)
        for j in axes(W, 2)
            HW[:, j] = Array(gauss_newton(k, xd, view(W, :, j)))
        end
        return HW
    end
end

# --- averaged subspace -------------------------------------------------------------------

"""
    LikelihoodSubspace

Eigenpairs `(λ, V)` of a symmetric positive semidefinite operator restricted to the
coordinates `free`: `V` has orthonormal columns (`length(free) × length(λ)`), `λ` is sorted
in decreasing order and every `λ[i] ≥ threshold`. `n` is the full latent dimension.
"""
struct LikelihoodSubspace{T, TV <: AbstractMatrix{T}, TF <: AbstractVector{Int}}
    λ::Vector{T}
    V::TV
    free::TF
    n::Int
    threshold::T
end

Base.length(s::LikelihoodSubspace) = length(s.λ)

"""
    likelihood_subspace(op, n; rank, free = 1:n, oversample = 10, power = 2,
                        threshold = 0.1, max_basis_bytes = 2^31, rng = Random.default_rng())

The eigenpairs with eigenvalue `≥ threshold` of the symmetric positive semidefinite operator
`op(v)` (`n`-vector to `n`-vector), restricted to the coordinates `free` (the others are held
at zero), by randomized subspace iteration: `power` passes on a Gaussian block of
`rank + oversample` columns, then a Rayleigh–Ritz step. Uses `(power + 2) * (rank + oversample)`
products.

Errors if the block would exceed `max_basis_bytes`, or if more than `rank` eigenvalues exceed
`threshold` (the subspace would be truncated; raise `rank`).
"""
function likelihood_subspace(
        op, n::Integer; rank::Integer, free::AbstractVector{Int} = 1:n, oversample::Integer = 10,
        power::Integer = 2, threshold::Real = 0.1, max_basis_bytes::Integer = 2^31,
        rng::Random.AbstractRNG = Random.default_rng()
    )
    oversample >= 1 ||
        throw(ArgumentError("oversample must be at least 1 to detect a truncated spectrum"))
    m = length(free)
    ℓ = min(rank + oversample, m)
    bytes = 8 * m * ℓ
    bytes <= max_basis_bytes || throw(
        ArgumentError(
            "the subspace block needs $(Base.format_bytes(bytes)) ($m × $ℓ Float64), over " *
                "max_basis_bytes = $(Base.format_bytes(max_basis_bytes))"
        )
    )
    allunique(free) && all(i -> 1 <= i <= n, free) ||
        throw(ArgumentError("free must be distinct coordinates in 1:$n"))
    opfree = _restricted(op, n, free)
    Q = _orthonormal(_apply(opfree, randn(rng, m, ℓ)))
    for _ in 1:power
        Q = _orthonormal(_apply(opfree, Q))
    end
    B = Symmetric(Q' * _apply(opfree, Q))
    E = eigen(B; sortby = -)
    keep = findall(>=(threshold), E.values)
    length(keep) <= rank || error(
        "$(length(keep)) eigenvalues exceed threshold = $threshold, more than rank = $rank: " *
            "the subspace would be truncated; raise rank"
    )
    return LikelihoodSubspace(E.values[keep], Q * E.vectors[:, keep], free, Int(n), float(threshold))
end

function _restricted(op, n, free)
    return function (v)
        x = zeros(n)
        x[free] = v
        return op(x)[free]
    end
end

function _apply(op, X::AbstractMatrix)
    Y = similar(X)
    for j in axes(X, 2)
        Y[:, j] = op(X[:, j])
    end
    return Y
end

# Thin Q factor; `Matrix(qr(Y).Q)` would build the full square factor.
_orthonormal(Y::AbstractMatrix) = qr(Y).Q * Matrix{eltype(Y)}(I, size(Y)...)

"""
    averaged_operator(product, us)

The operator `v -> mean(product(u, v) for u in us)`, e.g. the Gauss–Newton operator averaged
over posterior draws `us` with `product = (u, v) -> Array(gauss_newton(k, u, v))`.
"""
function averaged_operator(product, us)
    isempty(us) && throw(ArgumentError("averaged_operator needs at least one draw"))
    return v -> sum(u -> product(u, v), us) ./ length(us)
end

"""
    save_subspace(path, s::LikelihoodSubspace)
    load_subspace(path) -> LikelihoodSubspace

Serialize a subspace to `path` and read it back.
"""
save_subspace(path::AbstractString, s::LikelihoodSubspace) = serialize(path, s)

function load_subspace(path::AbstractString)
    s = deserialize(path)
    s isa LikelihoodSubspace ||
        error("$path holds a $(typeof(s)), not a LikelihoodSubspace")
    return s
end

"""
    gauss_newton_subspace(k::GaussNewtonKernels, us; rank, kwargs...) -> LikelihoodSubspace

The likelihood-informed subspace of the Gauss–Newton operator of `k`, averaged over the
latent draws `us`. Keywords are those of [`likelihood_subspace`](@ref).
"""
function gauss_newton_subspace(k::GaussNewtonKernels, us; rank::Integer, kwargs...)
    n = length(first(us))
    all(u -> length(u) == n, us) || throw(DimensionMismatch("the draws differ in length"))
    op = averaged_operator((u, v) -> Array(gauss_newton(k, u, v)), us)
    t = @elapsed s = likelihood_subspace(op, n; rank, kwargs...)
    λ = s.λ
    @info "likelihood-informed subspace: rank $(length(s)) of $(length(s.free)) free coordinates " *
        "(λ from $(isempty(λ) ? "-" : @sprintf("%.3g", first(λ))) to $(isempty(λ) ? "-" : @sprintf("%.3g", last(λ)))), " *
        "basis $(Base.format_bytes(sizeof(s.V))), $(length(us)) draws, $(round(t; digits = 1)) s"
    return s
end

"""
    gauss_newton_rows(post::VLBIPosterior, band) -> Union{Nothing, Vector{Int}}

The StdNormal latent coordinates of `post` the Gauss–Newton directions may use: all of them
except the white coefficients of the stationary random sky fields at wavenumbers above `band` times
the longest baseline of the data. The coefficients of a field are its Fourier modes, and the
data constrain a field mode only through the image spectrum it is convolved with, so the
curvature on modes far above the data's band is small. `nothing` (no restriction) when every
coordinate is kept, `band` is infinite, or the sky has no stationary random fields.
"""
function gauss_newton_rows(post::VLBIPosterior, band::Real)
    isinf(band) && return nothing
    md = _sky_metadata(post)
    (hasproperty(md, :base) && md.base isa SRF) || return nothing
    plan = md.base.plan
    view = Comrade.CoordinateView(post, PT.StdNormal())
    n = dimension(view.tbase)
    θ = Comrade.transform(view.tbase, zeros(n))
    # `plan.kx` is π × the DFT frequency in cycles per pixel
    pix = abs(step(md.grid.X))
    kmag = hypot.(plan.kx, plan.ky') ./ (π * pix)
    umax = maximum(d -> maximum(hypot.(Comrade.datatable(d).baseline.U, Comrade.datatable(d).baseline.V)), post.data)
    drop = Int[]
    for X in _scaled_fields(θ.sky)
        size(θ.sky[X]) == size(kmag) || continue
        c = Comrade.coords(view, _white_field(view, X))
        append!(drop, c[vec(kmag .> band * umax)])
    end
    isempty(drop) && return nothing
    return setdiff(1:n, drop)
end
