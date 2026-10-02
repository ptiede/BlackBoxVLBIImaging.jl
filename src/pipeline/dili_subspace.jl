# The likelihood-informed subspace of the DILI sampler: the leading eigenpairs of the
# Gauss–Newton operator of the potential, averaged over posterior draws, in the StdNormal
# latent coordinates where the prior is N(0, I).

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
    dili_subspace(k::DILIKernels, us; rank, kwargs...) -> LikelihoodSubspace

The likelihood-informed subspace of the Gauss–Newton operator of `k`, averaged over the
latent draws `us`. Keywords are those of [`likelihood_subspace`](@ref).
"""
function dili_subspace(k::DILIKernels, us; rank::Integer, kwargs...)
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
