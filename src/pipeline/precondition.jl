# Low-rank affine preconditioning of the flat latent space the samplers work in.
#
# A diagonal mass matrix is a per-coordinate rescale and cannot align with posterior
# correlations that couple parameter blocks (station gain phases trading against image
# structure being the canonical VLBI example). The preconditioner supplies the missing
# rotation: NUTS samples z, and the model sees x = b + A z with
#
#     A = Diagonal(d) * (I + V * Diagonal(s .- 1) * V'),    V'V = I  (n × m, m ≪ n),
#
# so the m leading correlation directions of a pilot posterior are rescaled to unit
# width while the other n − m directions pass through untouched and stay the diagonal
# metric's job. A is square: the latent dimension is unchanged, and log|det A| =
# Σ log d + Σ log s is a constant.

"""
    LowRankPreconditioner(b, d, V, s)

Affine reparameterization `x = b .+ d .* (z + V*((s .- 1) .* (V'z)))` of a sampler's
flat latent space: `b` and `d` are the per-coordinate center and scale (length `n`),
`V` an `n × m` matrix with orthonormal columns, and `s` the `m` per-direction scale
factors applied on `span(V)`.

Pass it as the `transport_method` of `sample` (the pipeline does this when the fitting
config has a `[precondition]` section); `PT.transport_to` composes it in front of the
posterior's `asflat` transform. Build it from a pilot run with
[`fit_preconditioner`](@ref).

Fits produce plain `Float64` arrays, which serialize stably into a run's
`transport.jls` and enter a Reactant-compiled program as constants; `_device_pre`
converts them to `ConcreteRArray` buffers, which trace as runtime inputs and can be
updated in place between compiled calls (recompile-free warmup refits).
"""
# Parametric so the arrays can be host `Array`s (baked as constants under Reactant)
# or `ConcreteRArray`s (traced as runtime inputs, updatable in place between compiled
# calls — the basis of recompile-free warmup refits). Validation runs only for host
# arrays; device buffers are produced from an already-validated host fit.
struct LowRankPreconditioner{TB <: AbstractVector, TV_ <: AbstractMatrix, TS <: AbstractVector}
    b::TB
    d::TB
    V::TV_
    s::TS
    function LowRankPreconditioner(b::TB, d::TB, V::TV_, s::TS) where {TB, TV_, TS}
        n, m = size(V)
        (length(b) == n && length(d) == n && length(s) == m) || throw(
            DimensionMismatch(
                "b ($(length(b))), d ($(length(d))), V ($(size(V))), s ($(length(s))) " *
                    "are inconsistent: need length(b) == length(d) == size(V, 1) and " *
                    "length(s) == size(V, 2)"
            )
        )
        if b isa Array && V isa Array
            all(>(0), d) || throw(ArgumentError("marginal scales d must all be positive"))
            all(>(0), s) || throw(ArgumentError("direction scales s must all be positive"))
            # Zero-padded columns (used to hold the device rank slot open) carry s = 1
            # and contribute nothing; only active columns must be orthonormal.
            act = findall(!=(1.0), s)
            isempty(act) || opnorm(V[:, act]' * V[:, act] - I) < 1.0e-6 ||
                throw(ArgumentError("V must have orthonormal columns"))
        end
        return new{TB, TV_, TS}(b, d, V, s)
    end
end

function Base.show(io::IO, p::LowRankPreconditioner)
    return print(
        io, "LowRankPreconditioner(n = $(length(p.b)), rank = $(length(p.s)), " *
            "scales = $(round.(p.s; digits = 2)))"
    )
end

_affine_logdet(p::LowRankPreconditioner) = sum(log, p.d) + sum(log, p.s; init = 0.0)

function _affine_fwd(p::LowRankPreconditioner, z::AbstractVector)
    isempty(p.s) && return p.b .+ p.d .* z
    return p.b .+ p.d .* (z .+ p.V * ((p.s .- 1) .* (p.V' * z)))
end

function _affine_inv(p::LowRankPreconditioner, x::AbstractVector)
    w = (x .- p.b) ./ p.d
    isempty(p.s) && return w
    # (I + V (S - I) V')⁻¹ = I + V (S⁻¹ - I) V' since V is orthonormal.
    return w .+ p.V * ((inv.(p.s) .- 1) .* (p.V' * w))
end

"""
    AnglePairPreconditioner(pre, i1, i2, G, H, ld)

A [`LowRankPreconditioner`](@ref) with an extra per-pair 2×2 stage for the latent
(sin, cos) pairs of angle-embedded parameters: `x = _affine_fwd(pre, G z)`, where `G`
applies a constant 2×2 block to each pair `(i1[k], i2[k])` and the identity elsewhere.

Each block is `D⁻¹ C^{1/2}` with `D` the marginal scales `pre.d` on the pair and `C`
the analytic wedge covariance of the pair's true latents — tangential variance from the
pilot draws' circular spread, radial variance from the angle transform's log-normal
radius pseudo-prior, rotated by the mean angle. This aligns and correctly scales the
thin rotated ellipse of every well-measured phase, which no diagonal metric can do
(the tight axis sits at the mean angle) and which the draw-fit low-rank stage cannot
see (draws inverted through wrapped parameters collapse the radius, hiding the pair's
true 2D geometry).

`G` blocks are stored row-major per pair in `G[:, k] = [g11, g12, g21, g22]` and `H`
holds the precomputed inverses; the log-determinant is computed lazily from `G`.
"""
struct AnglePairPreconditioner{P <: LowRankPreconditioner, TG <: AbstractMatrix}
    pre::P
    i1::Vector{Int}
    i2::Vector{Int}
    G::TG
    H::TG
    function AnglePairPreconditioner{P, TG}(pre, i1, i2, G, H) where {P, TG}
        m = length(i1)
        (length(i2) == m && size(G) == (4, m) && size(H) == (4, m)) ||
            throw(DimensionMismatch("inconsistent pair-block arrays"))
        all(i -> 1 <= i <= length(pre.b), i1) && all(i -> 1 <= i <= length(pre.b), i2) ||
            throw(ArgumentError("pair indices out of range"))
        return new{P, TG}(pre, i1, i2, G, H)
    end
end
AnglePairPreconditioner(pre::P, i1, i2, G::TG, H::TG) where {P <: LowRankPreconditioner, TG} =
    AnglePairPreconditioner{P, TG}(pre, i1, i2, G, H)
# Compatibility with the older 6-argument form (cached logdet); the value is now lazy.
AnglePairPreconditioner(pre, i1, i2, G, H, ld) = AnglePairPreconditioner(pre, i1, i2, G, H)

function Base.show(io::IO, p::AnglePairPreconditioner)
    return print(io, "AnglePairPreconditioner(npairs = $(length(p.i1)), inner = $(p.pre))")
end

# Apply the 2×2 blocks (rows of M = 4 × npairs) to the pairs of z; identity elsewhere.
# Vector-indexed gather/scatter so the traced form lowers to vectorized ops.
function _pairs_apply(i1, i2, M, z)
    y = copy(z)
    z1 = z[i1]
    z2 = z[i2]
    y[i1] = M[1, :] .* z1 .+ M[2, :] .* z2
    y[i2] = M[3, :] .* z1 .+ M[4, :] .* z2
    return y
end

function _affine_logdet(p::AnglePairPreconditioner)
    lg = sum(log.(abs.(p.G[1, :] .* p.G[4, :] .- p.G[2, :] .* p.G[3, :])); init = 0.0)
    return _affine_logdet(p.pre) + lg
end
_affine_fwd(p::AnglePairPreconditioner, z::AbstractVector) =
    _affine_fwd(p.pre, _pairs_apply(p.i1, p.i2, p.G, z))
_affine_inv(p::AnglePairPreconditioner, x::AbstractVector) =
    _pairs_apply(p.i1, p.i2, p.H, _affine_inv(p.pre, x))

"""
    StiffStage(pre, V, s)

An extra low-rank stage acting directly on the sampled latent `z`, before everything in
`pre`: `x = _affine_fwd(pre, z + V (diag(s) - I) Vᵀ z)` with `V` orthonormal and
`s .< 1`. This is the stiff-direction mirror of the draw-fit wide directions: its `V`
comes from the top eigendirections of the *gradient* covariance in z-space (a direction
with gradient variance λ has conditional width `1/√λ`; the correction scales it by
`s = 1/√λ` so it becomes unit). Rotated stiff directions cap the leapfrog step size and
are invisible both to any diagonal metric and to draw-based fits (a short chain cannot
resolve widths below its own sampling noise, and the smallest sample-covariance
eigenvalues are unestimable at `ndraws < n`); gradients measure them directly.
"""
struct StiffStage{P, TV_ <: AbstractMatrix, TS <: AbstractVector}
    pre::P
    V::TV_
    s::TS
    function StiffStage{P, TV_, TS}(pre, V::TV_, s::TS) where {P, TV_, TS}
        size(V, 2) == length(s) || throw(DimensionMismatch("V and s are inconsistent"))
        if V isa Array
            all(x -> 0 < x, s) || throw(ArgumentError("stiff scales must be positive"))
            isempty(s) || opnorm(V' * V - I) < 1.0e-6 ||
                throw(ArgumentError("V must have orthonormal columns"))
        end
        return new{P, TV_, TS}(pre, V, s)
    end
end
StiffStage(pre::P, V::TV_, s::TS) where {P, TV_, TS} = StiffStage{P, TV_, TS}(pre, V, s)

function Base.show(io::IO, p::StiffStage)
    return print(
        io, "StiffStage(nstiff = $(length(p.s)), " *
            "widths = $(round.(p.s; sigdigits = 2)), inner = $(p.pre))"
    )
end

_affine_logdet(p::StiffStage) = _affine_logdet(p.pre) + sum(log, p.s; init = 0.0)
_affine_fwd(p::StiffStage, z::AbstractVector) =
    _affine_fwd(p.pre, z .+ p.V * ((p.s .- 1) .* (p.V' * z)))
function _affine_inv(p::StiffStage, x::AbstractVector)
    w = _affine_inv(p.pre, x)
    return w .+ p.V * ((inv.(p.s) .- 1) .* (p.V' * w))
end

# Host mirrors for device-buffered stages. The sampler's host-side bookkeeping
# (logging the constrained draw each chunk, inverting positions for transfer) pushes
# plain host arrays through the transform; when its buffers live on the device they
# must be brought back to host for those calls, while traced/tracing calls must see
# the device buffers untouched. Dispatch on the INPUT array: plain host arrays get a
# hostified transform, everything else passes through.
_hostify(x::AbstractArray) = x isa Array ? x : Array(x)
_hostify(p::LowRankPreconditioner) =
    LowRankPreconditioner(_hostify(p.b), _hostify(p.d), _hostify(p.V), _hostify(p.s))
_hostify(p::AnglePairPreconditioner) =
    AnglePairPreconditioner(_hostify(p.pre), p.i1, p.i2, _hostify(p.G), _hostify(p.H))
_hostify(p::StiffStage) = StiffStage(_hostify(p.pre), _hostify(p.V), _hostify(p.s))
_devicebuffers(p::LowRankPreconditioner) = !(p.b isa Array)
_devicebuffers(p::AnglePairPreconditioner) = _devicebuffers(p.pre)
_devicebuffers(p::StiffStage) = _devicebuffers(p.pre)
_ishostvec(z) = z isa Array || (z isa SubArray && parent(z) isa Array)
_pre_for(p, z) = (_ishostvec(z) && _devicebuffers(p)) ? _hostify(p) : p

# The flat-transform node: the affine acts in LATENT space, before the inner transform
# (the mirror image of PT's `PushforwardTransform`, whose map acts after). The inner is
# the posterior's whole `asflat` node, so this consumes exactly `dimension(inner)`
# latent coordinates.
struct PreconditionedFlat{P, I <: TV.AbstractTransform} <: TV.VectorTransform
    pre::P
    inner::I
end

TV.dimension(t::PreconditionedFlat) = TV.dimension(t.inner)
PT.is_scalar_transport(t::PreconditionedFlat) = PT.is_scalar_transport(t.inner)

function TV.transform_with(
        flag::TV.LogJacFlag, t::PreconditionedFlat, z::AbstractVector, index
    )
    n = TV.dimension(t.inner)
    pre = _pre_for(t.pre, z)
    w = _affine_fwd(pre, view(z, index:(index + n - 1)))
    x, ℓi, _ = TV.transform_with(flag, t.inner, w, firstindex(w))
    flag isa TV.NoLogJac && return x, ℓi, index + n
    return x, ℓi + _affine_logdet(pre), index + n
end

TV.inverse_eltype(t::PreconditionedFlat, ::Type{T}) where {T} =
    TV.inverse_eltype(t.inner, T)

function TV.inverse_at!(z::AbstractVector, index, t::PreconditionedFlat, x)
    n = TV.dimension(t.inner)
    index′ = TV.inverse_at!(z, index, t.inner, x)
    zv = view(z, index:(index + n - 1))
    copyto!(zv, _affine_inv(_pre_for(t.pre, z), zv))
    return index′
end

# Lets `maybe_transport`/`resolve_disk_transport` accept the preconditioner as a latent
# "space": the sampled space is the flat one with the affine composed in front, and the
# result stays a `TransformedVLBIPosterior`, so every sampler/AD extension method
# dispatching on that concrete type keeps working.
_pre_dim(space::LowRankPreconditioner) = length(space.b)
_pre_dim(space::AnglePairPreconditioner) = length(space.pre.b)
_pre_dim(space::StiffStage) = _pre_dim(space.pre)

function PT.transport_to(
        post::Comrade.VLBIPosterior,
        space::Union{LowRankPreconditioner, AnglePairPreconditioner, StiffStage}
    )
    t0 = asflat(post).transform
    node0 = PT.transport_node(t0)
    n = TV.dimension(node0)
    _pre_dim(space) == n || throw(
        DimensionMismatch(
            "preconditioner dimension $(_pre_dim(space)) does not match the posterior " *
                "latent dimension $n — it was fit to a different model"
        )
    )
    node = PreconditionedFlat(space, node0)
    td = PT.TransportedDistribution(node, getfield(t0, :start), nothing)
    return Comrade.TransformedVLBIPosterior(post, td)
end

"""
    _lowrank_from_draws(Z; rank, min_scale=nothing) -> LowRankPreconditioner

Build the preconditioner from a matrix of flat-space posterior draws (`n × ndraws`,
one draw per column): center and scale by the per-coordinate mean and standard
deviation, then keep up to `rank` leading eigendirections of the sample correlation
matrix, with cross-validated per-direction scales.

Directions and scales come from a chain-ordered split-half cross-validation.
Candidate directions are the eigendirections of the FIRST half's sample correlation
whose eigenvalues clear the Marchenko-Pastur bulk edge `(1 + √(2n/ndraws))²`. Their
in-sample scales `√λ` are inflated — by finite draws and, far more, by chain
autocorrelation — so each candidate's scale is instead the standard deviation of the
held-out SECOND half projected onto it: exactly the width the correction divides out,
eigenvector misalignment included. A false candidate cross-validates to `s ≈ 1` and a
poorly estimated one to less than its true width, so scarce or autocorrelated draws
under-correct, never over-correct.

Directions with cross-validated `s` at or below `min_scale` (default 1.5 —
corrections that small are not worth a column) are dropped, as are directions whose
full-chain projected trace trends monotonically with draw index (|correlation| >
`max_trend`): the chain was still in transit along those, so no window estimates
their width. The columns of `Z` must therefore be in chain order.
"""
function _lowrank_from_draws(
        Z::AbstractMatrix; rank::Int, min_scale::Union{Nothing, Real} = nothing,
        max_trend::Real = 0.7, carry::Union{Nothing, LowRankPreconditioner} = nothing
    )
    n, ndraws = size(Z)
    ndraws >= 20 ||
        throw(ArgumentError("need at least 20 draws for the split-half fit, got $ndraws"))
    b = vec(mean(Z; dims = 2))
    σ = vec(std(Z; dims = 2))
    all(>(0), σ) || throw(
        ArgumentError(
            "$(count(iszero, σ)) latent coordinates have zero variance across the " *
                "pilot draws; the pilot chain did not move in them"
        )
    )
    W = (Z .- b) ./ σ                               # standardized draws, chain order
    h = ndraws ÷ 2
    # Carried directions from an earlier round (`carry`): re-expressed in this fit's
    # standardization, orthonormalized, re-scaled on the held-out half, and DEFLATED out
    # of the draws before detection. Detection from a short log only resolves the widest
    # directions, and each fit grabs a noise-selected subset of a much larger soft
    # subspace — without deflation a refit forgets what the previous round corrected.
    # Carried directions skip the floor and the drift guard: they were validated once,
    # and the re-measured scale is floored at 1 (a no-op at worst, never harmful).
    U0 = zeros(n, 0)
    s0 = Float64[]
    if carry !== nothing && !isempty(carry.s)
        length(carry.b) == n || throw(
            DimensionMismatch(
                "carried preconditioner has dimension $(length(carry.b)), draws have $n"
            )
        )
        m0 = length(carry.s)
        U0 = Matrix(qr((carry.d .* carry.V) ./ σ).Q)[:, 1:m0]
        s0 = vec(max.(std(U0' * view(W, :, (h + 1):ndraws); dims = 2), 1.0))
        W = W .- U0 * (U0' * W)
    end
    Ya = W[:, 1:h] ./ sqrt(h - 1)
    F = eigen(Symmetric(Ya' * Ya))                  # ascending; eigvals = correlation λ
    λ = reverse(F.values)
    U = reverse(F.vectors; dims = 2)
    edge = (1 + sqrt(n / h))^2
    # Candidates: first-half eigenvalues above the noise bulk. Twice `rank` are carried
    # so cross-validation, not the biased in-sample ordering, decides which survive.
    ncand = min(2 * rank, count(>(edge), λ))
    if ncand == 0
        isempty(s0) && @warn "no correlation direction exceeds the noise edge " *
            "√λ = $(sqrt(edge)) (largest is $(sqrt(max(first(λ), 0.0)))); returning " *
            "a diagonal-only preconditioner"
        return LowRankPreconditioner(b, σ, U0, s0)
    end
    cand = 1:ncand
    V = Matrix(qr(Ya * (U[:, cand] ./ sqrt.(λ[cand])')).Q)[:, cand]
    # Cross-validated scales: the held-out half's spread along each direction.
    s = [std(vec(V[:, j]' * view(W, :, (h + 1):ndraws))) for j in cand]
    # The floor drops corrections too small to be worth a column; cross-validation has
    # already sent false candidates to s ≈ 1, well below it.
    smin = isnothing(min_scale) ? 1.5 : Float64(min_scale)
    # Drift guard: a strong monotone trend in a direction's full-chain trace means the
    # chain was still in transit along it (burn-in, or a mode slower than the window),
    # so no window estimates its width — over-correcting is as costly as the
    # uncorrected ridge. Such directions are left to the diagonal metric; a later
    # refit from a longer chain picks them up.
    tidx = collect(1:ndraws)
    trending = [abs(cor(vec(V[:, j]' * W), tidx)) > max_trend for j in cand]
    ndrift = count(trending .& (s .> smin))
    ndrift > 0 && @warn "dropped $ndrift trending direction(s) " *
        "(scales $(round.(s[trending .& (s .> smin)]; digits = 1))): the pilot chain " *
        "drifts along them, so their widths cannot be estimated from this window"
    keep = [j for j in cand if s[j] > smin && !trending[j]]
    keep = keep[sortperm(s[keep]; rev = true)]      # cross-validated order, not λ order
    keep = keep[1:min(max(rank - length(s0), 0), length(keep))]   # rank caps the TOTAL
    if isempty(keep)
        isempty(s0) && @warn "no candidate direction survives cross-validation; " *
            "returning a diagonal-only preconditioner — the pilot chain is too short " *
            "or still burning in, let it run longer"
        return LowRankPreconditioner(b, σ, U0, s0)
    end
    # New directions come from the deflated draws, so they are orthogonal to the
    # carried set and the union stays orthonormal.
    return LowRankPreconditioner(b, σ, hcat(U0, V[:, keep]), vcat(s0, s[keep]))
end

"""
    _device_pre(pre; rank_cap) -> preconditioner with ConcreteRArray buffers

Convert a host-fitted preconditioner to device buffers, padding the low-rank stage to
`rank_cap` columns (zero columns with `s = 1` are exact no-ops). Device buffers trace
as runtime inputs of a compiled program, so a later refit updates them in place with
[`_update_device_pre!`](@ref) — no recompile — as long as the shapes (and, for the
pair stage, the pair index set) stay fixed.
"""
function _device_pre(pre::LowRankPreconditioner; rank_cap::Int = max(size(pre.V, 2), 1))
    n, m = size(pre.V)
    m <= rank_cap || throw(ArgumentError("rank_cap $rank_cap below fitted rank $m"))
    V = zeros(n, rank_cap)
    V[:, 1:m] = pre.V
    s = ones(rank_cap)
    s[1:m] = pre.s
    return LowRankPreconditioner(
        Reactant.to_rarray(copy(pre.b)), Reactant.to_rarray(copy(pre.d)),
        Reactant.to_rarray(V), Reactant.to_rarray(s)
    )
end

function _device_pre(p::AnglePairPreconditioner; rank_cap::Int = max(size(p.pre.V, 2), 1))
    return AnglePairPreconditioner(
        _device_pre(p.pre; rank_cap), p.i1, p.i2,
        Reactant.to_rarray(copy(p.G)), Reactant.to_rarray(copy(p.H))
    )
end

"""
    _update_device_pre!(dev, h)

Copy a fresh host fit `h` into the live device buffers `dev` (shapes fixed at
[`_device_pre`](@ref) time; the pair index set must be identical — freeze it with
`fit_preconditioner`'s `pair_set`).
"""
function _update_device_pre!(dev::LowRankPreconditioner, h::LowRankPreconditioner)
    n, cap = size(dev.V)
    m = size(h.V, 2)
    m <= cap || throw(ArgumentError("refit rank $m exceeds the device rank cap $cap"))
    V = zeros(n, cap)
    V[:, 1:m] = h.V
    s = ones(cap)
    s[1:m] = h.s
    copyto!(dev.b, h.b)
    copyto!(dev.d, h.d)
    copyto!(dev.V, V)
    copyto!(dev.s, s)
    return dev
end

function _update_device_pre!(dev::AnglePairPreconditioner, h::AnglePairPreconditioner)
    (dev.i1 == h.i1 && dev.i2 == h.i2) || throw(
        ArgumentError(
            "pair set changed across refits; device pair indices are compile-time " *
                "constants — freeze the set with fit_preconditioner's pair_set"
        )
    )
    _update_device_pre!(dev.pre, h.pre)
    copyto!(dev.G, h.G)
    copyto!(dev.H, h.H)
    return dev
end

"""
    _new_refit_cache() -> NamedTuple

Caches reused across warmup refits: `draws`/`scores` map a warmup-store draw index to
its inverted flat position and flat-space score (both invariant across refits — the
flat space never changes), and `fn` holds the once-compiled device score function and
its device posterior. Without these every refit recompiles the score program and
re-inverts the full window; with them a refit costs seconds.
"""
_new_refit_cache() = (;
    draws = Dict{Int, Vector{Float64}}(),
    scores = Dict{Int, Vector{Float64}}(),
    fn = Ref{Any}(nothing),
)

# One flat-space score, through the cache's compiled function (built on first use).
function _flat_score!(cache, post::Comrade.VLBIPosterior, x::Vector{Float64}; reactant::Bool)
    if cache.fn[] === nothing
        if reactant
            dpost = Comrade.prepare_device(post, Comrade.ComradeBase.ReactantEx())
            tflat = asflat(dpost)
            xr = Reactant.to_rarray(x)
            vg = Reactant.@compile sync = true _reactant_value_and_grad(tflat, xr)
            cache.fn[] = (; tflat, vg, reactant = true)
        else
            cache.fn[] = (; tflat = asflat(post), vg = nothing, reactant = false)
        end
    end
    c = cache.fn[]
    if c.reactant
        g, _ = c.vg(c.tflat, Reactant.to_rarray(x))
        return Array(g)
    end
    _, g = Comrade.LogDensityProblems.logdensity_and_gradient(c.tflat, x)
    return g
end

# Scores of the CANONICAL flat posterior at the given draw columns — the Fisher fit
# works in flat space directly (draws and scores in the same coordinates), unlike the
# staged fits, which measure gradients in the currently-sampled z-space.
function _collect_flat_grads(post::Comrade.VLBIPosterior, Z::AbstractMatrix, cols; reactant::Bool)
    G = zeros(size(Z, 1), length(cols))
    if reactant
        dpost = Comrade.prepare_device(post, Comrade.ComradeBase.ReactantEx())
        tflat = asflat(dpost)
        z1 = Reactant.to_rarray(collect(view(Z, :, first(cols))))
        vg = Reactant.@compile sync = true _reactant_value_and_grad(tflat, z1)
        for (k, j) in enumerate(cols)
            g, _ = vg(tflat, Reactant.to_rarray(collect(view(Z, :, j))))
            G[:, k] = Array(g)
        end
    else
        tflat = asflat(post)
        for (k, j) in enumerate(cols)
            _, g = Comrade.LogDensityProblems.logdensity_and_gradient(tflat, collect(view(Z, :, j)))
            G[:, k] = g
        end
    end
    return G
end

"""
    _score_init_pre(post, θ0, cache; reactant) -> LowRankPreconditioner

nutpie-style first-score initialization (Seyboldt, Carlson & Carpenter 2026, §3.1):
before ANY draws exist, one score evaluation at the start point sets the diagonal
scales `d = 1/√|α⁰|` (the one-sample estimate of the conditional widths — the mass
matrix `diag(|α⁰|)`, which also makes the sampler scale-free in the parameterization).
The start point becomes the center, so warmup segment 0 begins pre-scaled instead of
on a unit metric; the first windowed refit then replaces this with the full Fisher
fit. The score comes through the refit cache, so the compiled score program is shared
with every later refit.
"""
function _score_init_pre(post::Comrade.VLBIPosterior, θ0, cache; reactant::Bool)
    x0 = Comrade.inverse(asflat(post), θ0)
    g = _flat_score!(cache, post, x0; reactant)
    d = clamp.(inv.(sqrt.(abs.(g) .+ 1.0e-8)), 1.0e-4, 1.0e4)
    return LowRankPreconditioner(x0, d, zeros(length(x0), 0), Float64[])
end

# Symmetric-positive-definite geometric mean: the Σ solving Σ A Σ = B
# (Seyboldt, Carlson & Carpenter 2026, Algorithm 2).
function _spdm(A::Symmetric, B::Symmetric)
    FA = eigen(A)
    Ah = FA.vectors * Diagonal(sqrt.(FA.values)) * FA.vectors'
    Aih = FA.vectors * Diagonal(inv.(sqrt.(FA.values))) * FA.vectors'
    FM = eigen(Symmetric(Ah * B * Ah))
    Mh = FM.vectors * Diagonal(sqrt.(max.(FM.values, 0.0))) * FM.vectors'
    return Symmetric(Aih * Mh * Aih)
end

"""
    _fisher_lowrank(Z, G; rank, cutoff = 2.0, γ = 1e-5) -> LowRankPreconditioner

The Fisher-divergence estimator of Seyboldt, Carlson & Carpenter (arXiv:2603.18845,
Algorithm 1): fit the low-rank-plus-diagonal affine transform minimizing the sample
Fisher divergence from the transformed density to a standard normal, using the flat
draws `Z` AND their scores `G = ∇ log p` (n × N, same columns). Scores let the
estimator bypass the Cramér–Rao limit of draw-only estimation — on the subspace
spanned by draws and scores a Gaussian target is matched exactly — so short windows
carry real information and no sampling-noise thresholds are needed there.

The construction: diagonal scales are the coordinatewise geometric mean
`σ = √(var(x)/var(α))` (marginal vs conditional width — the closed-form diagonal
minimizer); the low-rank part comes from eigendecomposing the SPD geometric mean of
the projected draw covariance and inverse score covariance on the joint
draw-plus-score subspace, keeping eigenvalues `λ ≥ cutoff` or `≤ 1/cutoff` (wide AND
stiff directions in one two-sided criterion, coupling included), up to `rank` by
`|log λ|`. Returns the standard [`LowRankPreconditioner`](@ref) with `s = √λ`.
"""
function _fisher_lowrank(
        Z::AbstractMatrix, G::AbstractMatrix;
        rank::Int, cutoff::Real = 2.0, γ::Real = 1.0e-5, carry = nothing
    )
    n, N = size(Z)
    size(G) == (n, N) || throw(DimensionMismatch("draws and scores must align"))
    N >= 4 || throw(ArgumentError("need at least 4 draws, got $N"))
    vz = vec(var(Z; dims = 2))
    vg = vec(var(G; dims = 2))
    (all(>(0), vz) && all(>(0), vg)) || throw(
        ArgumentError("zero-variance coordinates in draws or scores")
    )
    σ = (vz ./ vg) .^ (1 // 4)                      # σ*² = √(var(x)/var(α))
    x̄ = vec(mean(Z; dims = 2))
    # The estimation is centered at the draw mean (Algorithm 1); the returned
    # transform's SHIFT is the score-informed center of Thm 2.2, μ* = x̄ + σ*² ⊙ ᾱ —
    # the mean score pulls it toward the mode, past the Cramér–Rao limit of x̄ alone.
    b = x̄ .+ σ .^ 2 .* vec(mean(G; dims = 2))
    X = (Z .- x̄) ./ σ                               # standardized draws
    A = (G .- mean(G; dims = 2)) .* σ               # standardized scores (contravariant)
    X0h = copy(X); A0h = copy(A)                    # pre-deflation, for carried re-scale
    # Directions from the FIRST half; scales validated on the held-out second half.
    # With N ≪ n the sample covariances are rank-deficient on the joint subspace, and
    # directions spanned by only one of {draws, scores} acquire regularizer-dominated
    # eigenvalues that pass the two-sided filter; on held-out data such directions
    # validate to scale ≈ 1 and are dropped, while true directions get an honest
    # held-out scale (var_x/var_α)^(1/4) along the estimated axis.
    # Accumulation: carry the previous transform's directions forward so a direction
    # once found is never lost when a later (still-stuck) window fails to see it. The
    # carried directions are re-expressed in the CURRENT standardization, deflated out
    # of the draws/scores so the residual fit spends its budget on NEW structure, and
    # re-scaled on the held-out data — keeping the wider of the carried and re-measured
    # scale for wide directions (a wide mode only ever gets wider as the chain frees).
    h = N ÷ 2
    U0 = zeros(n, 0); s0 = Float64[]
    if carry !== nothing && !isempty(carry.s)
        U0 = Matrix(qr((carry.d .* carry.V) ./ σ).Q)[:, 1:length(carry.s)]
        # Re-scale carried directions on the held-out half; keep the WIDER of carried
        # and re-measured for wide directions (a found ridge never shrinks), the fresh
        # value for stiff ones. Computed on the pre-deflation projections.
        Xb0 = view(X0h, :, (h + 1):N); Ab0 = view(A0h, :, (h + 1):N)
        s0now = (vec(var(U0' * Xb0; dims = 2)) ./ vec(var(U0' * Ab0; dims = 2))) .^ (1 // 4)
        s0 = map((sc, sn) -> sc > 1 ? max(sc, sn) : sn, carry.s, s0now)
        X .-= U0 * (U0' * X)
        A .-= U0 * (U0' * A)
    end
    Xa = X[:, 1:h]
    Aa = A[:, 1:h]
    Q = Matrix(qr(hcat(Xa, Aa)).Q)[:, 1:min(2h, n)]
    Px = Q' * Xa
    Pa = Q' * Aa
    Cx = Symmetric(Px * Px' ./ (h - 1) + γ * I)
    Ca = Symmetric(Pa * Pa' ./ (h - 1) + γ * I)
    Σ = _spdm(Ca, Cx)                               # Σ Ca Σ = Cx
    F = eigen(Σ)
    λ = F.values
    cand = findall(l -> l >= cutoff || l <= inv(cutoff), λ)
    isempty(cand) && return LowRankPreconditioner(b, σ, hcat(U0), vcat(s0))
    U = Q * F.vectors[:, cand]
    U = Matrix(qr(U).Q)[:, 1:length(cand)]          # re-orthonormalize after projection
    Xb = view(X, :, (h + 1):N)
    Ab = view(A, :, (h + 1):N)
    vX = vec(var(U' * Xb; dims = 2))
    vA = vec(var(U' * Ab; dims = 2))
    s = (vX ./ vA) .^ (1 // 4)
    sc = sqrt(cutoff)
    keep = [
        j for j in eachindex(cand) if
            (s[j] >= sc || s[j] <= inv(sc)) && (s[j] > 1) == (λ[cand[j]] > 1)
    ]
    if isempty(keep) && isempty(U0)
        return LowRankPreconditioner(b, σ, zeros(n, 0), Float64[])
    end
    keep = keep[sortperm(abs.(log.(s[keep])); rev = true)]
    # `rank` bounds per-fit cost; the device cap grows to match (see the imager refit
    # closure), so this only bites at the configured ceiling. A hit is logged there.
    keep = keep[1:min(max(rank - length(s0), 0), length(keep))]
    return LowRankPreconditioner(b, σ, hcat(U0, U[:, keep]), vcat(s0, s[keep]))
end

"""
    _angle_pairs_from_draws(Z, pre; rrad = 0.262, tmax = 0.5) -> AnglePairPreconditioner or pre

Detect the latent (sin, cos) pairs of angle-embedded parameters in the flat draws `Z`
(they lie exactly on the unit circle in every draw inverted through a wrapped
transform) and wrap `pre` with per-pair 2×2 blocks that align and scale each pair's
wedge geometry: tangential width from the draws' circular spread (floored at 0.005 —
an under-mixed trace under-measures the spread, which errs on the benign,
under-correcting side), radial width `rrad` from the angle transform's log-normal
radius pseudo-prior. Pairs with circular spread above `tmax` are skipped: a
near-uniform phase fills the ring, and no linear map helps a curved valley. Returns
`pre` unchanged when nothing qualifies.
"""
function _angle_pairs_from_draws(
        Z::AbstractMatrix, pre::LowRankPreconditioner;
        rrad::Real = 0.262, tmax::Real = 0.5,
        pair_set::Union{Nothing, Tuple{Vector{Int}, Vector{Int}}} = nothing
    )
    n, N = size(Z)
    i1 = Int[]
    i2 = Int[]
    G = Float64[]
    H = Float64[]
    ld = 0.0
    # With `pair_set` given (device buffers hold a frozen pair layout across refits),
    # detection is skipped and the given pairs are refit unconditionally, widths
    # clamped into (0.005, tmax].
    detect = isnothing(pair_set)
    queue = detect ? (1:0) : eachindex(pair_set[1])
    i = 1
    k = 0
    while (detect && i < n) || (!detect && k < length(pair_set[1]))
        if !detect
            k += 1
            i = pair_set[1][k]
        end
        oncircle = all(j -> abs(Z[i, j]^2 + Z[i + 1, j]^2 - 1) < 1.0e-8, 1:N)
        if !oncircle
            detect || throw(ArgumentError("frozen pair at index $i is not an angle pair"))
            i += 1
            continue
        end
        θ = atan.(view(Z, i, :), view(Z, i + 1, :))
        m = atan(mean(sin.(θ)), mean(cos.(θ)))
        t = std(rem2pi.(θ .- m, RoundNearest))
        if detect && t > tmax
            i += 2
            continue
        end
        t = clamp(t, 0.005, tmax)
        # wedge covariance square root: tangential t along (cos m, −sin m), radial
        # rrad along (sin m, cos m); C^{1/2} = t·u_t u_tᵀ + rrad·u_r u_rᵀ
        ut1, ut2 = cos(m), -sin(m)
        ur1, ur2 = sin(m), cos(m)
        c11 = t * ut1^2 + rrad * ur1^2
        c12 = t * ut1 * ut2 + rrad * ur1 * ur2
        c22 = t * ut2^2 + rrad * ur2^2
        # G = D⁻¹ C^{1/2}: the block also repairs the marginal scales, which the
        # radius-collapsed draws under-measure
        g11 = c11 / pre.d[i]
        g12 = c12 / pre.d[i]
        g21 = c12 / pre.d[i + 1]
        g22 = c22 / pre.d[i + 1]
        dt = g11 * g22 - g12 * g21
        push!(i1, i)
        push!(i2, i + 1)
        append!(G, (g11, g12, g21, g22))
        append!(H, (g22 / dt, -g12 / dt, -g21 / dt, g11 / dt))
        ld += log(abs(dt))
        detect && (i += 2)
    end
    isempty(i1) && return pre
    return AnglePairPreconditioner(pre, i1, i2, reshape(G, 4, :), reshape(H, 4, :), ld)
end

# Evaluate the z-space log-density gradient at the given draw columns of Z (each
# inverted through `pre`), on the device (compile once, then ~ms per draw — the device
# posterior is transient, the same pattern as the Reactant benchmarks) or on the
# CPU/Enzyme path. Columns are returned in the order of `cols`, so chain order is
# preserved for split-half use.
function _collect_grads(pre, post::Comrade.VLBIPosterior, Z::AbstractMatrix, cols; reactant::Bool)
    G = zeros(size(Z, 1), length(cols))
    if reactant
        dpost = Comrade.prepare_device(post, Comrade.ComradeBase.ReactantEx())
        tpre = Comrade.maybe_transport(dpost, pre)
        z1 = Reactant.to_rarray(_affine_inv(pre, view(Z, :, first(cols))))
        vg = Reactant.@compile sync = true _reactant_value_and_grad(tpre, z1)
        for (k, j) in enumerate(cols)
            zr = Reactant.to_rarray(_affine_inv(pre, view(Z, :, j)))
            g, _ = vg(tpre, zr)
            G[:, k] = Array(g)
        end
    else
        tpre = Comrade.maybe_transport(post, pre)
        for (k, j) in enumerate(cols)
            z = _affine_inv(pre, view(Z, :, j))
            _, g = Comrade.LogDensityProblems.logdensity_and_gradient(tpre, z)
            G[:, k] = g
        end
    end
    return G
end

"""
    _stiff_from_grads(G, pre; rank = 32, min_stiff = 1.5) -> StiffStage or pre

Fit stiff-direction corrections from z-space gradient draws `G` (n × ngrad, chain
order). The gradient covariance's top eigendirections are the posterior's conditionally
tightest axes (gradient std `√λ` along a direction means conditional width `1/√λ`);
these rotated directions cap the leapfrog step size and are invisible to diagonals and
to draw-based fits. Split-half as everywhere: directions from the first half's
eigenvalues above the Marchenko-Pastur bulk (scaled by the bulk gradient variance),
widths cross-validated as `1/std` of the held-out half's projections. Only directions
meaningfully stiffer than the bulk (`std > min_stiff·√bulk`) are kept, up to `rank`.
"""
function _stiff_from_grads(
        G::AbstractMatrix, pre; rank::Int = 32, min_stiff::Real = 1.5
    )
    n, N = size(G)
    N >= 20 || throw(ArgumentError("need at least 20 gradient draws, got $N"))
    Gc = G .- mean(G; dims = 2)
    h = N ÷ 2
    Ya = Gc[:, 1:h] ./ sqrt(h - 1)
    F = eigen(Symmetric(Ya' * Ya))
    λ = reverse(F.values)
    U = reverse(F.vectors; dims = 2)
    β = median(vec(mean(abs2, Gc[:, 1:h]; dims = 2)))   # bulk gradient variance
    edge = β * (1 + sqrt(n / h))^2
    ncand = min(2 * rank, count(>(edge), λ))
    if ncand == 0
        @warn "no stiff direction exceeds the gradient noise edge; skipping stiff stage"
        return pre
    end
    cand = 1:ncand
    V = Matrix(qr(Ya * (U[:, cand] ./ sqrt.(λ[cand])')).Q)[:, cand]
    gstd = [std(vec(V[:, j]' * view(Gc, :, (h + 1):N))) for j in cand]
    keep = [j for j in cand if gstd[j] > min_stiff * sqrt(β)]
    keep = keep[sortperm(gstd[keep]; rev = true)][1:min(rank, length(keep))]
    if isempty(keep)
        @warn "no stiff direction survives cross-validation; skipping stiff stage"
        return pre
    end
    s = clamp.(1 ./ gstd[keep], 0.002, 1.0)
    return StiffStage(pre, V[:, keep], s)
end

# Divide the map's response on coordinate i by f[i]: plain coordinates through the
# diagonal, angle-pair coordinates through their block rows (whose scales are set by
# the block, not by d). This is how the gradient balance below is applied.
function _scale_rows(pre::LowRankPreconditioner, f::AbstractVector)
    return LowRankPreconditioner(pre.b, pre.d ./ f, pre.V, pre.s)
end

function _scale_rows(p::AnglePairPreconditioner, f::AbstractVector)
    fin = copy(f)
    fin[p.i1] .= 1.0
    fin[p.i2] .= 1.0
    G = copy(p.G)
    H = copy(p.H)
    f1 = f[p.i1]
    f2 = f[p.i2]
    G[1, :] ./= f1; G[2, :] ./= f1
    G[3, :] ./= f2; G[4, :] ./= f2
    H[1, :] .*= f1; H[3, :] .*= f1
    H[2, :] .*= f2; H[4, :] .*= f2
    return AnglePairPreconditioner(_scale_rows(p.pre, fin), p.i1, p.i2, G, H)
end

"""
    _grad_balance(pre, post, Z; ngrad = 10, fmax = 100.0) -> (pre′, f)

Balance the preconditioner's per-coordinate scales between the posterior's marginal
and conditional widths. The draw-based fit (and Welford diagonal adaptation) scale
every coordinate to unit *marginal* width; for a coordinate on a ridge — marginally
wide but conditionally pinned, like a phase offset trading against the image — that
makes the local curvature enormous and caps the leapfrog step size at the conditional
width. The gradient RMS in the sampled space measures exactly that: `|g|ᵢ ≈ 1/σ_cond`
for a unit-marginal coordinate. Dividing the map's response by `f = √|g|` (clamped to
`[1, fmax]`) sets each coordinate to the geometric mean of the two widths, cutting the
per-coordinate condition number from `(marg/cond)²` to `marg/cond`.

Gradients are evaluated with the CPU/Enzyme path at `ngrad` draws spread over `Z`
(inverted through `pre` itself, so the measurement is in the space actually sampled).
`f` is clamped below at 1: coordinates are only narrowed, never widened — the widening
side is the diagonal metric's job during warmup.
"""
function _grad_balance(
        pre, post::Comrade.VLBIPosterior, Z::AbstractMatrix;
        ngrad::Int = 10, fmax::Real = 100.0, reactant::Bool = false
    )
    cols = unique(round.(Int, range(1, size(Z, 2), length = min(ngrad, size(Z, 2)))))
    G = _collect_grads(pre, post, Z, cols; reactant)
    f = clamp.(sqrt.(vec(sqrt.(mean(abs2, G; dims = 2)))), 1.0, Float64(fmax))
    nstiff = count(>(2.0), f)
    @info "Gradient balance: $nstiff coordinates narrowed by >2× " *
        "(max factor $(round(maximum(f); digits = 1)))"
    return _scale_rows(pre, f), f
end

"""
    fit_preconditioner(pilot, post; rank=16, nsamples=2000, min_scale=nothing, discard=0.0, augment=false, angle_pairs=false, grad_balance=false)

Fit a [`LowRankPreconditioner`](@ref) for `post` from the posterior draws of a pilot
run: `pilot` is a MCMC DiskStore directory containing `samples/` and `parameters.jls`
— a run's `outbase` for its post-warmup draws, or `<outbase>/warmup` for the draws
logged during adaptation. The pilot must have sampled the SAME model — its draws are
inverted through `asflat(post)`, and any model difference errors there or in the
dimension check at sampling time.

Up to `nsamples` draws are used, thinned evenly after dropping the first `discard`
fraction of the chain (drop at least half when fitting to warmup draws — the early
adaptation trajectory is far from the posterior); `rank` caps the number of corrected
directions and `min_scale` overrides the default floor (1.5) on their cross-validated
scales (see `_lowrank_from_draws`).

`augment = true` carries the pilot run's own preconditioner (its `transport.jls`) into
the fit: the carried directions are kept, re-scaled on the held-out half of the new
draws, and deflated out of detection so the fit spends its candidate budget on
structure not yet corrected. Use it whenever the pilot itself sampled with a
preconditioner — detection from a short log resolves only the widest directions, so an
un-augmented refit would forget corrections the pilot was already using. `rank` caps
the total (carried + new) directions.
"""
function fit_preconditioner(
        pilot::AbstractString, post::Comrade.VLBIPosterior;
        rank::Int = 16, nsamples::Int = 2000, min_scale::Union{Nothing, Real} = nothing,
        discard::Real = 0.0, augment::Bool = false, angle_pairs::Bool = false,
        grad_balance::Bool = false, grad_reactant::Bool = false, stiff_rank::Int = 0,
        fisher::Bool = false,
        pair_set::Union{Nothing, Tuple{Vector{Int}, Vector{Int}}} = nothing,
        refit_cache = nothing
    )
    0 <= discard < 1 || throw(ArgumentError("discard must be in [0, 1), got $discard"))
    # `augment`: carry the pilot run's own preconditioner (its `transport.jls`) into the
    # fit — its directions are kept (re-scaled on held-out draws) and deflated out of
    # detection, so successive rounds accumulate corrections instead of each short log
    # re-detecting only the widest directions and forgetting the rest.
    carry = nothing
    if augment
        tf = joinpath(
            basename(abspath(pilot)) == "warmup" ? dirname(abspath(pilot)) : pilot,
            "transport.jls"
        )
        old = isfile(tf) ? deserialize(tf) : nothing
        # Pair blocks and the stiff stage are refit fresh from the current draws and
        # gradients each round; only the low-rank wide directions carry over.
        old isa StiffStage && (old = old.pre)
        old isa AnglePairPreconditioner && (old = old.pre)
        if old isa LowRankPreconditioner
            carry = old
        else
            @warn "augment requested but the pilot has no LowRankPreconditioner at " *
                "$tf; fitting without carried directions"
        end
    end
    ntot = deserialize(joinpath(pilot, "parameters.jls")).params.nsamples
    start = round(Int, discard * ntot) + 1
    step = max(1, (ntot - start + 1) ÷ nsamples)
    usedidx = collect(start:step:ntot)
    tpost = asflat(post)
    n = dimension(tpost)
    if fisher
        # Fisher-divergence fit (Seyboldt+ 2026): draws + scores in flat space, one
        # joint estimator subsuming the wide/stiff/balance stages. Scores are exact on
        # the sampled subspace, so no carry (`augment`) is needed — each fit replaces.
        # Only the ≤192 columns actually used are loaded, each inverted and scored ONCE
        # across all refits via the cache, and the score program compiles once — a
        # cached refit costs seconds, enabling a nutpie-like frequent schedule.
        cache = isnothing(refit_cache) ? _new_refit_cache() : refit_cache
        sel = unique(round.(Int, range(1, length(usedidx), length = min(192, length(usedidx)))))
        need = usedidx[sel]
        Zf = zeros(n, length(need))
        Gf = zeros(n, length(need))
        for (k, idx) in enumerate(need)
            Zf[:, k] = get!(cache.draws, idx) do
                Comrade.inverse(tpost, only(Comrade.postsamples(load_samples(pilot, idx:idx))))
            end
            Gf[:, k] = get!(cache.scores, idx) do
                _flat_score!(cache, post, Zf[:, k]; reactant = grad_reactant)
            end
        end
        pre = _fisher_lowrank(Zf, Gf; rank, carry)
        angle_pairs && (pre = _angle_pairs_from_draws(Zf, pre; pair_set))
        @info "Fitted Fisher preconditioner from $(length(need)) draw/score pairs: $pre"
        return pre
    end
    chain = load_samples(pilot, start:step:ntot)
    θs = Comrade.postsamples(chain)
    Z = Matrix{Float64}(undef, n, length(θs))
    for (j, θ) in enumerate(θs)
        Z[:, j] = Comrade.inverse(tpost, θ)
    end
    pre = _lowrank_from_draws(Z; rank, min_scale, carry)
    angle_pairs && (pre = _angle_pairs_from_draws(Z, pre; pair_set))
    grad_balance && ((pre, _) = _grad_balance(pre, post, Z; reactant = grad_reactant))
    if stiff_rank > 0
        # A second gradient set, in the final (balanced) space: the stiff directions
        # must be measured in the coordinates the sampler will actually use.
        cols = unique(round.(Int, range(1, size(Z, 2), length = min(128, size(Z, 2)))))
        Gm = _collect_grads(pre, post, Z, cols; reactant = grad_reactant)
        pre = _stiff_from_grads(Gm, pre; rank = stiff_rank)
    end
    @info "Fitted preconditioner from $(length(θs)) pilot draws" *
        (carry === nothing ? "" : " (carrying $(length(carry.s)) directions)") * ": $pre"
    return pre
end
