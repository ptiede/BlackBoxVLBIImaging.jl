# Metropolis–Hastings moves run between the Reactant NUTS chunks through Comrade's
# `between_chunks` hook, each a map of the base-flat coordinates (see `SymmetryMoves`).

const SYMMETRY_MOVES = ("flux_gain", "field_scale", "rho_field", "mean_field", "phase_offset")

abstract type SymmetryMove end

# --- locating parameters in the base-flat vector ------------------------------------------

_flat_root(tpost) = _unprecondition(PT.transport_node(tpost.transform))
_unprecondition(node::Comrade.PreconditionedFlat) = node.inner
_unprecondition(node) = node

# The transform node at `path` below `t` and the flat coordinates it consumes, for a `t`
# whose coordinates start after `offset`.
function _locate(t, path::Tuple, offset::Int = 0)
    isempty(path) && return t, (offset + 1):(offset + TV.dimension(t))
    t isa TV.TransformTuple ||
        error("no flat coordinates at $(path): reached a $(nameof(typeof(t)))")
    children = getfield(t, :inner)
    for k in keys(children)
        k == first(path) && return _locate(children[k], Base.tail(path), offset)
        offset += TV.dimension(children[k])
    end
    return error("no parameter $(first(path)) among $(collect(keys(children)))")
end

_rget(x, i) = Comrade.ComradeBase.rgetindex(x, i)

function _replace(x, i::Int, v)
    x′ = copy(x)
    Comrade.ComradeBase.rsetindex!(x′, v, i)
    return x′
end

function _replace(x, r::AbstractUnitRange, v)
    x′ = copy(x)
    x′[r] = vec(v)
    return x′
end

_scalar(t, x, i) = TV.transform(t, _rget(x, i))

_sky_metadata(tpost) = tpost.lpost.skymodel.metadata

# Log of the per-wavenumber factor `A_k(ρ) · rtnrm(ρ)` by which `genfield` of a Markov
# power spectrum multiplies the white coefficients: `A_k = (1 + Σₙ (ρₙ² k²)ⁿ)^(-1/2)` and
# `rtnrm = (Σ_k A_k² · dk)^(-1/2)`, with `k2` the squared wavenumbers of the plan and `dk`
# its cell area over 2π.
function _log_amplitude(ρs::Tuple, k2, dk)
    q = one.(k2)
    for n in eachindex(ρs)
        q = q .+ (ρs[n]^2 .* k2) .^ n
    end
    return log.(q) ./ -2 .- log(sum(inv.(q)) * dk) / 2
end

# The Hartley transform `genfield` applies, `real(F v) + imag(F v)` with `F` the forward
# DFT; applying it twice multiplies by `length(v)`.
function _hartley(v)
    F = Comrade.VLBISkyModels.FFTW.fft(complex.(v))
    return real.(F) .+ imag.(F)
end

# --- the moves --------------------------------------------------------------------------

"""
    FluxGainMove

Trade the total flux against a common gain log-amplitude. The flat coordinate `y` of `ftot`
moves to `y + u`, so `ftot → ftot′ = ftot · exp(2c)`, and every feed-1 gain log-amplitude
`lg1` (every site, every time) moves by `-c`: the whitened coordinates of the `lg1`
Gauss–Markov chain shift by `-c w`, with `w` the whitened image of a unit shift of the chain
at its current hyperparameters. The visibilities see the gains only through `gᵢ conj(gⱼ)`,
so the likelihood is unchanged. The map is a translation along `y` followed by one along the
chain that depends on `y` and the hyperparameters only, so `log|det ∂x′/∂x| = 0`.

Construction needs a sampled `ftot` and an `lg1` chain with fitted hyperparameters, and fails
if the shift does not leave the likelihood unchanged (e.g. an `lg1` chain with a fixed first
stamp).
"""
struct FluxGainMove{T} <: SymmetryMove
    iflux::Int
    tflux::T
    gains::UnitRange{Int}
    nhyper::Int
end

"""
    FieldScaleMove

Trade a non-centered sky field `X` against its scale `σX`: the flat coordinate `y` of `σX`
moves to `y + u` and `X → X · σX / σX′`. The image depends on `σX .* X` only, so the
likelihood is unchanged. `log|det ∂x′/∂x| = n log(σX / σX′)` with `n = length(X)`.
"""
struct FieldScaleMove{T} <: SymmetryMove
    field::Symbol
    coeffs::UnitRange{Int}
    iscale::Int
    tscale::T
end

"""
    RhoFieldMove

Trade one correlation length `ρₙ` of a Markov random field against the field's white
coefficients: the flat coordinate of `ρₙ` (`log ρₙ` for the log-normal prior) moves by `u`
and every coefficient is multiplied by `A_k(ρ) rtnrm(ρ) / (A_k(ρ′) rtnrm(ρ′))`, the ratio
of the factors `genfield` applies to it, so the field and the likelihood are unchanged.
`log|det ∂x′/∂x| = Σ_k log(A_k(ρ) rtnrm(ρ)) − log(A_k(ρ′) rtnrm(ρ′))`. Each term of each
field is its own move.
"""
struct RhoFieldMove{T} <: SymmetryMove
    field::Symbol
    term::Int
    coeffs::UnitRange{Int}
    rho::UnitRange{Int}
    trho::T
    dk::Float64
end

"""
    MeanFieldMove

Trade one mean-model parameter against the log-intensity field `a` of the PolExp Markov RF
sky model: the parameter's flat coordinate moves by `u`, changing the mean image `m → m′`,
and the white coefficients of `a` move by the inverse of the linear map from coefficients to
`σa · δa`, applied to `log m − log m′`. Then `m′ exp(σa δa′) = m exp(σa δa)` pixelwise and
Stokes I, Q, U and V, the flux normalization and the centering are all unchanged. The
coefficient shift depends on the other coordinates only, so `log|det ∂x′/∂x| = 0`.
"""
struct MeanFieldMove{TM, TS, TR} <: SymmetryMove
    param::Symbol
    imean::Int
    mean::UnitRange{Int}
    tmean::TM
    coeffs::UnitRange{Int}
    iscale::Int
    tscale::TS
    rho::UnitRange{Int}
    trho::TR
    dk::Float64
    dims::NTuple{2, Int}
end

"""
    PhaseOffsetMove

Rotate one site's gain phase offset `gp1μ` by `u` by rotating the two coordinates of its
angle embedding, which keeps their radius, so `log|det ∂x′/∂x| = 0`. The likelihood changes;
this is a plain random-walk Metropolis step on the offset.
"""
struct PhaseOffsetMove <: SymmetryMove
    site::Symbol
    i::Int
end

move_name(::FluxGainMove) = "flux_gain"
move_name(m::FieldScaleMove) = "field_scale[$(m.field)]"
move_name(m::RhoFieldMove) = "rho_field[$(m.field),$(m.term)]"
move_name(m::MeanFieldMove) = "mean_field[$(m.param)]"
move_name(m::PhaseOffsetMove) = "phase_offset[$(m.site)]"

# Initial proposal scales in the moved flat coordinate. The scale step of a field of `n`
# white coefficients has a conditional width near 1/√(2n), and 2.4 widths is the usual
# one-dimensional random-walk scale.
initial_scale(::FluxGainMove) = 0.05
initial_scale(m::FieldScaleMove) = 2.4 / sqrt(2 * length(m.coeffs))
initial_scale(::RhoFieldMove) = 0.05
initial_scale(::MeanFieldMove) = 0.01
initial_scale(::PhaseOffsetMove) = 0.05

# Moves the likelihood does not see; construction checks that it does not.
is_invariant(::SymmetryMove) = true
is_invariant(::PhaseOffsetMove) = false

"""
    propose(move, x, u, tpost, ctx) -> (x′, logdet)

The base-flat point `x′ = M_u(x)` of `move` and `log|det ∂x′/∂x|`. `tpost` is the
transformed posterior whose flat layout `move` was built for (host or device), and `ctx`
holds the arrays the moves share (`k2`, the squared wavenumbers of the random-field plan).
"""
function propose end

function propose(m::FluxGainMove, x, u, tpost, ctx)
    y = _rget(x, m.iflux)
    c = (log(TV.transform(m.tflux, y + u)) - log(TV.transform(m.tflux, y))) / 2
    chain = (first(m.gains) + m.nhyper):last(m.gains)
    x′ = _replace(x, m.iflux, y + u)
    x′ = _replace(x′, chain, x[chain] .- c .* _gain_shift(m, x, tpost))
    return x′, zero(c)
end

# Whitened image of a unit shift of every `lg1` value at the current hyperparameters. The
# whitening is affine in the values, so the difference does not depend on them.
function _gain_shift(m::FluxGainMove, x, tpost)
    node, _ = _locate(_flat_root(tpost), (:instrument, :lg1))
    hp = TV.transform(node.hnode, x[first(m.gains):(first(m.gains) + m.nhyper - 1)])
    nchain = length(m.gains) - m.nhyper
    y0 = fill!(similar(x, length(node.dists)), 0)
    y1 = y0 .+ 1
    w0 = similar(x, nchain)
    w1 = similar(x, nchain)
    Comrade._whiten_specs_flat!(w0, 1, y0, Comrade._walk_units(node.dists, y0), hp)
    Comrade._whiten_specs_flat!(w1, 1, y1, Comrade._walk_units(node.dists, y1), hp)
    return w1 .- w0
end

function propose(m::FieldScaleMove, x, u, tpost, ctx)
    y = _rget(x, m.iscale)
    r = TV.transform(m.tscale, y) / TV.transform(m.tscale, y + u)
    x′ = _replace(x, m.iscale, y + u)
    x′ = _replace(x′, m.coeffs, x[m.coeffs] .* r)
    return x′, length(m.coeffs) * log(r)
end

function propose(m::RhoFieldMove, x, u, tpost, ctx)
    i = m.rho[m.term]
    x′ = _replace(x, i, _rget(x, i) + u)
    la = _log_amplitude(TV.transform(m.trho, x[m.rho]), ctx.k2, m.dk)
    la′ = _log_amplitude(TV.transform(m.trho, x′[m.rho]), ctx.k2, m.dk)
    x′ = _replace(x′, m.coeffs, x[m.coeffs] .* vec(exp.(la .- la′)))
    return x′, sum(la) - sum(la′)
end

function propose(m::MeanFieldMove, x, u, tpost, ctx)
    md = _sky_metadata(tpost)
    i = m.mean[m.imean]
    x′ = _replace(x, i, _rget(x, i) + u)
    lm = log.(baseimage(make_mean(md.meanmodel, md.grid, TV.transform(m.tmean, x[m.mean]))))
    lm′ = log.(baseimage(make_mean(md.meanmodel, md.grid, TV.transform(m.tmean, x′[m.mean]))))
    σ = _scalar(m.tscale, x, m.iscale)
    la = _log_amplitude(TV.transform(m.trho, x[m.rho]), ctx.k2, m.dk)
    Δ = _hartley((lm .- lm′) ./ σ) .* exp.(.-la) ./ sqrt(prod(m.dims))
    x′ = _replace(x′, m.coeffs, x[m.coeffs] .+ vec(Δ))
    return x′, zero(σ)
end

function propose(m::PhaseOffsetMove, x, u, tpost, ctx)
    p = _rget(x, m.i)
    q = _rget(x, m.i + 1)
    s, c = sincos(u)
    x′ = _replace(x, m.i, p * c + q * s)
    x′ = _replace(x′, m.i + 1, q * c - p * s)
    return x′, zero(u)
end

# --- building the moves for a posterior -------------------------------------------------

# Non-centered sky fields: arrays `X` with a scalar scale `σX` (the rule `rescale_fields`
# uses).
_scaled_fields(sky) = [
    k for k in keys(sky)
        if sky[k] isa AbstractArray && get(sky, Symbol(:σ, k), nothing) isa Real
]

function _white_coeffs(root, field)
    t, r = _locate(root, (:sky, field))
    t isa TV.ArrayTransformation{TV.Identity} || error(
        "sky field $field has a $(nameof(typeof(t))) flat transform; the moves need its " *
            "flat coordinates to be the white coefficients themselves"
    )
    return r
end

function _scale_coordinate(root, name)
    t, r = _locate(root, (:sky, name))
    t isa TV.ScalarTransform || error("sky parameter $name is not a scalar")
    return only(r), t
end

# Squared wavenumbers of the random-field plan and its cell area over 2π.
function _plan_wavenumbers(plan)
    return plan.kx .^ 2 .+ (plan.ky .^ 2)', step(plan.kx) * step(plan.ky) / (2π)
end

function _markov_plan(post, movename)
    md = post.skymodel.metadata
    (hasproperty(md, :base) && md.base isa SRF) || error(
        "move \"$movename\" needs a Markov RF sky model (order < 0)"
    )
    return md.base.plan
end

function _flux_gain_moves(root, post, θ)
    (haskey(θ.sky, :flux) && haskey(θ.sky.flux, :ftot)) || error(
        "move \"flux_gain\" needs a sampled total flux (a two-value [flux] ftot in the sky " *
            "config); the sky parameters are $(keys(θ.sky))"
    )
    haskey(θ.instrument, :lg1) || error(
        "move \"flux_gain\" needs a gain log-amplitude `lg1`; the instrument parameters " *
            "are $(keys(θ.instrument))"
    )
    node, gains = _locate(root, (:instrument, :lg1))
    node isa Comrade.WhitenedHierarchicalTransform || error(
        "move \"flux_gain\" needs `lg1` to be a Gauss–Markov chain with fitted " *
            "hyperparameters; its flat transform is a $(nameof(typeof(node)))"
    )
    tflux, rflux = _locate(root, (:sky, :flux, :ftot))
    return [FluxGainMove(only(rflux), tflux, gains, TV.dimension(node.hnode))]
end

function _field_scale_moves(root, post, θ)
    fields = _scaled_fields(θ.sky)
    isempty(fields) && error(
        "move \"field_scale\" found no non-centered field (an array `X` with a " *
            "scalar scale `σX`) among the sky parameters $(keys(θ.sky))"
    )
    return map(fields) do f
        FieldScaleMove(f, _white_coeffs(root, f), _scale_coordinate(root, Symbol(:σ, f))...)
    end
end

function _rho_field_moves(root, post, θ)
    plan = _markov_plan(post, "rho_field")
    _, dk = _plan_wavenumbers(plan)
    fields = [f for f in _scaled_fields(θ.sky) if get(θ.sky, Symbol(:ρ, f), nothing) isa Tuple]
    isempty(fields) && error(
        "move \"rho_field\" found no field `X` with a scale `σX` and correlation lengths " *
            "`ρX` among the sky parameters $(keys(θ.sky))"
    )
    return mapreduce(vcat, fields) do f
        trho, rrho = _locate(root, (:sky, Symbol(:ρ, f)))
        length(rrho) == length(θ.sky[Symbol(:ρ, f)]) ||
            error("the correlation lengths ρ$f do not have one flat coordinate each")
        [RhoFieldMove(f, n, _white_coeffs(root, f), rrho, trho, dk) for n in eachindex(rrho)]
    end
end

function _mean_field_moves(root, post, θ)
    post.skymodel.f === _polexp_markovrf_sky || error(
        "move \"mean_field\" needs the PolExp Markov RF sky model (polrep = \"PolExp\", " *
            "order < 0)"
    )
    means = get(θ.sky, :mean, NamedTuple())
    isempty(means) && error("move \"mean_field\" needs a mean model with free parameters")
    plan = _markov_plan(post, "mean_field")
    _, dk = _plan_wavenumbers(plan)
    tmean, rmean = _locate(root, (:sky, :mean))
    length(rmean) == length(means) ||
        error("the mean-model parameters do not have one flat coordinate each")
    coeffs = _white_coeffs(root, :a)
    trho, rrho = _locate(root, (:sky, :ρa))
    dims = size(θ.sky.a)
    return [
        MeanFieldMove(
                k, i, rmean, tmean, coeffs, _scale_coordinate(root, :σa)..., rrho, trho, dk, dims
            ) for (i, k) in enumerate(keys(means))
    ]
end

function _phase_offset_moves(root, post, θ)
    haskey(θ.instrument, Symbol("gp1μ")) || error(
        "move \"phase_offset\" needs a gain phase offset `gp1μ`; the instrument parameters " *
            "are $(keys(θ.instrument))"
    )
    node, r = _locate(root, (:instrument, Symbol("gp1μ")))
    inner = node.inner_transform
    free = inner isa Comrade.PartiallyFixedTransform ? inner.variate_index :
        eachindex(θ.instrument[Symbol("gp1μ")])
    angles = inner isa Comrade.PartiallyFixedTransform ? inner.transform : inner
    (angles isa TV.ArrayTransformation && TV.dimension(angles) == 2 * length(free)) || error(
        "move \"phase_offset\" needs `gp1μ` to be a DiagonalVonMises offset (two flat " *
            "coordinates per free site); its flat transform is a $(nameof(typeof(angles)))"
    )
    sites = θ.instrument[Symbol("gp1μ")].sites
    return [PhaseOffsetMove(sites[v], first(r) + 2 * (k - 1)) for (k, v) in enumerate(free)]
end

const _MOVE_BUILDERS = Dict(
    "flux_gain" => _flux_gain_moves,
    "field_scale" => _field_scale_moves,
    "rho_field" => _rho_field_moves,
    "mean_field" => _mean_field_moves,
    "phase_offset" => _phase_offset_moves,
)

function _parse_moves(names, post, root, θ)
    moves = SymmetryMove[]
    for name in names
        haskey(_MOVE_BUILDERS, name) ||
            error("unknown move \"$name\". Allowed: $(collect(SYMMETRY_MOVES))")
        append!(moves, _MOVE_BUILDERS[name](root, post, θ))
    end
    return Tuple(moves)
end

function _move_context(post, moves)
    any(m -> m isa Union{RhoFieldMove, MeanFieldMove}, moves) || return NamedTuple()
    k2, _ = _plan_wavenumbers(post.skymodel.metadata.base.plan)
    return (; k2 = collect(k2))
end

# --- one Metropolis–Hastings step -------------------------------------------------------

_baseflat(::Nothing, z) = z
_baseflat(pre, z) = Comrade._affine_fwd(pre, z)

# The latent step matching a base-flat step `Δx`: the linear part of the preconditioner's
# inverse, so latent coordinates the move does not touch stay exactly where they were
# when no preconditioner is composed.
_latent_step(::Nothing, Δx) = Δx
function _latent_step(pre, Δx)
    p = Comrade._pre_for(pre, Δx)
    w = Δx ./ p.d
    isempty(p.s) && return w
    return w .+ p.V * ((inv.(p.s) .- 1) .* (p.V' * w))
end

# The proposal of move `j`. On the device every move is evaluated and the one selected, so
# the compiled step serves every move.
_propose_selected(moves::Tuple, j::Integer, x, u, tpost, ctx) =
    propose(moves[j], x, u, tpost, ctx)
function _propose_selected(moves::Tuple, j, x, u, tpost, ctx)
    x′, logdet = x, zero(u)
    for (k, move) in enumerate(moves)
        xk, lk = propose(move, x, u, tpost, ctx)
        pick = j == k
        x′ = ifelse.(pick, xk, x′)
        logdet = ifelse(pick, lk, logdet)
    end
    return x′, logdet
end

_posterior_logdensity(tpost, z) = logdensityof(tpost, vec(z))

"""
    move_step(tpost, moves, ctx, z, ℓ, t, steps, logu; ldf = logdensityof) -> (z′, ℓ′, logα)

Slot `t` of a block of Metropolis–Hastings steps: move `j = mod1(t, length(moves))` proposes
`z′` (a vector) from the latent point `z` with step `steps[t]`, accepted when `logu[t] < logα`. `ℓ` is
`ldf(tpost, z)`; the returned `ℓ′` is that of the returned point. The same function runs on
host arrays and, compiled, on the device.
"""
function move_step(tpost, moves::Tuple, ctx, z0, ℓ, t, steps, logu; ldf = _posterior_logdensity)
    z = vec(z0)
    pre = Comrade._transport_pre(tpost)
    x = _baseflat(pre, z)
    j = rem(t - 1, length(moves)) + 1
    x′, logdet = _propose_selected(moves, j, x, _rget(steps, t), tpost, ctx)
    z′ = z .+ _latent_step(pre, x′ .- x)
    ℓ′ = ldf(tpost, z′)
    logα = ℓ′ - ℓ + logdet
    accept = _rget(logu, t) < logα
    return ifelse.(accept, z′, z), ifelse(accept, ℓ′, ℓ), logα
end

"""
    run_moves(tpost, moves, ctx, z, steps, logu; ldf = logdensityof) -> (z′, logα)

Apply the `length(steps)` Metropolis–Hastings steps of [`move_step`](@ref) in turn on the
host, starting from the latent point `z`, and return the final point (a vector) and every
step's `logα`.
"""
function run_moves(tpost, moves::Tuple, ctx, z, steps, logu; ldf = _posterior_logdensity)
    z = vec(z)
    ℓ = ldf(tpost, z)
    logα = similar(steps, Float64)
    for t in eachindex(steps)
        z, ℓ, logα[t] = move_step(tpost, moves, ctx, z, ℓ, t, steps, logu; ldf)
    end
    return z, logα
end

# --- tuning and the between-chunks hook -------------------------------------------------

# Robbins–Monro adaptation of the log proposal scale toward `target` acceptance during
# warmup, frozen afterward. Acceptance is counted separately per phase.
mutable struct MoveTuner
    logscale::Float64
    nwarmup::Int
    naccwarmup::Int
    nsampling::Int
    naccsampling::Int
end
MoveTuner(scale) = MoveTuner(log(scale), 0, 0, 0, 0)

function record!(t::MoveTuner, phase::Symbol, α, accepted::Bool, target)
    if phase === :warmup
        t.nwarmup += 1
        t.naccwarmup += accepted
        t.logscale = clamp(t.logscale + (α - target) / t.nwarmup^0.6, log(1.0e-8), log(10.0))
    else
        t.nsampling += 1
        t.naccsampling += accepted
    end
    return t
end

"""
    SymmetryMoves(post::VLBIPosterior, names, θ0; rounds = 1, target_accept = 0.45)

Metropolis–Hastings moves (`names`, a subset of `$(SYMMETRY_MOVES)`), callable as Comrade's
`between_chunks` hook of the Reactant NUTS sampler: `sm(state, tpost, info, rng) -> state`.
Each call makes `rounds` rounds, and each round one proposal per move: one per field for
`"field_scale"`, per field and correlation-length term for `"rho_field"`, per mean-model
parameter for `"mean_field"`, and per free site for `"phase_offset"` (see
[`FluxGainMove`](@ref), [`FieldScaleMove`](@ref), [`RhoFieldMove`](@ref),
[`MeanFieldMove`](@ref), [`PhaseOffsetMove`](@ref)).

A move maps the base-flat point `x` to `x′ = M_u(x)` with a step `u ~ N(0, τ²)`, and
`M_{-u}` inverts `M_u`. The latent point moves by the preconditioner's linear inverse of
`x′ − x`, and the proposal is accepted with probability `min(1, α)`,

    log α = ℓ(z′) − ℓ(z) + log|det ∂x′/∂x|,

where `ℓ = logdensityof(tpost, ⋅)` is the full posterior log density in the latent
coordinates (likelihood, prior and flat-transform log-Jacobian), so the acceptance is exact
whether or not the likelihood is invariant. The latent map
`z′ = z + A⁻¹ (M_u(x) − x)`, with `x = b + A z` the preconditioner, has the Jacobian
determinant of `M_u`. A move written instead as a map `θ′ = g_u(θ)` of the constrained
parameters has `log α = log π_θ(θ′) − log π_θ(θ) + log|det ∂θ′/∂θ|`, with `π_θ` the
posterior density of `θ`; the two agree because `ℓ` carries the flat Jacobian `J(x)` and
`log|det ∂x′/∂x| = log|det ∂θ′/∂θ| + log J(x) − log J(x′)`. The moves here are written in
`x`: a flat coordinate is shifted by `u` and a compensating map is applied to the
coordinates it trades against, so coordinates `θ` does not record (the radii of angle
embeddings) never pass through a `θ` round trip.

On a device posterior the step ([`move_step`](@ref)) is one compiled Reactant
program, compiled once per `tpost` and called once per proposal with the position, the
current log density and the random numbers kept on the device; the host draws the steps
and uniforms and reads back each `logα`. During warmup `log τ` of every move follows a
Robbins–Monro recursion toward `target_accept` acceptance, and it is frozen for sampling.
Every warmup call logs the NUTS step size and each move's acceptance and `τ`;
[`move_summary`](@ref) summarizes a phase.

Construction fails if a move does not apply to the model, or if at `θ0` (a constrained
parameter point of `post`) a move that should leave the likelihood unchanged does not.
"""
struct SymmetryMoves{M <: Tuple, C}
    moves::M
    ctx::C
    tuners::Vector{MoveTuner}
    target_accept::Float64
    rounds::Int
    compiled::Base.RefValue{Any}
end

function SymmetryMoves(post::VLBIPosterior, names, θ0; rounds::Integer = 1, target_accept = 0.45)
    rounds >= 1 || throw(ArgumentError("rounds must be at least 1, got $rounds"))
    θ0 = Comrade.Adapt.adapt(Array, θ0)
    tflat = asflat(post)
    moves = _parse_moves(names, post, _flat_root(tflat), θ0)
    ctx = _move_context(post, moves)
    x0 = Comrade.inverse(tflat, θ0)
    foreach(m -> is_invariant(m) && check_move_invariance(post, m, ctx, x0), moves)
    return SymmetryMoves(
        moves, ctx, [MoveTuner(initial_scale(m)) for m in moves], Float64(target_accept),
        Int(rounds), Ref{Any}(nothing)
    )
end

"""
    check_move_invariance(post, move, ctx, x; step = 0.05, rtol = 1e-9)

Error unless the proposal of `move` at the base-flat point `x` with the given `step` leaves
the log-likelihood of `post` unchanged (up to `rtol`).
"""
function check_move_invariance(post, move, ctx, x; step = 0.05, rtol = 1.0e-9)
    tflat = asflat(post)
    x′, _ = propose(move, x, step, tflat, ctx)
    l0 = Comrade.loglikelihood(post, Comrade.transform(tflat, x))
    l1 = Comrade.loglikelihood(post, Comrade.transform(tflat, x′))
    abs(l1 - l0) <= rtol * (abs(l0) + 1) || error(
        "move $(move_name(move)) changed the log-likelihood from $l0 to $l1; the model is " *
            "not invariant under it"
    )
    return nothing
end

_isdevice(z) = z isa Reactant.AbstractConcreteArray

function _compiled_step(sm::SymmetryMoves, tpost, z, steps)
    c = sm.compiled[]
    (!isnothing(c) && c.tpost === tpost && c.n == length(z)) && return c
    ctx = map(Reactant.to_rarray, sm.ctx)
    S = Reactant.to_rarray(vec(steps))
    ℓ = Reactant.ConcreteRNumber(0.0)
    t = Reactant.ConcreteRNumber(Int64(1))
    moves = sm.moves
    step = Reactant.Compiler.compile(
        (tp, ctx, z, ℓ, t, S, L) -> move_step(tp, moves, ctx, z, ℓ, t, S, L),
        (tpost, ctx, z, ℓ, t, S, copy(S))
    )
    ld = Reactant.Compiler.compile(_posterior_logdensity, (tpost, z))
    sm.compiled[] = (; tpost, n = length(z), ctx, step, ld)
    return sm.compiled[]
end

# The position is carried as a plain device vector; Comrade's hook restores the kernels'
# position shape.
function _device_moves(sm::SymmetryMoves, tpost, z0, steps, logu)
    z = Reactant.to_rarray(vec(Array(z0)))
    c = _compiled_step(sm, tpost, z, steps)
    S = Reactant.to_rarray(vec(steps))
    L = Reactant.to_rarray(vec(logu))
    ℓ = c.ld(tpost, z)
    logα = map(eachindex(steps)) do t
        z, ℓ, a = c.step(tpost, c.ctx, z, ℓ, Reactant.ConcreteRNumber(Int64(t)), S, L)
        a
    end
    return z, reshape(Float64.(logα), size(steps))
end

function (sm::SymmetryMoves)(state, tpost, info, rng)
    J = length(sm.moves)
    steps = [exp(sm.tuners[j].logscale) * randn(rng) for j in 1:J, _ in 1:sm.rounds]
    logu = log.(rand(rng, J, sm.rounds))
    z, logα = _isdevice(state.position) ?
        _device_moves(sm, tpost, state.position, steps, logu) :
        run_moves(tpost, sm.moves, sm.ctx, state.position, steps, logu)
    accepted = logu .< logα
    for j in 1:J, r in 1:sm.rounds
        α = isnan(logα[j, r]) ? 0.0 : min(1.0, exp(logα[j, r]))
        record!(sm.tuners[j], info.phase, α, accepted[j, r], sm.target_accept)
    end
    if info.phase === :warmup
        @info "warmup $(info.step)/$(info.total): step_size=$(@sprintf("%.3g", _host_scalar(state.step_size))) " *
            "moves: " * _chunk_summary(sm, accepted)
    end
    state.position = z
    return state
end

_host_scalar(x::Number) = Float64(x)
_host_scalar(x) = Float64(only(Array(x)))

function _chunk_summary(sm::SymmetryMoves, accepted)
    parts = map(enumerate(sm.moves)) do (j, m)
        t = sm.tuners[j]
        chunk = @sprintf("%.2f", count(view(accepted, j, :)) / size(accepted, 2))
        total = @sprintf("%.2f", t.naccwarmup / t.nwarmup)
        return "$(move_name(m)) acc=$chunk [$total] τ=$(@sprintf("%.3g", exp(t.logscale)))"
    end
    return join(parts, "; ")
end

"""
    move_summary(sm::SymmetryMoves, phase::Symbol) -> String

One line with the acceptance rate, count, and current proposal scale of every move in
`phase` (`:warmup` or `:sampling`).
"""
function move_summary(sm::SymmetryMoves, phase::Symbol)
    parts = map(sm.moves, sm.tuners) do m, t
        n, a = phase === :warmup ? (t.nwarmup, t.naccwarmup) : (t.nsampling, t.naccsampling)
        rate = n == 0 ? "n/a" : @sprintf("%.2f", a / n)
        return "$(move_name(m)) acc=$rate ($a/$n) τ=$(@sprintf("%.3g", exp(t.logscale)))"
    end
    return join(parts, "; ")
end
