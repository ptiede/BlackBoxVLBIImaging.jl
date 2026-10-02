# The 2π sheets of real-line Gauss–Markov phase chains in the StdNormal latent space. The
# likelihood sees a phase only through `e^{iφ}`, so adding 2π to a chain from some point on
# leaves it unchanged while the chain prior (and so the posterior weight) changes: each such
# shift is a separate mode that NUTS cannot cross.

# Instrument terms that are phases; a term counts as a phase chain when it is one of these
# and its prior is a hierarchical Gauss–Markov chain (a `(params, hyperparams)` value).
const PHASE_CHAIN_TERMS = (:gp1, :gprat)

_wrap_phase(d) = d - 2π * round(d / 2π)

_is_chain(v) = v isa NamedTuple && haskey(v, :params) && haskey(v, :hyperparams)

phase_chain_terms(x) = Tuple(t for t in PHASE_CHAIN_TERMS if haskey(x.instrument, t) && _is_chain(x.instrument[t]))

# Indices of each site's points within a chain's `SiteArray`, in time order.
function _site_points(sa)
    st = Comrade.sites(sa)
    ts = [t.t0 for t in Comrade.times(sa)]
    return map(unique(st)) do s
        I = findall(==(s), st)
        s => I[sortperm(ts[I])]
    end
end

# term => the chain points `post` fixes (reference or initial values): those whose value does
# not move when the StdNormal latent point does.
function _fixed_points(post)
    tp = stdnormal_posterior(post)
    n = dimension(tp)
    xa = Comrade.transform(tp, zeros(n))
    xb = Comrade.transform(tp, randn(Random.Xoshiro(1), n))
    return Dict(
        t => parent(xa.instrument[t].params) .== parent(xb.instrument[t].params)
            for t in phase_chain_terms(xa)
    )
end

"""
    unwrap_phase_chains(post, x) -> (x′, nchanged)

The parameters `x` of `post` with every real-line Gauss–Markov phase chain (see
`PHASE_CHAIN_TERMS`) rebuilt, within each site's path, as the running sum of its steps
wrapped to `[−π, π]`: the path takes the shortest step between consecutive points, and the
sum restarts at each point `post` fixes, which keeps its value. Only multiples of 2π change,
so the likelihood is unchanged. `nchanged` counts the free points that moved. For real-line
chains only (the StdNormal space); a wrapped chain stores angles on `[−π, π]`.
"""
function unwrap_phase_chains(post, x)
    fixed = _fixed_points(post)
    x′ = deepcopy(x)
    nchanged = 0
    for term in phase_chain_terms(x′)
        v = parent(x′.instrument[term].params)
        fx = fixed[term]
        for (_, I) in _site_points(x′.instrument[term].params)
            for j in 2:length(I)
                k, kp = I[j], I[j - 1]
                fx[k] && continue
                new = v[kp] + _wrap_phase(v[k] - v[kp])
                abs(new - v[k]) > π && (nchanged += 1)
                v[k] = new
            end
        end
    end
    return x′, nchanged
end

"""
    PhaseSheetMoves(post::VLBIPosterior; rounds = 1)

Metropolis–Hastings moves between the 2π sheets of the real-line Gauss–Markov phase chains of
`post` in the StdNormal latent space, callable as Comrade's `between_chunks` hook of the
Reactant NUTS sampler: `m(state, tpost, info, rng) -> state`. Each call makes `rounds`
proposals; a proposal picks a chain term and site, a point `k > 1` of that site's path and a
sign, all uniformly, and adds `±2π` to the path from free point `k` up to the next point the
model fixes (or the end). The chain's latent block is
pushed forward, shifted and pulled back; with the hyperparameters fixed the pull-back of the
shift is a shift of the latent coordinates that does not depend on them, so the map has unit
Jacobian, and the opposite sign undoes it. The proposal is accepted with probability
`min(1, π(z′)/π(z))` using the full log density of the sampled space (preconditioned or not).

Errors at construction if `post` has no real-line phase chain.
"""
struct PhaseSheetMoves{N, R}
    node::N          # host StdNormal transport node of the whole posterior
    ranges::R        # term => latent range of its chain block
    points::Vector{Tuple{Symbol, Vector{Int}}}   # (term, path indices) per site
    fixed::Dict{Symbol, BitVector}               # term => points the model fixes
    rounds::Int
    counts::Dict{Symbol, Vector{Int}}            # phase => [proposals, accepted]
    compiled::Base.RefValue{Any}
end

function PhaseSheetMoves(post::VLBIPosterior; rounds::Integer = 1)
    rounds >= 1 || throw(ArgumentError("rounds must be at least 1, got $rounds"))
    tp = stdnormal_posterior(post)
    node = PT.transport_node(tp.transform)
    L = latent_layout(tp)
    x = Comrade.transform(tp, zeros(dimension(tp)))
    terms = phase_chain_terms(x)
    isempty(terms) && error(
        "PhaseSheetMoves needs a real-line Gauss–Markov phase chain among $(PHASE_CHAIN_TERMS); the model has none"
    )
    ranges = Dict(t => L.instrument[t] for t in terms)
    fixed = _fixed_points(post)
    points = Tuple{Symbol, Vector{Int}}[]
    for t in terms, (_, I) in _site_points(x.instrument[t].params)
        count(k -> !fixed[t][k], I[2:end]) > 0 && push!(points, (t, I))
    end
    return PhaseSheetMoves(node, ranges, points, fixed, Int(rounds), Dict(:warmup => [0, 0], :sampling => [0, 0]), Ref{Any}(nothing))
end

_term_node(node, term) = getfield(getfield(getfield(node, :transports).instrument, :transports), term)

# The positions of `I` a proposal starting at its `j`-th point shifts: up to the next fixed one.
function _shift_span(fx, I, j)
    e = findnext(k -> fx[k], I, j + 1)
    return I[j:(isnothing(e) ? lastindex(I) : e - 1)]
end

"""
    sheet_proposal(m::PhaseSheetMoves, u, term, I, j, sign) -> u′

The StdNormal latent point `u` with the path of `term` at the site whose points are `I`
shifted by `sign * 2π` from its free `j`-th point up to the next fixed point.
"""
function sheet_proposal(m::PhaseSheetMoves, u::AbstractVector, term::Symbol, I::Vector{Int}, j::Int, sign::Int)
    g = _term_node(m.node, term)
    r = m.ranges[term]
    xg = PT.latent_pfwd(g, u[r])
    sa = deepcopy(xg.params)
    v = parent(sa)
    m.fixed[term][I[j]] && throw(ArgumentError("point $j of the path is fixed"))
    v[_shift_span(m.fixed[term], I, j)] .+= sign * 2π
    xg′ = (; params = sa, hyperparams = xg.hyperparams)
    ug′ = PT.latent_pback(g, xg′)
    back = parent(PT.latent_pfwd(g, ug′).params)
    maximum(abs, back .- v) < 1.0e-8 || error(
        "the shifted $term path does not round-trip through its transform (a fixed point lies on the shifted part)"
    )
    u′ = copy(u)
    u′[r] = ug′
    return u′
end

_base_latent(pre::Nothing, z) = z
_base_latent(pre, z) = Comrade._affine_fwd(Comrade._hostify(pre), z)
_sampled_latent(pre::Nothing, u) = u
_sampled_latent(pre, u) = Comrade._affine_inv(Comrade._hostify(pre), u)

function _logdensity_fn(m::PhaseSheetMoves, tpost, z)
    c = m.compiled[]
    (!isnothing(c) && c.tpost === tpost) && return c.f
    if _isdevice(z)
        cf = Reactant.Compiler.compile(_posterior_logdensity, (tpost, z))
        f = zz -> _host_scalar(cf(tpost, zz))
    else
        f = zz -> logdensityof(tpost, zz)
    end
    m.compiled[] = (; tpost, f)
    return f
end

function (m::PhaseSheetMoves)(state, tpost, info, rng)
    device = _isdevice(state.position)
    z = vec(Array(state.position))
    pre = Comrade._transport_pre(tpost)
    todev(v) = device ? Reactant.to_rarray(v) : v
    ℓf = _logdensity_fn(m, tpost, todev(z))
    ℓ = ℓf(todev(z))
    u = _base_latent(pre, z)
    nacc = 0
    for _ in 1:m.rounds
        term, I = m.points[rand(rng, eachindex(m.points))]
        j = rand(rng, filter(j -> !m.fixed[term][I[j]], 2:length(I)))
        u′ = sheet_proposal(m, u, term, I, j, rand(rng, (-1, 1)))
        z′ = _sampled_latent(pre, u′)
        ℓ′ = ℓf(todev(z′))
        if log(rand(rng)) < ℓ′ - ℓ
            u, z, ℓ = u′, z′, ℓ′
            nacc += 1
        end
    end
    c = m.counts[info.phase]
    c[1] += m.rounds
    c[2] += nacc
    nacc > 0 && @info "phase sheet moves ($(info.phase)): accepted $nacc of $(m.rounds)"
    state.position = todev(z)
    return state
end

"""
    sheet_summary(m::PhaseSheetMoves) -> String

The proposals and acceptances of the sheet moves in warmup and sampling.
"""
sheet_summary(m::PhaseSheetMoves) = join(
    ("$(p): $(m.counts[p][2])/$(m.counts[p][1]) accepted" for p in (:warmup, :sampling)), "; "
)
