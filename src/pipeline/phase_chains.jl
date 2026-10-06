# Real-line Gauss–Markov phase chains in the StdNormal latent space. The likelihood sees a
# phase only through `e^{iφ}`, so adding 2π to a chain from some point on leaves it
# unchanged while the chain prior changes: each such shift is a separate mode (a 2π sheet).

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
