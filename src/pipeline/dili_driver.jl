# The DILI sampling run: subspace, warmup with subspace rebuilds, sampling, and output in
# Comrade's DiskStore layout (readable with `load_samples`).

"""
    pinned_coordinates(tpost, paths) -> Vector{Int}

The latent coordinates of the parameters named by `paths` (dot-separated, e.g. `"sky.σa"`,
or `"instrument.lg1"` for a whole block) in the transported posterior `tpost`.
"""
function pinned_coordinates(tpost, paths)
    L = latent_layout(tpost)
    idx = Int[]
    for p in paths
        node = L
        for name in split(p, '.')
            s = Symbol(name)
            (node isa NamedTuple && haskey(node, s)) || error(
                "$(repr(p)) is not a parameter path: no $(repr(name)) among " *
                    (node isa NamedTuple ? join(keys(node), ", ") : "the entries of a leaf")
            )
            node = node[s]
        end
        _append_ranges!(idx, node)
    end
    return sort!(unique!(idx))
end

_append_ranges!(idx, r::AbstractRange) = append!(idx, r)
_append_ranges!(idx, t::Union{Tuple, NamedTuple}) = foreach(c -> _append_ranges!(idx, c), t)

# `ndraws` evenly spaced draws of the chain stored at `dir`, as latent coordinates of `tpost`.
function _chain_latents(tpost, dir, ndraws)
    ntot = deserialize(joinpath(dir, "parameters.jls")).params.nsamples
    ndraws <= ntot || error("$dir holds $ntot draws, fewer than dili.subspace_ndraws = $ndraws")
    idx = unique(round.(Int, range(1, ntot; length = ndraws)))
    return [Comrade.inverse(tpost, only(Comrade.postsamples(load_samples(dir, i:i)))) for i in idx]
end

_dili_subspace(k, us, free, cfg, rng) = dili_subspace(
    k, us; rank = cfg.rank, free, oversample = cfg.oversample, power = cfg.power,
    threshold = cfg.threshold, max_basis_bytes = round(Int, cfg.max_basis_gb * 2^30), rng
)

# Draws in Comrade's DiskStore layout: `samples/output_scan_NNNNN.jls` files of `stride`
# draws each (the last may be shorter) and the `parameters.jls` index.
mutable struct _DILIStore
    dir::String
    stride::Int
    nfiles::Int
    nsamples::Int
    samples::Vector{Any}
    stats::Vector{Any}
end

function _DILIStore(dir::AbstractString, stride::Integer)
    sampdir = joinpath(dir, "samples")
    (isfile(joinpath(dir, "parameters.jls")) || (isdir(sampdir) && !isempty(readdir(sampdir)))) &&
        error("$dir already holds a chain; choose another output path or remove it")
    mkpath(sampdir)
    return _DILIStore(abspath(dir), stride, 0, 0, Any[], Any[])
end

function _push_draw!(store::_DILIStore, x, stat)
    push!(store.samples, x)
    push!(store.stats, stat)
    length(store.samples) == store.stride && _flush!(store)
    return store
end

function _flush!(store::_DILIStore)
    isempty(store.samples) && return store
    store.nfiles += 1
    store.nsamples += length(store.samples)
    ps = PosteriorSamples([x for x in store.samples], [s for s in store.stats])
    serialize(
        joinpath(store.dir, "samples", @sprintf("output_scan_%05d.jls", store.nfiles)),
        (samples = Comrade.postsamples(ps), stats = Comrade.samplerstats(ps))
    )
    serialize(joinpath(store.dir, "parameters.jls"), (; params = _disk_output(store)))
    empty!(store.samples)
    empty!(store.stats)
    return store
end

_disk_output(store::_DILIStore) = Comrade.DiskOutput(store.dir, store.nfiles, store.stride, store.nsamples)

# Per-step record of the whole run, warmup first.
function _append_trace!(trace, phase, r)
    append!(trace.phase, fill(phase, length(r.logα)))
    append!(trace.logα, r.logα)
    append!(trace.accepted, r.accepted)
    append!(trace.scale, r.scale)
    append!(trace.seconds, r.seconds)
    append!(trace.logα_c, r.logα_c)
    append!(trace.accepted_c, r.accepted_c)
    append!(trace.scale_c, r.scale_c)
    return trace
end

function _log_segment(what, r, s::DILISampler)
    n = length(r.logα)
    n == 0 && return nothing
    δr, δc = step_sizes(s)
    acc = s.split ? @sprintf("%.3f (subspace) / %.3f (complement)", count(r.accepted) / n, count(r.accepted_c) / n) :
        @sprintf("%.3f", count(r.accepted) / n)
    @info @sprintf(
        "DILI %s: acceptance %s over %d steps, δr = %.3g, δc = %.3g, %.3g s/step, %d NaN rejections",
        what, acc, n, δr, δc, sum(r.seconds) / n, count(isnan, r.logα) + count(isnan, r.logα_c)
    )
    return nothing
end

"""
    sample_dili(out, post::VLBIPosterior, x0, cfg::DILIConfig; rng = Random.default_rng())
    sample_dili(out, k::DILIKernels, u0, cfg::DILIConfig; rng = Random.default_rng())
        -> Comrade.DiskOutput

Run the DILI sampler on [`dili_posterior`](@ref)`(post)` from the parameters `x0` (or on the
kernels `k` from the latent point `u0`) and write the run to the directory `out`:

  - `samples/`, `parameters.jls`: every `cfg.thin`-th sampling draw, as constrained
    parameters, in Comrade's DiskStore layout (`load_samples(out)`). Per-draw stats:
    `potential` (Φ), `acceptance` (the acceptance rate over the thinning window), `step`;
  - `dili_trace.jls`: per step of warmup and sampling, `phase`, `logα`, `accepted`, the
    step-size factor `scale` (the step sizes are `scale .* (δr, δc)`) and `seconds`, plus
    `δr`, `δc` and the warmup steps `refine_steps` after which the subspace was rebuilt. With
    `cfg.split_blocks`, `logα`/`accepted`/`scale` are the subspace steps' and
    `logα_c`/`accepted_c`/`scale_c` the complement steps' (step sizes `scale * δr`,
    `scale_c * δc`).
    Rewritten after every warmup segment and every stored file of draws;
  - `subspace_0.jls` (initial), `subspace_<i>.jls` (the `i`-th rebuild), `subspace.jls` (the
    one used for sampling).

The coordinates of `cfg.pin` stay at `u0`. Errors if `out` already holds a chain.
"""
function sample_dili(out::AbstractString, post::VLBIPosterior, x0, cfg::DILIConfig; rng::Random.AbstractRNG = Random.default_rng())
    k = DILIKernels(post)
    return sample_dili(out, k, Comrade.inverse(k.host, x0), cfg; rng)
end

function sample_dili(
        out::AbstractString, k::DILIKernels, u0::AbstractVector, cfg::DILIConfig;
        rng::Random.AbstractRNG = Random.default_rng()
    )
    tp = k.host
    n = dimension(tp)
    length(u0) == n || throw(DimensionMismatch("u0 has length $(length(u0)), the posterior $n"))
    store = _DILIStore(out, cfg.stride)
    pinned = pinned_coordinates(tp, cfg.pin)
    free = setdiff(1:n, pinned)
    isempty(pinned) ||
        @info "DILI: holding $(length(pinned)) latent coordinates ($(join(cfg.pin, ", "))) at the start point"

    sub = if !isnothing(cfg.subspace)
        s = load_subspace(cfg.subspace)
        (s.n == n && s.free == free) || error(
            "$(cfg.subspace) spans $(length(s.free)) of $(s.n) coordinates; this run samples " *
                "$(length(free)) of $n (check dili.pin)"
        )
        @info "DILI: loaded a rank-$(length(s)) subspace from $(cfg.subspace)"
        s
    else
        us = isnothing(cfg.subspace_draws) ? [collect(Float64, u0)] :
            _chain_latents(tp, cfg.subspace_draws, cfg.subspace_ndraws)
        _dili_subspace(k, us, free, cfg, rng)
    end
    save_subspace(joinpath(out, "subspace_0.jls"), sub)

    sampler(sub, tuner, tuner_c) = DILISampler(
        k, sub; langevin_complement = cfg.langevin_complement, δr = cfg.step_subspace,
        δc = cfg.step_complement, target_accept = cfg.target_accept, tuner,
        split = cfg.split_blocks, tuner_c
    )
    s = sampler(sub, MoveTuner(1.0), MoveTuner(1.0))
    state = dili_start(s, u0)
    trace = (;
        phase = Symbol[], logα = Float64[], accepted = Bool[], scale = Float64[], seconds = Float64[],
        logα_c = Float64[], accepted_c = Bool[], scale_c = Float64[],
        δr = cfg.step_subspace, δc = cfg.step_complement, refine_steps = Int[],
    )
    save_trace() = serialize(joinpath(out, "dili_trace.jls"), trace)

    ends = [round.(Int, cfg.refine_at .* cfg.nwarmup); cfg.nwarmup]
    start = 0
    for (i, stop) in enumerate(ends)
        len = stop - start
        m = min(len, cfg.subspace_ndraws)
        keep = m == 0 ? Set{Int}() : Set(ceil.(Int, range(len / m, len; length = m)))
        kept = Vector{Vector{Float64}}()
        r = dili_advance!(s, state, :warmup, len; rng) do t, st, _
            t in keep && push!(kept, Array(first(st)))
        end
        state = r.state
        _append_trace!(trace, :warmup, r)
        _log_segment("warmup $stop/$(cfg.nwarmup)", r, s)
        if i < length(ends)
            sub = _dili_subspace(k, kept, free, cfg, rng)
            save_subspace(joinpath(out, "subspace_$i.jls"), sub)
            push!(trace.refine_steps, stop)
            s = sampler(sub, s.tuner, s.tuner_c)
        end
        save_trace()
        start = stop
    end
    save_subspace(joinpath(out, "subspace.jls"), sub)

    chunk = cfg.thin * cfg.stride
    nacc = 0
    for first_step in 1:chunk:cfg.nsample
        len = min(chunk, cfg.nsample - first_step + 1)
        r = dili_advance!(s, state, :sampling, len; rng) do t, st, accepted
            nacc += accepted
            t % cfg.thin == 0 || return nothing
            u, Φ, _ = st
            step = first_step + t - 1
            _push_draw!(store, Comrade.transform(tp, Array(u)), (; potential = Φ, acceptance = nacc / cfg.thin, step))
            nacc = 0
            return nothing
        end
        state = r.state
        _flush!(store)
        _append_trace!(trace, :sampling, r)
        save_trace()
        _log_segment("sampling $(first_step + len - 1)/$(cfg.nsample)", r, s)
    end
    return _disk_output(store)
end
