# KineticSamplers MCLMC chains in the StdNormal latent space.

const MCLMC_MOVE_BLOCKS = 50
const MCLMC_START_JITTER = 0.001
const MCLMC_DRAWS_PER_FILE = 10
const MCLMC_CHECK_BLOCKS = 10
const MCLMC_RETURNED_DRAWS = 500

"""
    LatentTarget(tpost, dim)

The LogDensityProblems interface of the device posterior `tpost` in its latent space, with
gradients and Hessian-vector products by Enzyme under Reactant.
"""
struct LatentTarget{P}
    tpost::P
    dim::Int
end

LogDensityProblems.capabilities(::Type{<:LatentTarget}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(t::LatentTarget) = t.dim
LogDensityProblems.logdensity(t::LatentTarget, x) = Comrade.logdensityof(t.tpost, x)

function LogDensityProblems.logdensity_and_gradient(t::LatentTarget, x)
    mode = Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal)
    derivs, val = Enzyme.gradient(mode, Comrade.logdensityof, Enzyme.Const(t.tpost), x)
    return val, last(derivs)
end

KineticSamplers.has_hessian_vector_product(::Type{<:LatentTarget}) = true

function KineticSamplers.logdensity_and_hessian_vector_product(t::LatentTarget, x, v)
    ext = Base.get_extension(Comrade, :ComradeReactantEnzymeExt)
    isnothing(ext) && error("Hessian-vector products need Comrade's Reactant and Enzyme extension")
    # Comrade's product is of the potential's Hessian, the negative of the log density's.
    return Comrade.logdensityof(t.tpost, x), -ext._device_hvp_kernel(t.tpost, x, v)
end

mutable struct MoveState{P}
    position::P
end

# Device buffers are freed only when Julia collects their wrappers, which it does not do on
# device memory pressure.
collect_device_garbage(block) = block % 50 == 0 ? GC.gc() : (block % 5 == 0 && GC.gc(false))

"""
    sample_mclmc(out, post, xopt, strategy, rng; starts=nothing) -> Comrade.PosteriorSamples

`strategy.chains` chains of KineticSamplers MCLMC on `post` in the StdNormal latent space,
compiled with Reactant, started at `xopt` jittered by `$(MCLMC_START_JITTER)` standard normals
per latent coordinate, or chain `k` at `starts[k]` when `starts` is given. A
`PooledAdaptation` of `strategy.nadapt` gradients per chain, in blocks of `strategy.chunk_size`
steps, adapts one step size and one low-rank inverse mass with stiff directions for all
chains, without step halving; its history and parameters go to `out/mclmc_warmup.jls`. Then
$(MCLMC_CHECK_BLOCKS) blocks at the sampling parameters, whose energy statistics are logged per
chain, must pass `KineticSamplers.check_energy`. Sampling then keeps the last position of each
of `strategy.nsample` blocks per chain, chain `k` written as a Comrade sample directory
`out/chain_k`. The moves of `strategy` run on every chain every $(MCLMC_MOVE_BLOCKS) blocks of
the warmup's adaptation phase and of sampling; a chain that moved restarts with a fresh
velocity. Returns about $(MCLMC_RETURNED_DRAWS) draws, evenly thinned from each chain; the
chains on disk keep every draw (see [`load_mclmc_chains`](@ref)).
"""
function sample_mclmc(out, post, xopt, strategy::FittingStrategy, rng; starts = nothing)
    K = strategy.chains
    chain_dirs = [joinpath(out, "chain_$k") for k in 1:K]
    existing = filter(ispath, chain_dirs)
    isempty(existing) || error(
        "$(first(existing)) exists; MCLMC sampling does not resume, so move the earlier chains away"
    )
    post_cpu = @set post.admode = nothing
    space = PT.StdNormal()
    tpost = Comrade.maybe_transport(
        Comrade.prepare_device(post_cpu, Comrade.ComradeBase.ReactantEx()), space
    )
    tb = Comrade.CoordinateView(post_cpu, space).tbase
    d = dimension(tb)
    target = KineticSamplers.Target(LatentTarget(tpost, d); backend = ADTypes.AutoReactant())
    alg = KineticSamplers.MCLMC(; strategy.max_step_halvings)
    # Halving would let stored positions reach regions whose curvature caps the adapted step.
    warmup_alg = KineticSamplers.MCLMC(; max_step_halvings = 0)
    nadapt, block = strategy.nadapt, strategy.chunk_size
    mass = KineticSamplers.HistoryMassAdaptation(
        nadapt; max_rank = 100, max_wide = 80, stiff = KineticSamplers.StiffDirections(),
        vecs_eltype = Float32
    )
    adapt = KineticSamplers.PooledAdaptation(nadapt; mass, block)
    Θ = if isnothing(starts)
        Comrade.inverse(tb, xopt) .+ MCLMC_START_JITTER .* randn(rng, d, K)
    else
        length(starts) == K ||
            error("$(length(starts)) starting points for sampler.chains = $K chains")
        reduce(hcat, [Comrade.inverse(tb, s) for s in starts])
    end
    host_rngs(n) = [Random.Xoshiro(rand(rng, UInt64)) for _ in 1:n]
    restart_rngs = host_rngs(K)

    moves = isempty(strategy.moves) ? nothing : build_moves(
        post_cpu, strategy.moves, xopt; space, output = joinpath(mkpath(out), "moves.jls")
    )
    function move_chains(chains, phase, step, total)
        Θh = Array(chains.state.θ)
        moved = false
        for k in axes(Θh, 2)
            state = MoveState(Reactant.to_rarray(Θh[:, k]))
            moves(state, tpost, (; phase, step, total, report = false), rng)
            x = Array(state.position)
            if x != view(Θh, :, k)
                Θh[:, k] = x
                moved = true
            end
        end
        GC.gc()
        moved || return chains
        return KineticSamplers.init_chains(
            phase === :warmup ? warmup_alg : alg, target, Θh; rngs = restart_rngs
        )
    end

    @info "MCLMC: $K chains on $d latent coordinates, warmup $nadapt gradients per chain in " *
        "blocks of $block steps, $(strategy.nsample) draws per chain; $(repr(alg))"
    t0 = time()
    function between_blocks(chains, info)
        collect_device_garbage(info.block)
        if !isnothing(moves) && info.phase === :adaptation && info.block % MCLMC_MOVE_BLOCKS == 0
            chains = move_chains(chains, :warmup, info.gradients, nadapt)
        end
        info.block % 50 == 0 && @info @sprintf(
            "MCLMC warmup, %s block %d: %d gradients per chain (%.0f s)",
            info.phase, info.block, info.gradients, time() - t0
        )
        _flush_logs()
        return chains
    end
    chains, params, history = KineticSamplers.warmup(
        host_rngs(K), adapt, warmup_alg, target, Θ; between_blocks, check = false
    )
    _log_mclmc_warmup(params, history, d, time() - t0)
    serialize(joinpath(out, "mclmc_warmup.jls"), (; history, params = _host_params(params)))

    params = Reactant.to_rarray(params; track_numbers = Number)
    rngs = KineticSamplers.backend_rngs(ADTypes.AutoReactant(), host_rngs(K))
    chains, trace = check_mclmc(target, rngs, alg, params, chains, block, d)
    recorder = KineticSamplers.NoPositions()
    writers = [MCLMCDrawWriter(dir) for dir in chain_dirs]
    divergent, halved = zeros(Int, K), zeros(Int, K)
    t0 = time()
    for b in 1:strategy.nsample
        tb0 = time()
        chains, trace = KineticSamplers.run_block(
            target, rngs, alg, params, chains, trace, block, recorder
        )
        Θh = Array(chains.state.θ)
        logp, grads = Array(trace.logp), Array(trace.num_grads)
        diverged, halvings = Array(trace.diverged), Array(trace.step_halvings)
        divergent .+= vec(count(diverged; dims = 1))
        halved .+= vec(count(>(0), halvings; dims = 1))
        seconds = (time() - tb0) / K
        for (k, w) in pairs(writers)
            push_draw!(
                w, Comrade.transform(tb, Θh[:, k]);
                logp = logp[end, k], num_grads = sum(view(grads, :, k)), time = seconds,
                divergent = count(view(diverged, :, k)),
                halved = count(>(0), view(halvings, :, k))
            )
        end
        if !isnothing(moves) && b % MCLMC_MOVE_BLOCKS == 0
            chains = move_chains(chains, :sampling, b, strategy.nsample)
        end
        collect_device_garbage(b)
        if b % 50 == 0 || b == strategy.nsample
            @info @sprintf(
                "MCLMC sampling %d/%d draws per chain: divergent steps %s, halved steps %s, log density %s (%.0f s)",
                b, strategy.nsample, join(divergent, " "), join(halved, " "),
                join((@sprintf("%.0f", x) for x in view(logp, lastindex(logp, 1), :)), " "),
                time() - t0
            )
            _flush_logs()
        end
    end
    foreach(flush_draws!, writers)
    isnothing(moves) || foreach(move_summary(moves)) do m
        @info "move $(m.name): warmup $(m.warmup.accepted)/$(m.warmup.proposed), " *
            "sampling $(m.sampling.accepted)/$(m.sampling.proposed) accepted" * _scale_text(m.τ)
    end
    stride = max(1, strategy.nsample ÷ cld(MCLMC_RETURNED_DRAWS, K))
    return load_mclmc_chains(out, 1:stride:strategy.nsample)
end

"""
    load_mclmc_chains(out, nsamples=:) -> Comrade.PosteriorSamples

The draws of the chains `out/chain_1`, `out/chain_2`, … of an MCLMC run (see
[`sample_mclmc`](@ref)), concatenated in chain order, with `nsamples` selecting draws of each
chain.
"""
function load_mclmc_chains(out, nsamples = Base.Colon())
    per_chain = [Comrade.load_samples(dir, nsamples) for dir in _mclmc_chain_dirs(out)]
    return Comrade.PosteriorSamples(
        reduce(vcat, Comrade.postsamples.(per_chain)),
        reduce(vcat, Comrade.samplerstats.(per_chain));
        metadata = Dict(:sampler => :MCLMC, :chains => length(per_chain))
    )
end

function _mclmc_chain_dirs(out)
    names = filter(n -> occursin(r"^chain_\d+$", n), readdir(out))
    isempty(names) && error("no MCLMC chains (chain_1, chain_2, …) in $out")
    sort!(names; by = n -> parse(Int, n[7:end]))
    return [joinpath(out, n) for n in names]
end

# Runs MCLMC_CHECK_BLOCKS blocks and applies KineticSamplers' end-of-warmup energy check to them,
# logging each chain's statistics first.
function check_mclmc(target, rngs, alg, params, chains, block, d)
    recorder = KineticSamplers.NoPositions()
    trace = nothing
    energies = NamedTuple[]
    for _ in 1:MCLMC_CHECK_BLOCKS
        chains, trace = KineticSamplers.run_block(
            target, rngs, alg, params, chains, trace, block, recorder
        )
        push!(
            energies, (;
                energy_error = Array(trace.energy_error), diverged = Array(trace.diverged),
                step_halvings = Array(trace.step_halvings),
            )
        )
    end
    for k in axes(first(energies).energy_error, 2)
        chain_k = [map(a -> a[:, k:k], e) for e in energies]
        st = KineticSamplers.energy_statistics(chain_k, d)
        @info @sprintf(
            "MCLMC check, chain %d: halved %.3g, divergent %.3g, energy error variance per dimension %.3g",
            k, st.halved, st.divergent, st.energy_var
        )
    end
    KineticSamplers.check_energy(energies, d)
    return chains, trace
end

_host_params(params) = (;
    params.step_size, params.trajectory_length,
    inverse_mass = (;
        stds = Array(params.inverse_mass.stds), vecs = Array(params.inverse_mass.vecs),
        vals = Array(params.inverse_mass.vals),
    ),
)

function _log_mclmc_warmup(params, history, d, seconds)
    c = history.curvature
    isnothing(c) || @info @sprintf(
        "MCLMC curvature cap: step size cap %.4g at λ_q %.4g (λ median %.4g, max %.4g over %d positions)",
        c.step_size_cap, c.curvature, c.median, c.max, c.positions
    )
    @info @sprintf(
        "MCLMC warmup done in %.0f s: step size %.4g, trajectory length %.4g = %.2f √d, inverse mass rank %d",
        seconds, params.step_size, params.trajectory_length,
        params.trajectory_length / sqrt(d), size(params.inverse_mass.vecs, 2)
    )
    return nothing
end

"""
    MCLMCDrawWriter(dir)

Draws of one chain written to `dir` as a Comrade sample directory (`samples/output_scan_*.jls`
and `parameters.jls`), $(MCLMC_DRAWS_PER_FILE) draws per file, readable with
`Comrade.load_samples(dir)` while the chain runs.
"""
mutable struct MCLMCDrawWriter
    dir::String
    samples::Vector{Any}
    stats::Vector{NamedTuple}
    files::Int
    draws::Int
end

MCLMCDrawWriter(dir) = MCLMCDrawWriter(mkpath(dir), Any[], NamedTuple[], 0, 0)

function push_draw!(w::MCLMCDrawWriter, θ; stats...)
    push!(w.samples, θ)
    push!(w.stats, NamedTuple(stats))
    length(w.samples) == MCLMC_DRAWS_PER_FILE && flush_draws!(w)
    return w
end

function flush_draws!(w::MCLMCDrawWriter)
    isempty(w.samples) && return w
    w.files += 1
    w.draws += length(w.samples)
    ps = Comrade.PosteriorSamples(map(identity, w.samples), StructArray(map(identity, w.stats)))
    mkpath(joinpath(w.dir, "samples"))
    serialize(
        joinpath(w.dir, "samples", @sprintf("output_scan_%05d.jls", w.files)),
        (samples = Comrade.postsamples(ps), stats = Comrade.samplerstats(ps))
    )
    serialize(
        joinpath(w.dir, "parameters.jls"),
        (; params = Comrade.DiskOutput(abspath(w.dir), w.files, MCLMC_DRAWS_PER_FILE, w.draws))
    )
    empty!(w.samples)
    empty!(w.stats)
    return w
end
