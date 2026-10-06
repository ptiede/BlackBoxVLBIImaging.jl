# The imaging pipeline. `comrade_imager` runs the staged (noise-tempered) optimization and
# then samples the posterior with either AdvancedHMC NUTS or Reactant NUTS, depending on
# the `FittingStrategy`. Optimization always runs on the CPU/Enzyme posterior; for the
# Reactant sampler a device posterior is built only for the sampling stage.

# The Optimisers.jl rules (Adam/AdamW) are shared by both backends: `comrade_opt` and
# `reactant_opt`/`Optimisers.setup` all accept them. LBFGS (Optim.jl) is CPU-only.
function _select_optimizer(strategy::FittingStrategy)
    if strategy.opt_method == "Adam"
        return Adam(strategy.eta)
    elseif strategy.opt_method == "AdamW"
        return AdamW(strategy.eta)
    elseif strategy.opt_method == "LBFGS"
        strategy.use_reactant &&
            error("LBFGS is not available on the Reactant path (it is not an Optimisers.jl rule); use Adam or AdamW.")
        return LBFGS()
    else
        error("unknown optimizer '$(strategy.opt_method)'. Allowed: Adam, AdamW, LBFGS")
    end
end

# Benchmarks run in the latent space actually sampled: with a preconditioner configured,
# `maybe_transport` composes it in front of the flat transform, so its per-eval cost (and,
# on the Reactant path, its traceability) is measured here rather than discovered at
# sampling time.
function _run_benchmarks(post, strategy, transport_method)
    if strategy.use_reactant
        return _run_reactant_benchmarks(post, transport_method)
    end
    tpost = Comrade.maybe_transport(post, transport_method)
    x0 = randn(dimension(tpost))
    @info "Forward pass benchmark"
    show(IOContext(stdout), MIME("text/plain"), @benchmark logdensityof($tpost, $x0))
    println()
    @info "Reverse pass benchmark"
    show(IOContext(stdout), MIME("text/plain"), @benchmark Comrade.LogDensityProblems.logdensity_and_gradient($tpost, $x0))
    println()
    return nothing
end

# Benchmark the device forward pass and the Enzyme value+gradient used by `reactant_opt`.
# The compiled programs execute synchronously and return concrete arrays, so `@benchmark`
# of the call measures full device execution (compilation happens once, up front).
function _run_reactant_benchmarks(post, transport_method)
    dpost = Comrade.prepare_device(post, Comrade.ComradeBase.ReactantEx())
    tpost = Comrade.maybe_transport(dpost, Comrade._device_space(transport_method))
    xr = Reactant.to_rarray(Comrade.inverse(tpost, prior_sample(Random.default_rng(), dpost)))
    fwd = Reactant.@compile sync = true logdensityof(tpost, xr)
    vg = Reactant.@compile sync = true _reactant_value_and_grad(tpost, xr)
    @info "Forward pass benchmark (Reactant)"
    show(IOContext(stdout), MIME("text/plain"), @benchmark $fwd($tpost, $xr))
    println()
    @info "Reverse pass benchmark (Reactant)"
    brev = @benchmark $vg($tpost, $xr)
    show(IOContext(stdout), MIME("text/plain"), brev)
    println()
    # median seconds per value+gradient: the leapfrog unit cost, used by the sampling
    # callbacks to convert wall time per draw into leapfrogs (and thus tree depth)
    return median(brev.times) * 1e-9
end

"""
    best_image(post, ntrials=20, maxiters=10_000, rng=Random.default_rng(); opt=Adam(),
               held=Dict{Symbol, Float64}(), space=nothing)

Run `ntrials` random-restart optimizations of `post`, returning the valid solutions and
their log-densities sorted best-first. Each trial does two optimization passes and keeps
the better one. The sky parameters in `held` (name => value) stay at their values
throughout. The optimization runs in the latent space `space` (`nothing` is the flat space;
see `Comrade.maybe_transport`).
"""
function best_image(
        post, ntrials = 20, maxiters = 10_000, rng = Random.default_rng();
        opt = Adam(), held::AbstractDict = Dict{Symbol, Float64}(), space = nothing
    )
    nd = mapreduce(Comrade.ndata, +, post.data)
    sols = map(1:ntrials) do i
        xopt0, sol0 = _comrade_opt_held(
            post, opt, held;
            initial_params = prior_sample(rng, post), space, maxiters = maxiters ÷ 2, g_tol = 1.0e-1
        )
        c20 = mapreduce(sum, +, chi2(post, xopt0)) / nd
        @info "Preliminary image $i/$(ntrials) done minimum χ²: $(c20)"

        xopt1, sol1 = _comrade_opt_held(
            post, opt, held;
            initial_params = xopt0, space, maxiters = maxiters ÷ 2, g_tol = 1.0e-1
        )
        c21 = mapreduce(sum, +, chi2(post, xopt1)) / nd
        @info "Best image $i/$(ntrials) done minimum χ²: $(c21)"
        return (sol0.objective < sol1.objective ? xopt0 : xopt1)
    end
    lmaps = logdensityof.(Ref(post), sols)
    valid = .!isnan.(lmaps)
    sols_v = sols[valid]
    lm_v = lmaps[valid]
    inds = sortperm(lm_v, rev = true)
    return sols_v[inds], lm_v[inds]
end

# One tempering stage: optimize `post_i`, warm-started from `xprev`. The CPU path random-
# restarts with `best_image` on the first stage and refines with `comrade_opt` afterwards;
# the Reactant path uses `reactant_opt`, which folds restart-then-refine into a single call
# (it multi-starts when `initial_params === nothing`, and warm-starts otherwise).
function _optimize_stage(post_i, opt, xprev, i, nstage, frac, strategy, rng, held)
    if strategy.use_reactant
        # Mirror the CPU schedule: full maxiters on the first and last stage, half on the
        # intermediate refine stages; `g_tol` early-stop as in `comrade_opt`/`best_image`.
        mi = (i == 1 || i == nstage) ? strategy.maxiters : strategy.maxiters ÷ 2
        @info "Optimization stage $i/$nstage on Reactant (added noise = $frac)"
        x, _ = reactant_opt(
            post_i, opt; initial_params = xprev, maxiters = mi, ntrials = strategy.ntrials,
            g_tol = strategy.g_tol, verify = strategy.verify_reactant, rng, held,
            space = latent_space(strategy)
        )
        return x
    elseif i == 1
        @info "Optimization stage $i/$nstage: random restarts (added noise = $frac)"
        sols, _ = best_image(
            post_i, strategy.ntrials, strategy.maxiters, rng; opt, held, space = latent_space(strategy)
        )
        return sols[1]
    else
        mi = (i == nstage) ? strategy.maxiters : strategy.maxiters ÷ 2
        @info "Optimization stage $i/$nstage: refine (added noise = $frac)"
        x, _ = _comrade_opt_held(
            post_i, opt, held; initial_params = xprev, space = latent_space(strategy),
            maxiters = mi, g_tol = strategy.g_tol
        )
        return x
    end
end

function _optimize_tempered(imgbase, skym, intm, data, imgdata, strategy, opt, rng)
    # `nothing` so stage 1 random-restarts on both paths (CPU `best_image`, and `reactant_opt`
    # whose multi-start triggers only when `initial_params === nothing`); later stages
    # warm-start from the previous stage's result.
    xprev = nothing
    post_i = nothing
    held = Dict{Symbol, Float64}()
    nstage = length(strategy.noise_schedule)
    for (i, frac) in enumerate(strategy.noise_schedule)
        dat_i = frac == 0.0 ? data : map(d -> add_fractional_noise(d, frac), data)
        post_i = VLBIPosterior(skym, intm, dat_i...; imgdata)
        if i == 1 && !isempty(strategy.fix_scales)
            held = held_scale_values(post_i, strategy.fix_scales)
            @info "Holding sky scales at their prior medians during optimization: $held"
        end
        xprev = _optimize_stage(post_i, opt, xprev, i, nstage, frac, strategy, rng, held)
        _check_held(xprev, held)
        plot_residuals_png(imgbase * "_residuals_step$(i)_map.png", post_i, xprev)
    end
    isempty(held) && return xprev
    xres = rescale_fields(post_i, xprev)
    @info "Rescaled the sky fields to unit rms: " *
        join(("$k = $(xres.sky[k])" for k in keys(xres.sky) if xres.sky[k] isa Real && startswith(String(k), "σ")), ", ")
    return xres
end

# `start` accepts either a serialized parameter file — a raw NamedTuple, or a run's
# `_optimum_allres.jls` Dict carrying it under :xopt — or a MCMC DiskStore directory,
# from which the LAST stored draw is used: a posterior sample lies in the typical set,
# which a high-dimensional MAP does not.
function _load_start(path)
    if isdir(path)
        ntot = deserialize(joinpath(path, "parameters.jls")).params.nsamples
        return only(Comrade.postsamples(load_samples(path, ntot:ntot)))
    end
    x = deserialize(path)
    x isa AbstractDict && return x[:xopt]
    return x
end

function _sample_ahmc(out, post, tpost, xopt, strategy, rng, restart, transport_method)
    integrator = Leapfrog(strategy.step_size)
    metric = DiagEuclideanMetric(dimension(tpost))
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StanHMCAdaptor(
        MassMatrixAdaptor(metric), StepSizeAdaptor(strategy.target_accept, integrator);
        init_buffer = strategy.init_buffer, term_buffer = strategy.term_buffer
    )
    smplr = HMCSampler(kernel, metric, adaptor)
    # `nsample` is the number of POST-warmup samples (matching the Reactant path). AdvancedHMC
    # counts adaptation in its total chain length, so draw `nadapt + nsample` and keep the tail.
    total = strategy.nadapt + strategy.nsample
    trace = sample(
        rng, post, smplr, total;
        saveto = DiskStore(mkpath(out), 25), n_adapts = strategy.nadapt,
        initial_params = xopt, restart = restart, transport_method = transport_method
    )
    return trace, (strategy.nadapt + 1):10:total
end

# The warmup metric strategy a fitting config asks for. A `[precondition]` refit schedule
# selects a low-rank adaptor (Fisher or Gauss–Newton, per `refit_kind`), which refits the
# latent space in-run and holds the metric at identity; without one the sampler's own
# diagonal adaptation runs unless the config turned it off (as it must be when a fitted
# transform is supplied up front, since diagonal adaptation renormalizes the marginals the
# transform deliberately set). Gauss–Newton refits need `curvature`, the function
# `gauss_newton_curvature` builds.
function _metric_adaptor(strategy::FittingStrategy, curvature = nothing; rows = nothing)
    sched = if strategy.precond_refit_schedule == "stan"
        :stan
    elseif strategy.precond_refit_schedule == "nutpie"
        :nutpie
    elseif !isempty(strategy.precond_refit_at)
        strategy.precond_refit_at
    else
        nothing
    end
    isnothing(sched) && return strategy.adapt_mass_matrix ?
        Comrade.WelfordDiagonal() : Comrade.FixedMetric()
    if strategy.precond_refit_kind == "gauss_newton"
        isnothing(curvature) && throw(ArgumentError("Gauss–Newton refits need a curvature function"))
        k = strategy.precond_rank + strategy.precond_oversample
        return Comrade.GaussNewtonLowRank(
            curvature; rank = strategy.precond_rank, oversample = strategy.precond_oversample,
            probes_per_draw = strategy.precond_probes_per_draw == 0 ? k : strategy.precond_probes_per_draw,
            threshold = strategy.precond_threshold, schedule = sched, rows
        )
    end
    return Comrade.FisherLowRank(;
        rank = strategy.precond_rank, schedule = sched, discard = strategy.precond_discard,
        carry = Symbol(strategy.precond_refit_carry)
    )
end

# A stored preconditioner to seed warmup from; it must act in the run's latent space.
function _seed_transport(strategy::FittingStrategy)
    t = deserialize(strategy.precond_seed_transport)
    want = strategy.latent_space == "flat" ? Comrade.LowRankPreconditioner : Comrade.Preconditioned
    t isa want || error(
        "precondition.seed_transport $(strategy.precond_seed_transport) holds a $(nameof(typeof(t))); " *
            "run.latent_space = \"$(strategy.latent_space)\" needs a $(nameof(want))"
    )
    return t
end

# The paths of the non-finite entries of a constrained parameter point, with the first
# offending value of each array.
function _nonfinite_paths(x, path = "")
    x isa NamedTuple && return reduce(vcat, (_nonfinite_paths(v, "$path.$k") for (k, v) in pairs(x)); init = String[])
    x isa Tuple && return reduce(vcat, (_nonfinite_paths(v, "$path[$i]") for (i, v) in pairs(x)); init = String[])
    if x isa AbstractArray{<:Number}
        bad = findall(!isfinite, vec(parent(x)))
        return isempty(bad) ? String[] : ["$path ($(length(bad)) of $(length(x)) entries, e.g. $(vec(parent(x))[first(bad)]))"]
    end
    return x isa Number && !isfinite(x) ? ["$path = $x"] : String[]
end

# Error, naming the parameters, if a draw has a non-finite value.
function _check_finite_draw(x, where)
    bad = _nonfinite_paths(x)
    isempty(bad) || error("the $where draw has non-finite parameters: $(join(bad, "; "))")
    return x
end

# The wall time and the moves' total time (`Comrade.move_seconds`) at the end of the last
# sampler callback of a phase; `t = NaN` before the first.
mutable struct _NUTSClock
    t::Float64
    moved::Float64
end
_NUTSClock() = _NUTSClock(NaN, 0.0)
_restart!(c::_NUTSClock, t, moved) = (c.t = t; c.moved = moved; c)

# Leapfrog steps per draw over the `nsteps` draws since the clock was restarted: the wall
# time to `t` less the moves' time in between, divided by the time of one gradient
# `tgrad`. Empty before the first restart (that interval includes compile time) or without
# `tgrad`. With `maxdepth`, a mean of at least 95% of the `2^maxdepth − 1` leapfrog steps
# of a full tree is flagged: the trajectories end at the depth cap, not at a U-turn.
function _depth_note(c::_NUTSClock, t, moved, nsteps, tgrad; maxdepth = nothing)
    dt = t - c.t - (moved - c.moved)
    (isnan(dt) || isnothing(tgrad) || nsteps <= 0) && return ""
    lf = dt / nsteps / tgrad
    capped = !isnothing(maxdepth) && lf >= 0.95 * (2^maxdepth - 1)
    return " ~lf/step=$(round(Int, lf)) (depth≈$(round(log2(max(lf, 1)); digits = 1)))" *
        (capped ? ", at the depth cap $maxdepth" : "")
end

function _sample_reactant(out, post, xopt, strategy, restart, gimg, imgbase, transport_method, tgrad = nothing)
    @info "Building Reactant device posterior for sampling"
    # Reuse the already-built posterior, just dropping the Enzyme AD mode: `prepare_device`
    # iterates every field and would try to `to_rarray` the admode, and the device computes
    # its own gradients. This avoids rebuilding the instrument Jones matrices / FFT plans.
    post_cpu = @set post.admode = nothing
    rpost = Comrade.prepare_device(post_cpu, Comrade.ComradeBase.ReactantEx())
    curvature = strategy.precond_refit_kind == "gauss_newton" ?
        gauss_newton_curvature(GaussNewtonKernels(post_cpu, rpost)) : nothing
    rows = strategy.precond_refit_kind == "gauss_newton" ?
        gauss_newton_rows(post_cpu, strategy.precond_band_limit) : nothing
    isnothing(rows) || @info "Gauss–Newton directions on $(length(rows)) of " *
        "$(dimension(Comrade.transport_to(post_cpu, PT.StdNormal()))) latent coordinates " *
        "(sky-field modes up to $(strategy.precond_band_limit) × the longest baseline)"
    adaptor = _metric_adaptor(strategy, curvature; rows)
    moves = isempty(strategy.moves) ? nothing : build_moves(
        post_cpu, strategy.moves, xopt;
        space = latent_space(strategy), output = joinpath(mkpath(out), "moves.jls")
    )
    isnothing(moves) || @info "Moves between NUTS chunks (proposals per call): " *
        join(("$(Comrade.move_name(m)) ×$r" for (m, r) in zip(moves.moves, moves.rounds)), ", ")
    smplr = Comrade.ReactantNUTS(;
        n_adapts = strategy.nadapt, init_step_size = strategy.step_size,
        max_tree_depth = strategy.max_tree_depth,
        metric_adaptor = adaptor
    )

    # `sample_checkpoint` sets the sampling DiskStore stride (= batch / checkpoint frequency,
    # independent of the optimization checkpoint stride); falls back to `chunk_size` when off.
    stride = strategy.sample_checkpoint > 0 ? strategy.sample_checkpoint : strategy.chunk_size
    if strategy.sample_checkpoint > 0
        # Post-warmup per-batch checkpoint: render the latest draw and save FITS+PNG+resid.
        # Tree depth is not exposed by the ProbProg backend (its diagnostics carry only
        # the divergence flag), so `_depth_note` estimates it from the NUTS wall time
        # between callbacks; each phase has its own clock.
        movesec() = isnothing(moves) ? 0.0 : Comrade.move_seconds(moves)
        tsample = _NUTSClock()
        cb = function (info)
            t = time()
            params = _check_finite_draw(Comrade.Adapt.adapt(Array, info.params), "sampling batch $(info.round)")
            save_checkpoint(post_cpu, params, gimg, imgbase, "sample_round$(info.round)")
            ndiv = count(info.numerical_error)
            tg = something(get(info.extras, :gradient_time, nothing), tgrad, Some(nothing))
            note = _depth_note(tsample, t, movesec(), stride, tg; maxdepth = strategy.max_tree_depth)
            @info "sampling batch $(info.round)/$(info.nrounds): n_divergences=$ndiv$note (checkpoint saved)"
            _restart!(tsample, time(), movesec())
            return (; info.round, n_divergences = ndiv)
        end
        # Warmup now runs in chunks of the same `stride`, and its callback fires after EVERY
        # chunk (not once, as with the old fused warmup): render the current draw so warmup
        # progress is watchable, while Comrade checkpoints the adaptation state to disk each
        # chunk (making warmup itself resumable via `restart`). The warmup `info` carries
        # `step`/`total` (steps done / n_adapts) plus host-side `step_size`/`params` — NOT the
        # sampling `round`/`nrounds` fields.
        wstep = Ref(0)
        twarm = _NUTSClock()
        wcb = function (info)
            t = time()
            params = _check_finite_draw(Comrade.Adapt.adapt(Array, info.params), "warmup step $(info.step)")
            save_checkpoint(post_cpu, params, gimg, imgbase, "warmup_step$(info.step)")
            tg = something(get(info, :gradient_time, nothing), tgrad, Some(nothing))
            note = _depth_note(twarm, t, movesec(), info.step - wstep[], tg; maxdepth = strategy.max_tree_depth)
            wstep[] = info.step
            @info "warmup $(info.step)/$(info.total): step_size=$(info.step_size)$note (checkpoint saved)"
            _restart!(twarm, time(), movesec())
            return (; info.step, info.total, info.step_size)
        end
        # nutpie-style init: with in-run refits configured and no pilot transform, one
        # score at the start point sets the initial diagonal metric, so warmup's first
        # segment starts pre-scaled instead of on a unit metric.
        if adaptor isa Comrade.FisherLowRank && transport_method isa Union{Nothing, PT.StdNormal} &&
                !restart && !isnothing(xopt)
            @info "Initializing transform from the start point's score (nutpie-style)"
            transport_method = Comrade._score_init_pre(
                post, xopt; reactant = true, space = latent_space(strategy)
            )
        end
        # The sampling DiskStore records wall time per draw; with the cost of one gradient
        # stored next to it, leapfrog steps per draw (and so tree depth) can be recovered
        # from the run directory alone.
        isnothing(tgrad) || serialize(joinpath(mkpath(out), "gradient_time.jls"), tgrad)
        disk = DiskStore(; name = mkpath(out), stride = stride, callback = cb)
        trace = sample(
            rpost, smplr, strategy.nsample;
            saveto = disk, initial_params = xopt, restart = restart, warmup_callback = wcb,
            transport_method = transport_method, between_chunks = moves
        )
    else
        disk = DiskStore(mkpath(out), stride)
        trace = sample(
            rpost, smplr, strategy.nsample;
            saveto = disk, initial_params = xopt, restart = restart,
            transport_method = transport_method, between_chunks = moves
        )
    end
    isnothing(moves) || foreach(move_summary(moves)) do m
        @info "move $(m.name): warmup $(m.warmup.accepted)/$(m.warmup.proposed), " *
            "sampling $(m.sampling.accepted)/$(m.sampling.proposed) accepted" *
            _scale_text(m.τ)
    end
    return trace.out, 1:10:strategy.nsample
end

_scale_text(::Nothing) = ""
_scale_text(τ::Real) = ", τ = $(@sprintf("%.3g", τ))"
_scale_text(τ::AbstractVector) = ", τ = $(@sprintf("%.3g", minimum(τ)))–$(@sprintf("%.3g", maximum(τ))) over $(length(τ)) components"

"""
    check_start(post, space, x) -> Float64

The log density of `post` in the latent space `space` (see `Comrade.maybe_transport`) at the
parameters `x`. Errors if it is not finite, naming the parameters whose latent coordinates are
not finite.
"""
function check_start(post, space, x)
    tpost = Comrade.maybe_transport(post, space)
    u = Comrade.inverse(tpost, x)
    ℓ = logdensityof(tpost, u)
    isfinite(ℓ) && return ℓ
    bad = String[]
    _nonfinite_paths!(bad, "", latent_layout(tpost), u)
    where_ = isempty(bad) ? "every latent coordinate is finite" :
        "non-finite latent coordinates in " * join(bad, ", ")
    return error(
        "the start point has log density $ℓ in the $(isnothing(space) ? "flat" : nameof(typeof(space))) " *
            "latent space ($where_)"
    )
end

function _nonfinite_paths!(bad, prefix, node, u)
    if node isa AbstractRange
        all(isfinite, view(u, node)) || push!(bad, isempty(prefix) ? "(all)" : prefix)
    else
        for (k, v) in pairs(node)
            _nonfinite_paths!(bad, isempty(prefix) ? string(k) : "$prefix.$k", v, u)
        end
    end
    return bad
end

"""
    comrade_imager(outbase, skym, intm, data...; strategy, imgdata=nothing,
                   rng=Random.default_rng(), restart=false)

Run the full imaging pipeline: staged noise-tempered optimization (or restart/start),
save the optimal image + caltables, then sample the posterior (AdvancedHMC or Reactant
NUTS per `strategy`) and write posterior FITS draws. Returns the path the run was written
to.

`restart=true` resumes from a previously serialized optimum at `outbase` instead of
re-optimizing. It is a one-off run action (not part of `strategy`).
"""
function comrade_imager(
        outbase::String, skym, intm, data...;
        strategy::FittingStrategy, imgdata = nothing, rng = Random.default_rng(),
        restart::Bool = false
    )
    @info "Imaging output base: $outbase"
    mkpath(dirname(outbase))
    outimg = mkpath(joinpath(dirname(outbase), "images"))
    out = outbase
    # Rendered sky images + residual PNGs go under images/, caltables under caltables/; the
    # .jls and MCMC DiskStore stay at the run root. These are per-file prefixes inside each.
    imgbase = joinpath(outimg, basename(outbase))
    caltabbase = joinpath(dirname(outbase), "caltables", basename(outbase))

    # CPU / Enzyme posterior used for optimization, residuals and serialization.
    post = VLBIPosterior(skym, intm, data...; imgdata)
    tpost = Comrade.maybe_transport(post, latent_space(strategy))

    # The posterior is a property of the run, not of the optimizer, so it gets its own file
    # rather than riding along in `_optimum_allres.jls` (which is rewritten per optimization
    # stage and is only about `xopt`). Written up front — before optimization — so anything
    # reloading the run has it from the moment the job starts, including mid-warmup.
    serialize(out * "_posterior.jls", post)

    # On the Reactant path, verify the device posterior reproduces the CPU/Enzyme
    # log-density AND gradient before fitting — a broken device model silently yields garbage
    # fits (e.g. blown-up leakage). `post` here carries the Enzyme AD mode needed for the
    # CPU reference. Errors on mismatch.
    if strategy.use_reactant && strategy.verify_reactant
        @info "Verifying Reactant device posterior against the CPU/Enzyme reference"
        check_reactant_consistency(
            post, Comrade.prepare_device(post, Comrade.ComradeBase.ReactantEx());
            rng, space = latent_space(strategy)
        )
    end

    # The preconditioner participates in every log-density evaluation, so it is fit
    # before benchmarking. `sample` persists it to `<out>/transport.jls`; on restart the
    # stored space wins, so refitting is skipped (a restart with a pilot configured
    # would otherwise warn and be ignored).
    transport_method = if restart
        nothing
    elseif !isempty(strategy.precond_seed_transport)
        isnothing(strategy.precond_pilot) ||
            error("precondition.pilot and precondition.seed_transport are exclusive; set one")
        @info "Seeding transform from $(strategy.precond_seed_transport)"
        _seed_transport(strategy)
    elseif isnothing(strategy.precond_pilot)
        latent_space(strategy)
    else
        @info "Fitting latent-space preconditioner from pilot run $(strategy.precond_pilot)"
        fit_preconditioner(
            strategy.precond_pilot, post;
            rank = strategy.precond_rank, nsamples = strategy.precond_nsamples,
            discard = strategy.precond_discard, augment = strategy.precond_augment,
            space = latent_space(strategy),
            # tune gradients on the same backend the run samples on
            grad_reactant = strategy.use_reactant
        )
    end

    bench_space = isnothing(transport_method) ? latent_space(strategy) : transport_method
    tgrad = strategy.benchmark ? _run_benchmarks(post, strategy, bench_space) : nothing

    g = post.skymodel.grid.imgdomain
    gimg = refinespatial(g, 2)
    opt = _select_optimizer(strategy)

    # ---- optimization / start / restart ----------------------------------------------
    if restart
        @info "Restarting from $(out)_optimum_allres.jls"
        xopt = deserialize(out * "_optimum_allres.jls")[:xopt]
    elseif !isnothing(strategy.start)
        startx = _load_start(strategy.start)
        @info "Starting from $(strategy.start); logdensity = $(logdensityof(post, startx))"
        xopt = startx
        save_optimal(imgbase, post, xopt, gimg; label = "start")
        plot_residuals_png(imgbase * "_residuals_map.png", post, xopt)
        write_caltables(caltabbase, xopt)
        serialize(out * "_optimum_allres.jls", Dict(:xopt => xopt))
    else
        xopt = _optimize_tempered(imgbase, skym, intm, data, imgdata, strategy, opt, rng)
        save_optimal(imgbase, post, xopt, gimg; label = "optimal")
        plot_residuals_png(imgbase * "_residuals_final_map.png", post, xopt)
        write_caltables(caltabbase, xopt)
        serialize(out * "_optimum_allres.jls", Dict(:xopt => xopt))
    end

    # ---- sampling --------------------------------------------------------------------
    if strategy.latent_space == "stdnormal"
        xopt, nsheet = unwrap_phase_chains(post, xopt)
        nsheet > 0 && @info "Re-wrapped $nsheet phase-chain step(s) of the start point to their shortest form"
    end
    check_start(post, latent_space(strategy), xopt)
    if strategy.use_reactant
        trace, range = _sample_reactant(out, post, xopt, strategy, restart, gimg, imgbase, transport_method, tgrad)
    else
        trace, range = _sample_ahmc(out, post, tpost, xopt, strategy, rng, restart, transport_method)
    end

    chain = load_samples(trace, range)
    nchain = length(Comrade.postsamples(chain))

    nres = min(10, nchain)
    if nres > 0
        plot_chain_residuals_png(imgbase * "_residuals.png", post, sample(chain, nres))
    end

    @info "Saving posterior images"
    ndraws = min(500, nchain)
    samples = skymodel.(Ref(post), sample(chain, ndraws))
    save_posterior_draws(outimg, outbase, post, samples, gimg)
    return out
end
