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
    tpost = Comrade.maybe_transport(dpost, transport_method)
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
    best_image(post, ntrials=20, maxiters=10_000, rng=Random.default_rng(); opt=Adam())

Run `ntrials` random-restart optimizations of `post`, returning the valid solutions and
their log-densities sorted best-first. Each trial does two optimization passes and keeps
the better one.
"""
function best_image(post, ntrials = 20, maxiters = 10_000, rng = Random.default_rng(); opt = Adam())
    nd = mapreduce(Comrade.ndata, +, post.data)
    sols = map(1:ntrials) do i
        xopt0, sol0 = comrade_opt(
            post, opt;
            initial_params = prior_sample(rng, post), maxiters = maxiters ÷ 2, g_tol = 1.0e-1
        )
        c20 = mapreduce(sum, +, chi2(post, xopt0)) / nd
        @info "Preliminary image $i/$(ntrials) done minimum χ²: $(c20)"

        xopt1, sol1 = comrade_opt(
            post, opt;
            initial_params = xopt0, maxiters = maxiters ÷ 2, g_tol = 1.0e-1
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
function _optimize_stage(post_i, opt, xprev, i, nstage, frac, strategy, rng)
    if strategy.use_reactant
        # Mirror the CPU schedule: full maxiters on the first and last stage, half on the
        # intermediate refine stages; `g_tol` early-stop as in `comrade_opt`/`best_image`.
        mi = (i == 1 || i == nstage) ? strategy.maxiters : strategy.maxiters ÷ 2
        @info "Optimization stage $i/$nstage on Reactant (added noise = $frac)"
        x, _ = reactant_opt(
            post_i, opt; initial_params = xprev, maxiters = mi, ntrials = strategy.ntrials,
            g_tol = strategy.g_tol, verify = strategy.verify_reactant, rng = rng
        )
        return x
    elseif i == 1
        @info "Optimization stage $i/$nstage: random restarts (added noise = $frac)"
        sols, _ = best_image(post_i, strategy.ntrials, strategy.maxiters, rng; opt = opt)
        return sols[1]
    else
        mi = (i == nstage) ? strategy.maxiters : strategy.maxiters ÷ 2
        @info "Optimization stage $i/$nstage: refine (added noise = $frac)"
        x, _ = comrade_opt(post_i, opt; initial_params = xprev, maxiters = mi, g_tol = strategy.g_tol)
        return x
    end
end

function _optimize_tempered(imgbase, skym, intm, data, imgdata, strategy, opt, rng)
    # `nothing` so stage 1 random-restarts on both paths (CPU `best_image`, and `reactant_opt`
    # whose multi-start triggers only when `initial_params === nothing`); later stages
    # warm-start from the previous stage's result.
    xprev = nothing
    nstage = length(strategy.noise_schedule)
    for (i, frac) in enumerate(strategy.noise_schedule)
        dat_i = frac == 0.0 ? data : map(d -> add_fractional_noise(d, frac), data)
        post_i = VLBIPosterior(skym, intm, dat_i...; imgdata)
        xprev = _optimize_stage(post_i, opt, xprev, i, nstage, frac, strategy, rng)
        plot_residuals_png(imgbase * "_residuals_step$(i)_map.png", post_i, xprev)
    end
    return xprev
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
# selects the Fisher low-rank adaptor, which refits the latent space in-run and holds the
# metric at identity; without one the sampler's own diagonal adaptation runs unless the
# config turned it off (as it must be when a fitted transform is supplied up front, since
# diagonal adaptation renormalizes the marginals the transform deliberately set).
function _metric_adaptor(strategy::FittingStrategy)
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
    return Comrade.FisherLowRank(; rank = strategy.precond_rank, schedule = sched)
end

function _sample_reactant(out, post, xopt, strategy, restart, gimg, imgbase, transport_method, tgrad = nothing)
    @info "Building Reactant device posterior for sampling"
    # Reuse the already-built posterior, just dropping the Enzyme AD mode: `prepare_device`
    # iterates every field and would try to `to_rarray` the admode, and the device computes
    # its own gradients. This avoids rebuilding the instrument Jones matrices / FFT plans.
    post_cpu = @set post.admode = nothing
    rpost = Comrade.prepare_device(post_cpu, Comrade.ComradeBase.ReactantEx())
    adaptor = _metric_adaptor(strategy)
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
        # the divergence flag), but wall time per draw divided by the benchmarked
        # gradient time is the leapfrog count, and log2 of that is the depth. The first
        # callback after a (re)compile is skipped — it includes compile time.
        tlast = Ref(NaN)
        depth_note = function (nsteps)
            t = time()
            dt = t - tlast[]
            tlast[] = t
            (isnan(dt) || isnothing(tgrad) || nsteps <= 0) && return ""
            lf = dt / nsteps / tgrad
            return " ~lf/step=$(round(Int, lf)) (depth≈$(round(log2(max(lf, 1)); digits = 1)))"
        end
        cb = function (info)
            params = Comrade.Adapt.adapt(Array, info.params)
            save_checkpoint(post_cpu, params, gimg, imgbase, "sample_round$(info.round)")
            ndiv = count(info.numerical_error)
            @info "sampling batch $(info.round)/$(info.nrounds): n_divergences=$ndiv$(depth_note(stride)) (checkpoint saved)"
            return (; info.round, n_divergences = ndiv)
        end
        # Warmup now runs in chunks of the same `stride`, and its callback fires after EVERY
        # chunk (not once, as with the old fused warmup): render the current draw so warmup
        # progress is watchable, while Comrade checkpoints the adaptation state to disk each
        # chunk (making warmup itself resumable via `restart`). The warmup `info` carries
        # `step`/`total` (steps done / n_adapts) plus host-side `step_size`/`params` — NOT the
        # sampling `round`/`nrounds` fields.
        wstep = Ref(0)
        wcb = function (info)
            params = Comrade.Adapt.adapt(Array, info.params)
            save_checkpoint(post_cpu, params, gimg, imgbase, "warmup_step$(info.step)")
            note = depth_note(info.step - wstep[])
            wstep[] = info.step
            @info "warmup $(info.step)/$(info.total): step_size=$(info.step_size)$note (checkpoint saved)"
            return (; info.step, info.total, info.step_size)
        end
        # nutpie-style init: with in-run refits configured and no pilot transform, one
        # score at the start point sets the initial diagonal metric, so warmup's first
        # segment starts pre-scaled instead of on a unit metric.
        if adaptor isa Comrade.FisherLowRank && isnothing(transport_method) &&
                !restart && !isnothing(xopt)
            if !isempty(strategy.precond_seed_transport)
                @info "Seeding transform from $(strategy.precond_seed_transport)"
                transport_method = deserialize(strategy.precond_seed_transport)
            else
                @info "Initializing transform from the start point's score (nutpie-style)"
                transport_method = Comrade._score_init_pre(post, xopt; reactant = true)
            end
        end
        disk = DiskStore(; name = mkpath(out), stride = stride, callback = cb)
        trace = sample(
            rpost, smplr, strategy.nsample;
            saveto = disk, initial_params = xopt, restart = restart, warmup_callback = wcb,
            transport_method = transport_method
        )
    else
        disk = DiskStore(mkpath(out), stride)
        trace = sample(
            rpost, smplr, strategy.nsample;
            saveto = disk, initial_params = xopt, restart = restart,
            transport_method = transport_method
        )
    end
    return trace.out, 1:10:strategy.nsample
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
    tpost = asflat(post)

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
        check_reactant_consistency(post, Comrade.prepare_device(post, Comrade.ComradeBase.ReactantEx()); rng = rng)
    end

    # The preconditioner participates in every log-density evaluation, so it is fit
    # before benchmarking. `sample` persists it to `<out>/transport.jls`; on restart the
    # stored space wins, so refitting is skipped (a restart with a pilot configured
    # would otherwise warn and be ignored).
    transport_method = if isnothing(strategy.precond_pilot) || restart
        nothing
    else
        @info "Fitting latent-space preconditioner from pilot run $(strategy.precond_pilot)"
        fit_preconditioner(
            strategy.precond_pilot, post;
            rank = strategy.precond_rank, nsamples = strategy.precond_nsamples,
            discard = strategy.precond_discard, augment = strategy.precond_augment,
            # tune gradients on the same backend the run samples on
            grad_reactant = strategy.use_reactant
        )
    end

    tgrad = strategy.benchmark ? _run_benchmarks(post, strategy, transport_method) : nothing

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
