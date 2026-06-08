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

function _run_benchmarks(post, strategy)
    if strategy.use_reactant
        return _run_reactant_benchmarks(post)
    end
    tpost = asflat(post)
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
function _run_reactant_benchmarks(post)
    dpost = Comrade.prepare_device(post, Comrade.ComradeBase.ReactantEx())
    tpost = asflat(dpost)
    xr = Reactant.to_rarray(Comrade.inverse(tpost, prior_sample(Random.default_rng(), dpost)))
    fwd = Reactant.@compile sync = true logdensityof(tpost, xr)
    vg = Reactant.@compile sync = true _reactant_value_and_grad(tpost, xr)
    @info "Forward pass benchmark (Reactant)"
    show(IOContext(stdout), MIME("text/plain"), @benchmark $fwd($tpost, $xr))
    println()
    @info "Reverse pass benchmark (Reactant)"
    show(IOContext(stdout), MIME("text/plain"), @benchmark $vg($tpost, $xr))
    println()
    return nothing
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

function _optimize_tempered(imgbase, skym, intm, data, imgdata, strategy, opt, rng)
    xprev = nothing
    nstage = length(strategy.noise_schedule)
    for (i, frac) in enumerate(strategy.noise_schedule)
        dat_i = frac == 0.0 ? data : map(d -> add_fractional_noise(d, frac), data)
        if strategy.use_reactant
            # Reactant path: Optimisers.jl loop on the device (no random restarts, since each
            # device posterior recompiles). `post_i` (admode=nothing, like the sampler path)
            # stays on the host for the residual plot; `reactant_opt` moves its own copy to
            # the device.
            post_i = VLBIPosterior(skym, intm, dat_i...; imgdata, admode = nothing)
            @info "Optimization stage $i/$nstage on Reactant (added noise = $frac)"
            xprev, _ = reactant_opt(
                post_i, opt; initial_params = xprev, maxiters = strategy.maxiters,
                ntrials = strategy.ntrials, verify = strategy.verify_reactant, rng = rng
            )
            plot_residuals_png(imgbase * "_residuals_step$(i)_map.png", post_i, xprev)
            continue
        end
        post_i = VLBIPosterior(skym, intm, dat_i...; imgdata)
        if i == 1
            @info "Optimization stage $i/$nstage: random restarts (added noise = $frac)"
            sols, _ = best_image(post_i, strategy.ntrials, strategy.maxiters, rng; opt = opt)
            xprev = sols[1]
        else
            mi = (i == nstage) ? strategy.maxiters : strategy.maxiters ÷ 2
            @info "Optimization stage $i/$nstage: refine (added noise = $frac)"
            xprev, _ = comrade_opt(post_i, opt; initial_params = xprev, maxiters = mi, g_tol = strategy.g_tol)
        end
        plot_residuals_png(imgbase * "_residuals_step$(i)_map.png", post_i, xprev)
    end
    return xprev
end

function _sample_ahmc(out, post, tpost, xopt, strategy, rng, restart)
    integrator = Leapfrog(strategy.step_size)
    metric = DiagEuclideanMetric(dimension(tpost))
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StanHMCAdaptor(
        MassMatrixAdaptor(metric), StepSizeAdaptor(strategy.target_accept, integrator);
        init_buffer = strategy.init_buffer, term_buffer = strategy.term_buffer
    )
    smplr = HMCSampler(kernel, metric, adaptor)
    trace = sample(
        rng, post, smplr, strategy.nsample;
        saveto = DiskStore(mkpath(out), 25), n_adapts = strategy.nadapt,
        initial_params = xopt, restart = restart
    )
    return trace, (strategy.nadapt + 1):10:strategy.nsample
end

function _sample_reactant(out, skym, intm, data, imgdata, xopt, strategy, restart, gimg, imgbase)
    @info "Building Reactant device posterior for sampling"
    post_cpu = VLBIPosterior(skym, intm, data...; imgdata, admode = nothing)
    rpost = Comrade.prepare_device(post_cpu, Comrade.ComradeBase.ReactantEx())
    smplr = Comrade.ReactantNUTS(;
        n_adapts = strategy.nadapt, init_step_size = strategy.step_size,
        max_tree_depth = strategy.max_tree_depth, init_buffer = strategy.init_buffer,
        term_buffer = strategy.term_buffer, base_window = strategy.base_window
    )

    # `sample_checkpoint` sets the sampling DiskStore stride (= batch / checkpoint frequency,
    # independent of the optimization checkpoint stride); falls back to `chunk_size` when off.
    stride = strategy.sample_checkpoint > 0 ? strategy.sample_checkpoint : strategy.chunk_size
    if strategy.sample_checkpoint > 0
        # Post-warmup per-batch checkpoint: render the latest draw and save FITS+PNG+resid.
        cb = function (info)
            params = Comrade.Adapt.adapt(Array, info.params)
            save_checkpoint(post_cpu, params, gimg, imgbase, "sample_round$(info.round)")
            ndiv = count(info.numerical_error)
            @info "sampling batch $(info.round)/$(info.nrounds): n_divergences=$ndiv (checkpoint saved)"
            return (; info.round, n_divergences = ndiv)
        end
        # Warmup runs as a separate phase that bypasses the DiskStore, so it needs its own
        # callback to also render checkpoints during the adaptation rounds. The warmup `info`
        # carries `phase`/`round`/`params` (no `numerical_error`); `params` is already host-side.
        wcb = function (info)
            params = Comrade.Adapt.adapt(Array, info.params)
            save_checkpoint(post_cpu, params, gimg, imgbase, "warmup_$(info.phase)_round$(info.round)")
            @info "warmup round $(info.round)/$(info.nrounds) phase=$(info.phase) step_size=$(info.step_size) (checkpoint saved)"
            return (; info.round, info.phase, info.step_size)
        end
        disk = DiskStore(; name = mkpath(out), stride = stride, callback = cb)
        trace = sample(
            rpost, smplr, strategy.nsample;
            saveto = disk, initial_params = xopt, restart = restart, warmup_callback = wcb
        )
    else
        disk = DiskStore(mkpath(out), stride)
        trace = sample(rpost, smplr, strategy.nsample; saveto = disk, initial_params = xopt, restart = restart)
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

    # On the Reactant path, verify the device posterior reproduces the CPU/Enzyme
    # log-density AND gradient before fitting — a broken device model silently yields garbage
    # fits (e.g. blown-up leakage). `post` here carries the Enzyme AD mode needed for the
    # CPU reference. Errors on mismatch.
    if strategy.use_reactant && strategy.verify_reactant
        @info "Verifying Reactant device posterior against the CPU/Enzyme reference"
        check_reactant_consistency(post, Comrade.prepare_device(post, Comrade.ComradeBase.ReactantEx()); rng = rng)
    end

    if strategy.benchmark
        _run_benchmarks(post, strategy)
    end

    g = post.skymodel.grid.imgdomain
    gimg = refinespatial(g, 2)
    opt = _select_optimizer(strategy)

    # ---- optimization / start / restart ----------------------------------------------
    if restart
        @info "Restarting from $(out)_optimum_allres.jls"
        xopt = deserialize(out * "_optimum_allres.jls")[:xopt]
    elseif !isnothing(strategy.start)
        startx = deserialize(strategy.start)
        @info "Starting from $(strategy.start); logdensity = $(logdensityof(post, startx))"
        xopt = startx
        save_optimal(imgbase, post, xopt, gimg; label = "start")
        plot_residuals_png(imgbase * "_residuals_map.png", post, xopt)
        write_caltables(caltabbase, xopt)
        serialize(out * "_optimum_allres.jls", Dict(:xopt => xopt, :post => post))
    else
        xopt = _optimize_tempered(imgbase, skym, intm, data, imgdata, strategy, opt, rng)
        save_optimal(imgbase, post, xopt, gimg; label = "optimal")
        plot_residuals_png(imgbase * "_residuals_final_map.png", post, xopt)
        write_caltables(caltabbase, xopt)
        serialize(out * "_optimum_allres.jls", Dict(:xopt => xopt, :post => post))
    end

    # ---- sampling --------------------------------------------------------------------
    if strategy.use_reactant
        trace, range = _sample_reactant(out, skym, intm, data, imgdata, xopt, strategy, restart, gimg, imgbase)
    else
        trace, range = _sample_ahmc(out, post, tpost, xopt, strategy, rng, restart)
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
