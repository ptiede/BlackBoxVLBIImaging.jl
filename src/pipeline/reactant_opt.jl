# Reactant-based optimization. Comrade's `comrade_opt` (Optimization.jl) has no Reactant
# backend, so for the Reactant path we drive Optimisers.jl directly and let Reactant compile
# the whole step — value, Enzyme reverse-mode gradient, and the optimizer update — into a
# single XLA program (the pattern from Comrade's NeuralFields example). `reactant_opt` is
# shaped like `comrade_opt(post, optimiser; initial_params, maxiters)` and returns
# `(xopt, sol)` with `xopt` a host NamedTuple, so it is a drop-in within the pipeline.

# Value + gradient of the log-posterior under Reactant/Enzyme. Returns
# `(grad_wrt_x, logdensity)`; `last(derivs)` is the derivative for the (non-Const) `x`.
function _reactant_value_and_grad(tpost, x)
    derivs, val = Enzyme.gradient(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        Comrade.logdensityof, Enzyme.Const(tpost), x
    )
    return last(derivs), val
end

# One optimization step: gradient of the log-posterior, then an *in-place* Optimisers update
# on the negated gradient (Optimisers minimizes; the log-posterior is maximized). Updating in
# place (`update!`) mutates `opt_state`/`x` and reuses their device buffers — the allocating
# `update` would hand back fresh device arrays every step, which Julia's GC can't see and the
# device runs out of memory. Returns the (host-readable) loss; `opt_state`/`x` are mutated.
function _reactant_opt_step!(opt_state, tpost, x)
    grad, val = _reactant_value_and_grad(tpost, x)
    Optimisers.update!(opt_state, x, -1 .* grad)
    return -val
end

"""
    reactant_opt(post::VLBIPosterior, optimiser; initial_params=nothing, maxiters=10_000,
                 ntrials=1, log_stride=250, rng=Random.default_rng()) -> (xopt, sol)

Optimize `post` on the Reactant device with an Optimisers.jl `optimiser` (e.g.
`Optimisers.Adam(η)`). Mirrors `comrade_opt`/`best_image`: with no `initial_params`, runs
`ntrials` random-restart optimizations (from independent prior draws) and keeps the one with
the best objective — this is what avoids the bad local minima that a single start falls
into. A supplied `initial_params` (e.g. a later tempering stage) is a single warm start.
Returns the optimum `xopt` as a host NamedTuple and `sol = (; objective = -logdensity)`.

The posterior is moved to the device internally via `prepare_device`; the per-step program
(value + Enzyme gradient + update) is compiled once and reused for every trial (the shapes
are identical, so restarts are essentially free). Convergence is reported via `@info` every
`log_stride` iterations; no per-iteration image checkpoints are written (rendering them on
the host each step is far slower than the device step itself).
"""
function reactant_opt(
        post::VLBIPosterior, optimiser; initial_params = nothing, maxiters = 10_000,
        ntrials = 1, log_stride = 250, gc_stride = 100, rng = Random.default_rng()
    )
    dpost = Comrade.prepare_device(post, Comrade.ComradeBase.ReactantEx())
    tpost = asflat(dpost)

    # Multi-start only from random prior draws; a supplied initial_params is one warm start.
    nstarts = isnothing(initial_params) ? max(ntrials, 1) : 1
    _start() = isnothing(initial_params) ? prior_sample(rng, dpost) : initial_params

    # Allocate the parameter buffer once and reuse it across trials. The step is compiled once
    # and `_reactant_opt_step!` mutates `opt_state`/`xr` in place every iteration (no per-step
    # allocation); between trials we overwrite `xr` in place (copyto!) and re-init the (small)
    # optimizer state fresh, since Adam carries no momentum across restarts.
    xr = Reactant.to_rarray(Comrade.inverse(tpost, _start()))
    opt_state = Reactant.@jit Optimisers.setup(optimiser, xr)
    step_jit = Reactant.@compile _reactant_opt_step!(opt_state, tpost, xr)

    best_params = nothing
    best_loss = Inf
    for t in 1:nstarts
        if t > 1
            # Reuse the parameter buffer: overwrite xr with a fresh start in place, and start
            # the optimizer state from scratch for this restart.
            copyto!(xr, Comrade.inverse(tpost, _start()))
            opt_state = Reactant.@jit Optimisers.setup(optimiser, xr)
        end
        loss = nothing
        for i in 1:maxiters
            loss = step_jit(opt_state, tpost, xr)
            if i == 1 || i % log_stride == 0 || i == maxiters
                @info "reactant_opt trial $t/$nstarts iter $i: -logdensity = $(Reactant.@allowscalar Float64(loss))"
            end
            i % gc_stride == 0 && GC.gc()
        end
        l = Reactant.@allowscalar Float64(loss)
        @info "reactant_opt trial $t/$nstarts final -logdensity = $l"
        # Materialize the winner to the host so the device buffer can be reused next trial.
        if l < best_loss
            best_loss = l
            best_params = Comrade.Adapt.adapt(Array, Reactant.@jit Comrade.transform(tpost, xr))
        end
        GC.gc()
    end
    # Fall back to the last point if every trial returned a non-finite objective.
    isnothing(best_params) &&
        (best_params = Comrade.Adapt.adapt(Array, Reactant.@jit Comrade.transform(tpost, xr)))

    return best_params, (; objective = best_loss)
end
