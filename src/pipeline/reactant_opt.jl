# Reactant-based optimization. Comrade's `comrade_opt` (Optimization.jl) has no Reactant
# backend, so for the Reactant path we drive Optimisers.jl directly and let Reactant compile
# the whole step — value, Enzyme reverse-mode gradient, and the optimizer update — into a
# single XLA program (the pattern from Comrade's NeuralFields example). `reactant_opt` is
# shaped like `comrade_opt(post, optimiser; initial_params, maxiters)` and returns
# `(xopt, sol)` with `xopt` a host NamedTuple, so it is a drop-in within the pipeline.

# Value + gradient of the log-posterior under Reactant/Enzyme. Returns
# `(grad_wrt_x, logdensity)`; `last(derivs)` is the derivative for the (non-Const) `x`.
function _reactant_value_and_grad(tpost, x)
    f(tpost, x) = -logdensityof(tpost, x)
    derivs, val = Enzyme.gradient(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        f, Enzyme.Const(tpost), x
    )
    return last(derivs), val
end

# One optimization step: gradient of the log-posterior, then an Optimisers update on the
# negated gradient (Optimisers minimizes; the log-posterior is maximized). We MUST return and
# reuse the updated `(opt_state, x)`: under Reactant's `@compile`, `update!` hands back fresh
# device arrays for the optimizer state instead of mutating the caller's in place, so
# discarding the return leaves the Adam moments pinned at their initial (zero) value forever.
# That silently degrades Adam to ~`η·sign(grad)` steps, which lets weakly-constrained
# parameters (e.g. a station's leakage with its intra-site baseline flagged) drift by ~η each
# iteration and blow up — while the loss still falls because the well-constrained sky
# dominates. The dead buffers are reclaimed by the periodic GC in `reactant_opt`.
function _reactant_opt_step!(opt_state, tpost, x)
    grad, val = _reactant_value_and_grad(tpost, x)
    new_state, new_x = Optimisers.update!(opt_state, x, grad)
    return new_state, new_x, val
end

"""
    check_reactant_consistency(post::VLBIPosterior, dpost; x=nothing, rtol=1e-2,
                               rng=Random.default_rng()) -> NamedTuple

Verify that the Reactant *device* posterior `dpost` (from `prepare_device(post, ...)`)
computes the same log-density **and gradient** as the CPU/Enzyme reference `post`, at a
parameter point `x` (flat; a prior draw if `nothing`). Errors if either disagrees beyond
`rtol`. This guards against silent device/AD discrepancies that yield garbage Reactant fits
while the CPU fit is fine — the cheapest way to catch a broken device model up front.

`post` MUST carry a working AD mode (the default-Enzyme posterior, e.g. the one built in
`comrade_imager` — NOT the `admode=nothing` posterior handed to `reactant_opt`): the CPU
reference uses Comrade's own `logdensity_and_gradient`, the same path `comrade_opt` uses.

The device value+gradient are read through the same compiled path the optimizer uses
(`_reactant_opt_step!`): a unit `Optimisers.Descent(1)` step leaves `x -> x + ∇logdensity`
and returns `-logdensity`, so both are recovered exactly without separately compiling the
value-and-gradient entry point (which overflows Julia's inference as a top-level `@compile`).
"""
function check_reactant_consistency(
        post::VLBIPosterior, dpost; x = nothing, rtol = 1.0e-2, rng = Random.default_rng()
    )
    tpost = asflat(post)      # CPU / Enzyme reference (post must carry an AD mode)
    tpostd = asflat(dpost)    # Reactant device
    xx = isnothing(x) ? Comrade.inverse(tpost, prior_sample(rng, post)) : x

    # CPU reference: log-density + gradient via Comrade's own Enzyme path (the one
    # `comrade_opt` uses); the bare `Enzyme.gradient` API crashes Enzyme on this posterior.
    vc, gc = Comrade.LogDensityProblems.logdensity_and_gradient(tpost, xx)
    gc = collect(gc)

    # Device: one Descent(1) step makes xr -> xr + ∇logdensity and returns -logdensity.
    xr = Reactant.to_rarray(copy(xx))
    st = Reactant.@jit Optimisers.setup(Optimisers.Descent(1), xr)
    step = Reactant.@compile _reactant_opt_step!(st, tpostd, xr)
    _, x1, loss = step(st, tpostd, xr)
    vd = -(Reactant.@allowscalar Float64(loss))
    gd = collect(Comrade.Adapt.adapt(Array, x1)) .- xx

    vrel = abs(vc - vd) / max(abs(vc), 1)
    gscale = max(maximum(abs, gc), eps())
    i = argmax(abs.(gc .- gd))
    grel = abs(gc[i] - gd[i]) / gscale
    @info "Reactant consistency: logdensity CPU=$vc device=$vd (reldiff=$vrel); " *
        "grad max reldiff=$grel (worst idx $i: cpu=$(gc[i]) device=$(gd[i]))"
    if vrel > rtol || grel > rtol
        error(
            "Reactant device posterior disagrees with the CPU/Enzyme reference beyond " *
            "rtol=$rtol: logdensity reldiff=$vrel, gradient max reldiff=$grel " *
            "(worst gradient component $i: cpu=$(gc[i]) device=$(gd[i])). The Reactant " *
            "model or its gradient is wrong — fits will be garbage. Aborting."
        )
    end
    return (; value_reldiff = vrel, grad_reldiff = grel)
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
        ntrials = 1, log_stride = 250, gc_stride = 100, verify = true, rtol = 1.0e-2,
        rng = Random.default_rng()
    )
    dpost = Comrade.prepare_device(post, Comrade.ComradeBase.ReactantEx())
    tpost = asflat(dpost)
    tpost_cpu = asflat(post)   # CPU reference (value-only; no AD mode needed)

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

    best_x_flat = nothing
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
            opt_state, xr, loss = step_jit(opt_state, tpost, xr)
            if i == 1 || i % log_stride == 0 || i == maxiters
                @info "reactant_opt trial $t/$nstarts iter $i: -logdensity = $(Reactant.@allowscalar Float64(loss))"
            end
            i % gc_stride == 0 && GC.gc()
        end
        l = Reactant.@allowscalar Float64(loss)
        @info "reactant_opt trial $t/$nstarts final -logdensity = $l"
        # Keep the winner's flat host vector so the device buffer can be reused next trial.
        if l < best_loss
            best_loss = l
            best_x_flat = collect(Comrade.Adapt.adapt(Array, xr))
        end
        GC.gc()
    end
    # Fall back to the last point if every trial returned a non-finite objective.
    isnothing(best_x_flat) && (best_x_flat = collect(Comrade.Adapt.adapt(Array, xr)))

    # Optimum consistency: compare the device objective to the CPU/Enzyme log-density at the
    # SAME flat optimum. The device can converge to a point it scores well but the CPU model
    # scores poorly — a parameter-dependent device discrepancy a prior-draw check misses (the
    # leakage prior is narrow, so prior draws never probe the large d-terms the device drifts
    # to). Both are transformed-space log-densities, so directly comparable; no `inverse`.
    if verify
        # Evaluate BOTH at the same point (best_x_flat). Note `best_loss` is the step's loss,
        # which is one Adam update *behind* `best_x_flat` (the step returns the value at x but
        # then mutates x), so it cannot be used here — recompute the device value at the point.
        xeval = Reactant.to_rarray(best_x_flat)
        ld_jit = Reactant.@compile Comrade.logdensityof(tpost, xeval)
        dev_ld = Reactant.@allowscalar Float64(ld_jit(tpost, xeval))
        cpu_ld = Comrade.logdensityof(tpost_cpu, best_x_flat)
        reldiff = abs(cpu_ld - dev_ld) / max(abs(cpu_ld), 1)
        @info "reactant_opt optimum check: CPU logdensity=$cpu_ld device=$dev_ld (reldiff=$reldiff)"
        if reldiff > rtol
            error(
                "Reactant device optimum disagrees with the CPU/Enzyme log-density at the same " *
                "point (CPU=$cpu_ld, device=$dev_ld, reldiff=$reldiff > rtol=$rtol). The device " *
                "model is wrong somewhere the optimizer reached — the fit would be garbage."
            )
        end
    end

    # Materialize the optimum on the HOST from the flat host vector via the CPU transform, so
    # every field is plain Float64 — the device transform leaves scalar params as
    # `ConcretePJRTNumber`, which break downstream CPU rendering/serialization.
    best_params = Comrade.transform(tpost_cpu, best_x_flat)
    return best_params, (; objective = best_loss)
end
