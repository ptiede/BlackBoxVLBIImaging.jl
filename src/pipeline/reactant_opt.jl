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

# One optimization step: gradient of the log-posterior, then an Optimisers update on the
# negated gradient (Optimisers minimizes; the log-posterior is to be maximized).
function _reactant_opt_step(opt_state, tpost, x)
    grad, val = _reactant_value_and_grad(tpost, x)
    new_state, new_x = Optimisers.update(opt_state, x, -1 .* grad)
    return new_state, new_x, -val
end

"""
    reactant_opt(post::VLBIPosterior, optimiser; initial_params=nothing, maxiters=10_000,
                 log_stride=250, checkpoint=0, outbase="", gimg=nothing,
                 rng=Random.default_rng()) -> (xopt, sol)

Optimize `post` on the Reactant device with an Optimisers.jl `optimiser` (e.g.
`Optimisers.Adam(η)`). Mirrors `comrade_opt`: pass `initial_params` (a NamedTuple in the
original parameter space) to warm-start, otherwise a prior sample is used. Returns the
optimum `xopt` as a host NamedTuple and `sol = (; objective = -logdensity)`.

The posterior is moved to the device internally via `prepare_device`; the per-step program
(value + Enzyme gradient + update) is compiled once and reused for all `maxiters` steps.

If `checkpoint > 0` and `outbase` is non-empty, every `checkpoint` iterations the current
image is written (FITS + PNG + residuals) via [`save_checkpoint`](@ref); `gimg` is the
render grid (defaults to the sky grid refined 2×).
"""
function reactant_opt(
        post::VLBIPosterior, optimiser; initial_params = nothing,
        maxiters = 10_000, log_stride = 250, checkpoint = 0, outbase = "",
        gimg = nothing, rng = Random.default_rng()
    )
    dpost = Comrade.prepare_device(post, Comrade.ComradeBase.ReactantEx())
    tpost = asflat(dpost)

    x0 = isnothing(initial_params) ? prior_sample(rng, dpost) : initial_params
    xr = Reactant.to_rarray(Comrade.inverse(tpost, x0))

    opt_state = Reactant.@jit Optimisers.setup(optimiser, xr)
    step_jit = Reactant.@compile sync = true _reactant_opt_step(opt_state, tpost, xr)

    docheck = checkpoint > 0 && !isempty(outbase)
    cgrid = isnothing(gimg) ? refinespatial(post.skymodel.grid.imgdomain, 2) : gimg

    xcur = xr
    state = opt_state
    loss = nothing
    for i in 1:maxiters
        state, xcur, loss = step_jit(state, tpost, xcur)
        if i == 1 || i % log_stride == 0 || i == maxiters
            @info "reactant_opt iter $i: -logdensity = $(Reactant.@allowscalar Float64(loss))"
        end
        if docheck && (i % checkpoint == 0 || i == maxiters)
            params = Comrade.Adapt.adapt(Array, Reactant.@jit Comrade.transform(tpost, xcur))
            save_checkpoint(post, params, cgrid, outbase, "opt_iter$(i)")
        end
    end

    xopt = Comrade.Adapt.adapt(Array, Reactant.@jit Comrade.transform(tpost, xcur))
    return xopt, (; objective = Reactant.@allowscalar Float64(loss))
end
