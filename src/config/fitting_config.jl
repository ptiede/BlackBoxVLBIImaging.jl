# The fitting-strategy config. Replaces the ad-hoc CLI flags that controlled optimization,
# the noise-tempering schedule, and the MCMC sampler (AdvancedHMC vs Reactant NUTS).

"""
    FittingStrategy

Typed description of how to fit: the optimizer, the fractional-noise tempering schedule,
the MCMC sampler and its tuning, and run-level flags (Reactant, benchmark, start).

Note: `restart` is deliberately NOT part of the strategy — resuming a run is a one-off
action passed at call time (the `--restart` CLI flag / `comrade_imager(...; restart=true)`),
not configuration to track in a TOML.
"""
Base.@kwdef struct FittingStrategy
    # optimizer
    opt_method::String = "Adam"     # Adam | AdamW | LBFGS (LBFGS is CPU-only)
    maxiters::Int = 10_000
    ntrials::Int = 5
    g_tol::Float64 = 0.1
    eta::Float64 = 0.001            # learning rate for the Optimisers.jl rules (Adam/AdamW)
    # tempering: fractional-noise level per optimization round (0.0 = full data)
    noise_schedule::Vector{Float64} = [0.05, 0.025, 0.0]
    nsample::Int = 10_000
    nadapt::Int = 5_000
    step_size::Float64 = 0.01
    target_accept::Float64 = 0.9
    init_buffer::Int = 200
    term_buffer::Int = 500
    max_tree_depth::Int = 10
    chunk_size::Int = 100
    base_window::Int = 25
    # Welford diagonal metric adaptation during warmup (Reactant path). Turn OFF for
    # preconditioned rounds: the metric adapts to marginal variances, which undoes any
    # transform component that trades marginal against conditional width (gradient
    # balance); with it off, the fitted preconditioner IS the metric and only the step
    # size adapts. Keep ON for pilot rounds with no preconditioner.
    adapt_mass_matrix::Bool = true
    # preconditioning: before sampling, fit a low-rank affine reparameterization of the
    # flat latent space from a pilot run's posterior draws (see `fit_preconditioner`),
    # and sample in the preconditioned coordinates. `precond_pilot` is the pilot's MCMC
    # DiskStore directory; `nothing` disables preconditioning.
    precond_pilot::Union{Nothing, String} = nothing
    precond_rank::Int = 16
    precond_nsamples::Int = 2000
    precond_min_scale::Union{Nothing, Float64} = nothing
    precond_discard::Float64 = 0.0
    # Carry the pilot's own preconditioner into the fit (see `fit_preconditioner`); set
    # this whenever the pilot itself sampled preconditioned, so rounds accumulate.
    precond_augment::Bool = false
    # Per-pair 2×2 whitening of angle-embedded (sin, cos) latent pairs (see
    # `_angle_pairs_from_draws`): aligns each well-measured phase's rotated wedge, which
    # otherwise pins the step size at the tightest tangential width.
    precond_angle_pairs::Bool = false
    # Balance marginal vs conditional widths per coordinate using gradient evaluations
    # at fit time (see `_grad_balance`); lifts the step-size cap set by ridge
    # coordinates (marginally wide, conditionally pinned). Only effective with
    # adapt_mass_matrix = false — Welford re-normalizes marginals and undoes it.
    precond_grad_balance::Bool = false
    # Stiff-direction corrections fit from the gradient covariance (see
    # `_stiff_from_grads`); 0 disables. Rotated conditionally-tight directions cap the
    # step size and are invisible to diagonals and draw fits. Like grad_balance, only
    # effective with adapt_mass_matrix = false.
    precond_stiff_rank::Int = 0
    # In-run windowed refits (Reactant path): fractions of nadapt at which warmup pauses,
    # refits the preconditioner from the run's own warmup log (augment carries the
    # current transform), and continues in the new coordinates with the metric frozen.
    # With this set, no pilot run is needed: launch once, segment 0 adapts Welford-style,
    # the refits take over. Empty disables.
    precond_refit_at::Vector{Float64} = Float64[]
    # Refit schedule: "manual" uses refit_at; "nutpie" refits every chunk to 30% of
    # warmup then every 8 chunks to 85% (Seyboldt+ §3 cadence); "stan" refits at
    # doubling gaps from step 100 to 85% — fewer refits, with exponentially longer
    # uninterrupted dual-averaging segments as warmup progresses (each refit restarts
    # dual averaging, so late refits must be sparse for the step size to stabilize).
    precond_refit_schedule::String = "manual"
    # Fisher-divergence estimator (Seyboldt, Carlson & Carpenter 2026): one joint fit
    # from draws AND scores, subsuming the wide/stiff/balance stages (angle_pairs still
    # composes on top). Exact on the sampled subspace, so short refit windows work and
    # augment is unnecessary. Requires adapt_mass_matrix = false to survive warmup.
    precond_fisher::Bool = false
    # Seed the windowed run from an existing transport.jls instead of the score-init
    # diagonal: the fit then REFINES a known-good transform (e.g. a manual precond run
    # that already found the wide image ridges) rather than discovering everything from
    # a stuck chain. "" = score-init as usual.
    precond_seed_transport::String = ""
    # run
    use_reactant::Bool = false
    benchmark::Bool = true
    # On the Reactant path, check the device log-density+gradient against the CPU/Enzyme
    # reference at a prior draw before optimizing, and error on mismatch (catches broken
    # device models that silently produce garbage fits).
    verify_reactant::Bool = false
    start::Union{Nothing, String} = nothing
    # Reactant checkpointing (FITS + PNG + residuals): render every `sample_checkpoint`
    # samples (this is also the sampling DiskStore stride). Warmup runs in chunks of the same
    # stride and is checkpointed too — the adaptation state is saved every chunk (so warmup is
    # resumable via `restart`) and the current draw is rendered. 0 disables it.
    # Optimization is deliberately NOT checkpointed — rendering on the host each step is far
    # slower than the device step.
    sample_checkpoint::Int = 0
end

"""
    build_fitting_config(cfg::AbstractDict) -> FittingStrategy

Parse a fitting-strategy TOML into a [`FittingStrategy`](@ref). Sections: `[optimizer]`,
`[tempering]`, `[sampler]` (NUTS tuning), `[run]`. The sampler is always NUTS; the backend
(AdvancedHMC NUTS vs Reactant NUTS) follows `run.use_reactant`.
"""
function build_fitting_config(cfg::AbstractDict)
    check_config_keys(
        cfg, ("optimizer", "tempering", "sampler", "precondition", "run"),
        "the fitting config (top level)"
    )
    opt = get(cfg, "optimizer", Dict{String, Any}())
    temp = get(cfg, "tempering", Dict{String, Any}())
    samp = get(cfg, "sampler", Dict{String, Any}())
    prec = get(cfg, "precondition", Dict{String, Any}())
    run = get(cfg, "run", Dict{String, Any}())

    check_config_keys(opt, ("method", "maxiters", "ntrials", "g_tol", "eta"), "[optimizer]")
    check_config_keys(temp, ("noise_schedule",), "[tempering]")
    check_config_keys(
        samp,
        (
            "nsample", "nadapt", "step_size", "target_accept", "init_buffer",
            "term_buffer", "max_tree_depth", "chunk_size", "base_window",
            "adapt_mass_matrix",
        ),
        "[sampler]"
    )
    check_config_keys(
        prec,
        (
            "pilot", "rank", "nsamples", "min_scale", "discard", "augment",
            "angle_pairs", "grad_balance", "stiff_rank", "refit_at", "fisher",
            "refit_schedule", "seed_transport",
        ),
        "[precondition]"
    )
    check_config_keys(
        run,
        ("use_reactant", "benchmark", "verify_reactant", "start", "sample_checkpoint", "checkpoint"),
        "[run]"
    )

    use_reactant = Bool(get(run, "use_reactant", false))

    opt_method = String(get(opt, "method", "Adam"))
    opt_method in ("Adam", "AdamW", "LBFGS") ||
        error("unknown optimizer.method '$opt_method'. Allowed: Adam, AdamW, LBFGS")

    startval = get(run, "start", "")
    start = (startval == "") ? nothing : String(startval)

    pilotval = get(prec, "pilot", "")
    precond_pilot = (pilotval == "") ? nothing : String(pilotval)
    precond_min_scale = haskey(prec, "min_scale") ? Float64(prec["min_scale"]) : nothing

    return FittingStrategy(
        opt_method = opt_method,
        maxiters = Int(get(opt, "maxiters", 10_000)),
        ntrials = Int(get(opt, "ntrials", 5)),
        g_tol = Float64(get(opt, "g_tol", 0.1)),
        eta = Float64(get(opt, "eta", 0.001)),
        noise_schedule = Float64.(get(temp, "noise_schedule", [0.05, 0.025, 0.0])),
        nsample = Int(get(samp, "nsample", 10_000)),
        nadapt = Int(get(samp, "nadapt", 5_000)),
        step_size = Float64(get(samp, "step_size", 0.01)),
        target_accept = Float64(get(samp, "target_accept", 0.9)),
        init_buffer = Int(get(samp, "init_buffer", 200)),
        term_buffer = Int(get(samp, "term_buffer", 500)),
        max_tree_depth = Int(get(samp, "max_tree_depth", 10)),
        chunk_size = Int(get(samp, "chunk_size", 100)),
        base_window = Int(get(samp, "base_window", 25)),
        adapt_mass_matrix = Bool(get(samp, "adapt_mass_matrix", true)),
        precond_pilot = precond_pilot,
        precond_rank = Int(get(prec, "rank", 16)),
        precond_nsamples = Int(get(prec, "nsamples", 2000)),
        precond_min_scale = precond_min_scale,
        precond_discard = Float64(get(prec, "discard", 0.0)),
        precond_augment = Bool(get(prec, "augment", false)),
        precond_angle_pairs = Bool(get(prec, "angle_pairs", false)),
        precond_grad_balance = Bool(get(prec, "grad_balance", false)),
        precond_stiff_rank = Int(get(prec, "stiff_rank", 0)),
        precond_refit_at = Float64.(get(prec, "refit_at", Float64[])),
        precond_fisher = Bool(get(prec, "fisher", false)),
        precond_refit_schedule = String(get(prec, "refit_schedule", "manual")),
        precond_seed_transport = String(get(prec, "seed_transport", "")),
        use_reactant = use_reactant,
        benchmark = Bool(get(run, "benchmark", true)),
        verify_reactant = Bool(get(run, "verify_reactant", false)),
        start = start,
        # `checkpoint` is accepted as an alias for `sample_checkpoint`.
        sample_checkpoint = Int(get(run, "sample_checkpoint", get(run, "checkpoint", 0))),
    )
end
