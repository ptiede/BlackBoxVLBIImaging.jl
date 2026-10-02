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
    # sky scale parameters held at their prior medians during optimization; non-empty also
    # rescales the optimum's fields to unit rms (see `build_fitting_config`)
    fix_scales::Vector{Symbol} = Symbol[]
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
    # Metropolis–Hastings moves run between the Reactant NUTS chunks (see
    # `SymmetryMoves`), `moves_per_chunk` rounds of every move each time. Empty = none.
    moves::Vector{String} = String[]
    moves_per_chunk::Int = 1
    # preconditioning: before sampling, fit a low-rank affine reparameterization of the
    # flat latent space from a pilot run's posterior draws (see `fit_preconditioner`),
    # and sample in the preconditioned coordinates. `precond_pilot` is the pilot's MCMC
    # DiskStore directory; `nothing` disables preconditioning.
    precond_pilot::Union{Nothing, String} = nothing
    precond_rank::Int = 16
    precond_nsamples::Int = 2000
    precond_discard::Float64 = 0.0
    # Carry the pilot's own preconditioner into the fit (see `fit_preconditioner`); set
    # this whenever the pilot itself sampled preconditioned, so rounds accumulate.
    precond_augment::Bool = false
    # In-run windowed refits (Reactant path): fractions of nadapt at which warmup pauses,
    # refits the preconditioner from the run's own warmup log, and continues in the new
    # coordinates with the metric frozen.
    # With this set, no pilot run is needed: launch once, segment 0 adapts Welford-style,
    # the refits take over. Empty disables.
    precond_refit_at::Vector{Float64} = Float64[]
    # Refit schedule: "manual" uses refit_at; "nutpie" refits every chunk to 30% of
    # warmup then every 8 chunks to 85% (Seyboldt+ §3 cadence); "stan" refits at
    # doubling gaps from step 100 to 85% — fewer refits, with exponentially longer
    # uninterrupted dual-averaging segments as warmup progresses (each refit restarts
    # dual averaging, so late refits must be sparse for the step size to stabilize).
    precond_refit_schedule::String = "manual"
    # What each in-run refit builds on (`Comrade.FisherLowRank` `carry`): "none" fits from
    # scratch, "rescale" keeps the current directions re-scaled on the window, "keep" keeps
    # the starting transform (`seed_transport`) exactly and refits only the directions added
    # to it.
    precond_refit_carry::String = "none"
    # What in-run refits fit: "fisher" (`Comrade.FisherLowRank`, from warmup draws and
    # scores) or "gauss_newton" (`Comrade.GaussNewtonLowRank`, from the likelihood's
    # Gauss–Newton curvature at the warmup draws; "stdnormal" space, Reactant only). For
    # "gauss_newton", `precond_rank` is the most directions a fit keeps, `precond_threshold`
    # the smallest curvature eigenvalue kept, `precond_oversample` the extra probe columns,
    # and `precond_probes_per_draw` the probe columns applied per warmup draw (0 = all).
    precond_refit_kind::String = "fisher"
    precond_threshold::Float64 = 100.0
    precond_oversample::Int = 10
    precond_probes_per_draw::Int = 0
    # Start in the latent space stored in a transport.jls (a low-rank preconditioner of the
    # run's latent space). With in-run refits they refine it instead of the score-init
    # diagonal; without, it is the fixed transform. Exclusive with `precond_pilot`.
    # "" = none.
    precond_seed_transport::String = ""
    # run
    # Latent space optimized and sampled in: "flat" (`asflat`) or "stdnormal" (the
    # ProbabilityTransports `StdNormal` transport, where the prior is N(0, I); needed by
    # priors with no flat transform, e.g. `AngularProjectedNormal`).
    latent_space::String = "flat"
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

`[optimizer] fix_scales` lists scalar sky parameters (e.g. `["σb", "σc", "σd"]`) that every
optimization stage holds at the median of their prior; the other parameters are optimized
as usual. After the last stage, every non-centered sky field `X` with scale `σX` (in
`polexp_markovrf`: `a`, `b`, `c`, `d`) is rescaled to `σX * rms(X)` and `X / rms(X)`, which
leaves the image unchanged and puts the white coefficients at the radius the prior's typical
set has (`rms(X) = 1`); see [`rescale_fields`](@ref). The sampler starts from that point
with every parameter free. Empty (the default) changes nothing. A name that is not a scalar
sky parameter of the model is an error when optimization starts.

`[sampler] moves` lists Metropolis–Hastings moves run between the Reactant NUTS chunks (see
[`SymmetryMoves`](@ref)):

  - `"flux_gain"`: total flux against a common gain log-amplitude `lg1`;
  - `"field_scale"`: each non-centered sky field against its scale `σX`;
  - `"rho_field"`: each correlation length of each Markov RF field against the field's
    white coefficients;
  - `"mean_field"`: each mean-model parameter against the log-intensity field `a` (PolExp
    Markov RF models);
  - `"phase_offset"`: a random-walk step of each free site's gain phase offset `gp1μ`;
  - `"phase_sheet"`: a `±2π` shift of a real-line Gauss–Markov phase chain from a point on
    (see [`PhaseSheetMoves`](@ref)); `"stdnormal"` space only.

In the `"stdnormal"` space only `"phase_sheet"` and `"mean_field"` run, alone or together.

The first four and `"phase_sheet"` leave the likelihood unchanged. `[sampler] moves_per_chunk` (an integer ≥ 1,
default 1) is the number of rounds of every move made between two chunks. Absent or empty
`moves` runs no moves. Unknown names, repeated names, moves on the AdvancedHMC path
(`use_reactant = false`), and `moves_per_chunk` without `moves` are errors.

`[run] latent_space` picks the space the optimizer and sampler work in: `"flat"` (default,
`asflat`) or `"stdnormal"` (the `StdNormal` transport, where the prior is exactly N(0, I);
required by priors without a flat transform such as `AngularProjectedNormal`). Moves other
than `"phase_sheet"` need the flat space; the low-rank preconditioner acts in front of
either. In the `"stdnormal"` space the start point's real-line phase chains are re-wrapped
to their shortest steps (see [`unwrap_phase_chains`](@ref)) before sampling.
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

    check_config_keys(
        opt, ("method", "maxiters", "ntrials", "g_tol", "eta", "fix_scales"), "[optimizer]"
    )
    check_config_keys(temp, ("noise_schedule",), "[tempering]")
    check_config_keys(
        samp,
        (
            "nsample", "nadapt", "step_size", "target_accept", "init_buffer",
            "term_buffer", "max_tree_depth", "chunk_size", "base_window",
            "adapt_mass_matrix", "moves", "moves_per_chunk",
        ),
        "[sampler]"
    )
    check_config_keys(
        prec,
        (
            "pilot", "rank", "nsamples", "discard", "augment",
            "refit_at", "refit_schedule", "refit_carry", "seed_transport",
            "refit_kind", "threshold", "oversample", "probes_per_draw",
        ),
        "[precondition]"
    )
    check_config_keys(
        run,
        (
            "use_reactant", "benchmark", "verify_reactant", "start", "sample_checkpoint",
            "checkpoint", "latent_space",
        ),
        "[run]"
    )

    use_reactant = Bool(get(run, "use_reactant", false))
    latent_space = String(get(run, "latent_space", "flat"))
    latent_space in ("flat", "stdnormal") ||
        error("unknown run.latent_space '$latent_space'. Allowed: flat, stdnormal")

    opt_method = String(get(opt, "method", "Adam"))
    opt_method in ("Adam", "AdamW", "LBFGS") ||
        error("unknown optimizer.method '$opt_method'. Allowed: Adam, AdamW, LBFGS")

    startval = get(run, "start", "")
    start = (startval == "") ? nothing : String(startval)

    fix_scales = get(opt, "fix_scales", String[])
    (fix_scales isa AbstractVector && all(v -> v isa AbstractString, fix_scales)) ||
        error("optimizer.fix_scales must be a list of sky parameter names, got $(repr(fix_scales))")
    allunique(fix_scales) || error("optimizer.fix_scales lists a name twice: $fix_scales")

    moves = get(samp, "moves", String[])
    (moves isa AbstractVector && all(v -> v isa AbstractString, moves)) ||
        error("sampler.moves must be a list of move names, got $(repr(moves))")
    unknown_moves = setdiff(moves, (SYMMETRY_MOVES..., "phase_sheet"))
    isempty(unknown_moves) ||
        error("unknown sampler.moves $(unknown_moves). Allowed: $(collect(SYMMETRY_MOVES)), phase_sheet")
    allunique(moves) || error("sampler.moves lists a move twice: $moves")
    isempty(moves) || use_reactant ||
        error("sampler.moves needs the Reactant sampler (run.use_reactant = true); got $moves")
    moves_per_chunk = get(samp, "moves_per_chunk", 1)
    (moves_per_chunk isa Integer && moves_per_chunk >= 1) || error(
        "sampler.moves_per_chunk must be an integer ≥ 1, got $(repr(moves_per_chunk))"
    )
    (haskey(samp, "moves_per_chunk") && isempty(moves)) &&
        error("sampler.moves_per_chunk is set but sampler.moves lists no moves")
    flat_moves = filter(m -> !(m in ("phase_sheet", "mean_field")), moves)
    (latent_space == "stdnormal" && !isempty(flat_moves)) &&
        error("sampler.moves $(flat_moves) act on the flat latent space; they cannot run with run.latent_space = \"stdnormal\"")
    ("phase_sheet" in moves && latent_space != "stdnormal") &&
        error("sampler.moves \"phase_sheet\" needs run.latent_space = \"stdnormal\" (the flat space's wrapped chains have no sheets)")

    pilotval = get(prec, "pilot", "")
    precond_pilot = (pilotval == "") ? nothing : String(pilotval)
    refit_carry = get(prec, "refit_carry", "none")
    refit_carry in ("none", "rescale", "keep") ||
        error("precondition.refit_carry must be \"none\", \"rescale\" or \"keep\", got $(repr(refit_carry))")
    (refit_carry == "keep" && get(prec, "seed_transport", "") == "") &&
        error("precondition.refit_carry = \"keep\" keeps the starting transform; set precondition.seed_transport")
    refit_kind = get(prec, "refit_kind", "fisher")
    refit_kind in ("fisher", "gauss_newton") ||
        error("precondition.refit_kind must be \"fisher\" or \"gauss_newton\", got $(repr(refit_kind))")
    gn_keys = filter(k -> haskey(prec, k), ["threshold", "oversample", "probes_per_draw"])
    if refit_kind == "gauss_newton"
        use_reactant || error("precondition.refit_kind = \"gauss_newton\" needs run.use_reactant = true")
        latent_space == "stdnormal" ||
            error("precondition.refit_kind = \"gauss_newton\" needs run.latent_space = \"stdnormal\"")
        haskey(prec, "rank") ||
            error("precondition.refit_kind = \"gauss_newton\" needs precondition.rank (the most directions a fit keeps)")
        refit_carry == "none" ||
            error("precondition.refit_carry applies to Fisher refits; remove it for refit_kind = \"gauss_newton\"")
        (haskey(prec, "refit_at") || get(prec, "refit_schedule", "manual") != "manual") ||
            error("precondition.refit_kind = \"gauss_newton\" needs refit_schedule or refit_at")
        get(prec, "pilot", "") == "" ||
            error("precondition.pilot fits a Fisher transform; use seed_transport to start a \"gauss_newton\" run from a stored one")
    else
        isempty(gn_keys) ||
            error("precondition.$(join(gn_keys, ", ")) apply to refit_kind = \"gauss_newton\" only")
    end
    return FittingStrategy(
        opt_method = opt_method,
        maxiters = Int(get(opt, "maxiters", 10_000)),
        ntrials = Int(get(opt, "ntrials", 5)),
        g_tol = Float64(get(opt, "g_tol", 0.1)),
        eta = Float64(get(opt, "eta", 0.001)),
        fix_scales = Symbol.(fix_scales),
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
        moves = String.(moves),
        moves_per_chunk = Int(moves_per_chunk),
        precond_pilot = precond_pilot,
        precond_rank = Int(get(prec, "rank", 16)),
        precond_nsamples = Int(get(prec, "nsamples", 2000)),
        precond_discard = Float64(get(prec, "discard", 0.0)),
        precond_augment = Bool(get(prec, "augment", false)),
        precond_refit_at = Float64.(get(prec, "refit_at", Float64[])),
        precond_refit_schedule = String(get(prec, "refit_schedule", "manual")),
        precond_refit_carry = refit_carry,
        precond_refit_kind = refit_kind,
        precond_threshold = Float64(get(prec, "threshold", 100.0)),
        precond_oversample = Int(get(prec, "oversample", 10)),
        precond_probes_per_draw = Int(get(prec, "probes_per_draw", 0)),
        precond_seed_transport = String(get(prec, "seed_transport", "")),
        latent_space = latent_space,
        use_reactant = use_reactant,
        benchmark = Bool(get(run, "benchmark", true)),
        verify_reactant = Bool(get(run, "verify_reactant", false)),
        start = start,
        # `checkpoint` is accepted as an alias for `sample_checkpoint`.
        sample_checkpoint = Int(get(run, "sample_checkpoint", get(run, "checkpoint", 0))),
    )
end

# The `space` argument of `Comrade.maybe_transport` for a strategy: `nothing` is the flat space.
latent_space(strategy::FittingStrategy) = strategy.latent_space == "stdnormal" ? PT.StdNormal() : nothing
