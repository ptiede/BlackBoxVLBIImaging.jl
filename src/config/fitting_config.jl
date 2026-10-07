# The fitting-strategy config. Replaces the ad-hoc CLI flags that controlled optimization,
# the noise-tempering schedule, and the MCMC sampler (AdvancedHMC vs Reactant NUTS).

const MOVE_KINDS = ("flux_gain", "field_scale", "rho_field", "mean_field", "phase_sheet", "chain_hyper")

"""
    MoveSpec

One `[[sampler.moves]]` table of the fitting config: the move `kind` (one of
`$(MOVE_KINDS)`), the parameters it acts on (`params`, `nothing` for all that apply), the
proposals per call (`rounds`), the warmup acceptance target, and the initial step scale of a
random-walk move (`nothing` for the kind's default). See [`build_moves`](@ref).
"""
Base.@kwdef struct MoveSpec
    kind::String
    params::Union{Nothing, Vector{String}} = nothing
    rounds::Int = 1
    target_accept::Float64 = 0.45
    initial_scale::Union{Nothing, Float64} = nothing
end

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
    # Metropolis–Hastings moves run between the Reactant NUTS chunks (see `build_moves`).
    moves::Vector{MoveSpec} = MoveSpec[]
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
    # `precond_probes_per_draw` the probe columns applied per warmup draw (0 = all), and
    # `precond_band_limit` the highest wavenumber of a stationary random sky field's white
    # coefficients the directions may use, in multiples of the longest baseline (Inf = all).
    precond_refit_kind::String = "fisher"
    precond_threshold::Float64 = 100.0
    precond_oversample::Int = 10
    precond_probes_per_draw::Int = 0
    precond_band_limit::Float64 = 3.0
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
`polexp_srf`: `a`, `b`, `c`, `d`) is rescaled to `σX * rms(X)` and `X / rms(X)`, which
leaves the image unchanged and puts the white coefficients at the radius the prior's typical
set has (`rms(X) = 1`); see [`rescale_fields`](@ref). The sampler starts from that point
with every parameter free. Empty (the default) changes nothing. A name that is not a scalar
sky parameter of the model is an error when optimization starts.

`[[sampler.moves]]` tables list Metropolis–Hastings moves run between the Reactant NUTS
chunks (see [`build_moves`](@ref)), each table one move kind:

```toml
[[sampler.moves]]
kind = "mean_field"
params = ["fwhm"]     # optional: a subset of the parameters the kind acts on
rounds = 10           # optional: proposals of each move per call (default 1)
target_accept = 0.45  # optional: warmup acceptance target (default 0.45)
initial_scale = 0.01  # optional: initial random-walk step scale (default per kind)
```

The kinds, all of which leave the likelihood unchanged:

  - `"flux_gain"`: total flux against a common shift of the gain log-amplitudes, `lg1μ` when
    the gain scheme has it and the `lg1` chain otherwise (no `params`);
  - `"field_scale"`: each non-centered sky field against its scale `σX` (`params`: fields);
  - `"rho_field"`: each spectral parameter of each stationary random field (Markov RF
    correlation lengths, Matérn outer scale and slope) against the field's white
    coefficients (`params`: fields);
  - `"mean_field"`: each mean-model parameter against the log-intensity field `a` (PolExp
    stationary random-field models; `params`: mean-model parameters);
  - `"phase_sheet"`: a `±2π` shift of a real-line Gauss–Markov phase chain from a point on
    (`params`: chain terms; no `initial_scale`); `"stdnormal"` space only;
  - `"chain_hyper"`: each fitted hyperparameter field (`σ`, `τ`, `D`) of a Gauss–Markov
    instrument chain against the chain's whitened innovations, with the chain values fixed;
    one move per term and field, which steps every site's hyperparameter and accepts each
    site on its own, so a round is one sweep over the sites (`params`: chain terms);
    `"stdnormal"` space only.

A kind may appear more than once only with disjoint `params`. Moves need the Reactant sampler
(`use_reactant = true`). Unknown kinds and keys are errors, as are the list form
`moves = [...]` and `moves_per_chunk`.

`[run] latent_space` picks the space the optimizer and sampler work in: `"flat"` (default,
`asflat`) or `"stdnormal"` (the `StdNormal` transport, where the prior is exactly N(0, I);
required by priors without a flat transform such as `AngularProjectedNormal`). The
low-rank preconditioner acts in front of either. In the `"stdnormal"` space the start point's real-line phase chains are re-wrapped
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
            "refit_kind", "threshold", "oversample", "probes_per_draw", "band_limit",
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

    haskey(samp, "moves_per_chunk") && error(
        "sampler.moves_per_chunk is not a fitting-config key; set `rounds` in each [[sampler.moves]] table"
    )
    moves = _parse_move_specs(get(samp, "moves", Any[]), latent_space)
    isempty(moves) || use_reactant ||
        error("sampler.moves needs the Reactant sampler (run.use_reactant = true)")

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
    gn_keys = filter(k -> haskey(prec, k), ["threshold", "oversample", "probes_per_draw", "band_limit"])
    band_limit = get(prec, "band_limit", 3.0)
    (band_limit isa Real && band_limit > 0) ||
        error("precondition.band_limit must be a positive multiple of the longest baseline (inf for no limit), got $(repr(band_limit))")
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
    return FittingStrategy(;
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
        moves,
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
        precond_band_limit = Float64(band_limit),
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

const _MOVE_KEYS = ("kind", "params", "rounds", "target_accept", "initial_scale")

function _parse_move_specs(tables, latent_space)
    (tables isa AbstractVector && all(t -> t isa AbstractDict, tables)) || error(
        "sampler.moves must be [[sampler.moves]] tables, each naming a move kind, e.g.\n" *
            "[[sampler.moves]]\nkind = \"mean_field\"\nrounds = 10\ngot $(repr(tables))"
    )
    specs = map(enumerate(tables)) do (i, t)
        where_ = "[[sampler.moves]] table $i"
        check_config_keys(t, _MOVE_KEYS, where_)
        kind = get(t, "kind", nothing)
        kind isa AbstractString || error("$where_ needs a kind (one of $(collect(MOVE_KINDS)))")
        kind in MOVE_KINDS || error("unknown move kind \"$kind\" in $where_. Allowed: $(collect(MOVE_KINDS))")
        params = get(t, "params", nothing)
        isnothing(params) || (params isa AbstractVector && !isempty(params) && all(p -> p isa AbstractString, params)) ||
            error("$where_: params must be a non-empty list of names, got $(repr(params))")
        rounds = get(t, "rounds", 1)
        (rounds isa Integer && rounds >= 1) || error("$where_: rounds must be an integer ≥ 1, got $(repr(rounds))")
        target_accept = get(t, "target_accept", 0.45)
        (target_accept isa Real && 0 < target_accept < 1) ||
            error("$where_: target_accept must lie in (0, 1), got $(repr(target_accept))")
        initial_scale = get(t, "initial_scale", nothing)
        isnothing(initial_scale) || (initial_scale isa Real && initial_scale > 0) ||
            error("$where_: initial_scale must be positive, got $(repr(initial_scale))")
        (kind == "phase_sheet" && !isnothing(initial_scale)) &&
            error("$where_: phase_sheet takes discrete ±2π steps and has no initial_scale")
        (kind == "flux_gain" && !isnothing(params)) && error("$where_: flux_gain takes no params")
        (kind == "phase_sheet" && latent_space != "stdnormal") && error(
            "$where_: phase_sheet needs run.latent_space = \"stdnormal\" (the flat space's wrapped chains have no sheets)"
        )
        (kind == "chain_hyper" && latent_space != "stdnormal") && error(
            "$where_: chain_hyper needs run.latent_space = \"stdnormal\" (its sites are accepted without the likelihood)"
        )
        MoveSpec(;
            kind = String(kind), params = isnothing(params) ? nothing : String.(params),
            rounds = Int(rounds), target_accept = Float64(target_accept),
            initial_scale = isnothing(initial_scale) ? nothing : Float64(initial_scale),
        )
    end
    for kind in unique(s.kind for s in specs)
        same = [s.params for s in specs if s.kind == kind]
        length(same) == 1 && continue
        (any(isnothing, same) || !allunique(reduce(vcat, same))) && error(
            "sampler.moves lists kind \"$kind\" more than once with overlapping params; give each table disjoint params"
        )
    end
    return MoveSpec[specs...]
end
