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
    # run
    use_reactant::Bool = false
    benchmark::Bool = true
    # On the Reactant path, check the device log-density+gradient against the CPU/Enzyme
    # reference at a prior draw before optimizing, and error on mismatch (catches broken
    # device models that silently produce garbage fits).
    verify_reactant::Bool = false
    start::Union{Nothing, String} = nothing
    # Reactant sampling checkpointing (FITS + PNG + residuals): render every
    # `sample_checkpoint` samples (this is also the sampling DiskStore stride). 0 disables it.
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
    opt = get(cfg, "optimizer", Dict{String, Any}())
    temp = get(cfg, "tempering", Dict{String, Any}())
    samp = get(cfg, "sampler", Dict{String, Any}())
    run = get(cfg, "run", Dict{String, Any}())

    use_reactant = Bool(get(run, "use_reactant", false))

    opt_method = String(get(opt, "method", "Adam"))
    opt_method in ("Adam", "AdamW", "LBFGS") ||
        error("unknown optimizer.method '$opt_method'. Allowed: Adam, AdamW, LBFGS")

    startval = get(run, "start", "")
    start = (startval == "") ? nothing : String(startval)

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
        use_reactant = use_reactant,
        benchmark = Bool(get(run, "benchmark", true)),
        verify_reactant = Bool(get(run, "verify_reactant", false)),
        start = start,
        # `checkpoint` is accepted as an alias for `sample_checkpoint`.
        sample_checkpoint = Int(get(run, "sample_checkpoint", get(run, "checkpoint", 0))),
    )
end
