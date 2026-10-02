# The DILI Markov step in the StdNormal latent coordinates, where the target is
# `exp(-Φ(u)) N(u; 0, I)`.

"""
    DILIProposal(s::LikelihoodSubspace; langevin_complement = false)

The subspace of `s` in full latent coordinates: `V` (`n × r`, zero rows outside `s.free`),
eigenvalues `λ` and the 0/1 `mask` of the free coordinates. With `langevin_complement` the
complement move of [`dili_step`](@ref) carries a gradient drift.
"""
struct DILIProposal{TV, TL, TM}
    V::TV
    λ::TL
    mask::TM
    langevin_complement::Bool
end

function DILIProposal(s::LikelihoodSubspace; langevin_complement::Bool = false)
    V = zeros(s.n, length(s))
    V[s.free, :] = s.V
    mask = zeros(s.n)
    mask[s.free] .= 1
    return DILIProposal(V, collect(Float64, s.λ), mask, langevin_complement)
end

_cn(δ) = ((2 - δ) / (2 + δ), sqrt(8δ) / (2 + δ))

"""
    dili_step(fg, p::DILIProposal, u, Φu, gu, δr, δc, ξr, ξ, logu) -> (u′, Φ′, g′, logα)

One Metropolis–Hastings step on the target `exp(-Φ(u)) N(u; 0, I)` over the free coordinates
of `p`, with `fg(u) = (Φ(u), ∇Φ(u))` and `(Φu, gu) = fg(u)`. Write `u = V a + u⊥` on the free
coordinates, `γ = 1 ./ (1 .+ λ)` and `(s, β) = ((2 - δ)/(2 + δ), √(8δ)/(2 + δ))`. The proposal is

    a′  = s_r a - (1 - s_r) γ ⊙ (Vᵀ∇Φ(u) - λ ⊙ a) + β_r √γ ⊙ ξr
    u⊥′ = s_c u⊥ - κ (1 - s_c) g⊥(u) + β_c (I - V Vᵀ) ξ

with `g⊥ = (I - V Vᵀ)(mask ⊙ ∇Φ)`, `κ = 1` if `p.langevin_complement` and `0` otherwise:
Crank–Nicolson Langevin in the subspace with reference `N(0, diag γ)` and potential
`Φ - ½ Σ λ a²`, and Crank–Nicolson (Langevin when `κ = 1`) in the complement with reference
`N(0, I - V Vᵀ)`. Then

    log α = Φ(u) - Φ(u′) + ½‖a‖² - ½‖a′‖² + log q(a | u′) - log q(a′ | u)
            + κ [⟨g⊥, u⊥′ - s_c u⊥⟩ - ⟨g⊥′, u⊥ - s_c u⊥′⟩] / (1 + s_c)
            + κ (1 - s_c) (‖g⊥‖² - ‖g⊥′‖²) / (2 (1 + s_c)),

exact for any orthonormal `V` and any `λ ≥ 0`. The complement's prior and proposal terms
quadratic in `u⊥` cancel analytically and are not computed. The step is accepted when
`logu < logα`, so a NaN `logα` rejects; an infinite `Φ(u′)` gives `logα = -Inf`. Coordinates
outside the mask are unchanged. `δr, δc ≥ 0`: `δr = 0` freezes the subspace and `δc = 0`
the complement. The step is `dili_propose`, then `fg`, then `dili_accept`; each half runs on
host arrays and, compiled, on the device.
"""
function dili_step(fg, p::DILIProposal, u, Φu, gu, δr, δc, ξr, ξ, logu)
    a, c = p.V' * u, p.V' * gu
    u′, a′ = dili_propose(p, u, gu, a, c, δr, δc, ξr, ξ)
    Φ′, g′ = fg(u′)
    u″, Φ″, g″, logα, _, _ = dili_accept(p, u, Φu, gu, u′, a, a′, c, Φ′, g′, δr, δc, logu)
    return u″, Φ″, g″, logα
end

"""
    dili_propose(p::DILIProposal, u, gu, a, c, δr, δc, ξr, ξ) -> (u′, a′)

The proposal of [`dili_step`](@ref) from `u` with its subspace coordinates `a = Vᵀu` and
`c = Vᵀ∇Φ(u)`. `V` has zero rows outside the mask and orthonormal columns, so the products
`Vᵀξ` and `V (⋯)` give `u′`, and `a′ = Vᵀu′` is the proposed subspace coordinate itself.
"""
function dili_propose(p::DILIProposal, u, gu, a, c, δr, δc, ξr, ξ)
    V, λ, mask = p.V, p.λ, p.mask
    γ = inv.(1 .+ λ)
    sr, βr = _cn(δr)
    sc, βc = _cn(δc)
    b = V' * ξ
    a′ = sr .* a .- (1 - sr) .* γ .* (c .- λ .* a) .+ βr .* sqrt.(γ) .* ξr
    κ = p.langevin_complement ? 1 - sc : zero(sc)
    u′ = u .- (1 - sc) .* (mask .* u) .+ βc .* (mask .* ξ) .- κ .* (mask .* gu) .+
        V * (a′ .- sc .* a .- βc .* b .+ κ .* c)
    return u′, a′
end

"""
    dili_accept(p::DILIProposal, u, Φu, gu, u′, a, a′, c, Φ′, g′, δr, δc, logu)
        -> (u, Φ, ∇Φ, logα, Vᵀu, Vᵀ∇Φ) of the chain after the step

The Metropolis–Hastings decision of [`dili_step`](@ref) for the proposal `u′` (with `a′`) from
`u` (with `a`, `c`), returning the chain's next state with its subspace coordinates.
"""
function dili_accept(p::DILIProposal, u, Φu, gu, u′, a, a′, c, Φ′, g′, δr, δc, logu)
    V, λ, mask = p.V, p.λ, p.mask
    γ = inv.(1 .+ λ)
    sr, βr = _cn(δr)
    c′ = V' * g′
    gr = c .- λ .* a
    gr′ = c′ .- λ .* a′
    var = βr^2 .* γ
    logq_fwd = -sum(abs2.(a′ .- sr .* a .+ (1 - sr) .* γ .* gr) ./ var) / 2
    logq_bwd = -sum(abs2.(a .- sr .* a′ .+ (1 - sr) .* γ .* gr′) ./ var) / 2
    # with δr = 0 the subspace does not move and its proposal terms are absent (0/0 above)
    dq = ifelse(δr > 0, logq_bwd - logq_fwd, zero(logq_bwd))
    logα = Φu - Φ′ + (sum(abs2, a) - sum(abs2, a′)) / 2 + dq
    if p.langevin_complement
        sc, _ = _cn(δc)
        # complement inner products through ⟨mx - V Vᵀx, my - V Vᵀy⟩ = ⟨mx, my⟩ - ⟨Vᵀx, Vᵀy⟩
        um, u′m, gm, g′m = mask .* u, mask .* u′, mask .* gu, mask .* g′
        hw′ = sum(gm .* u′m) - sum(c .* a′)
        hw = sum(gm .* um) - sum(c .* a)
        h′w = sum(g′m .* um) - sum(c′ .* a)
        h′w′ = sum(g′m .* u′m) - sum(c′ .* a′)
        hh = sum(abs2, gm) - sum(abs2, c)
        h′h′ = sum(abs2, g′m) - sum(abs2, c′)
        logα += ((hw′ - sc * hw) - (h′w - sc * h′w′)) / (1 + sc) + (1 - sc) * (hh - h′h′) / (2 * (1 + sc))
    end
    logα = ifelse(Φ′ == Inf, oftype(logα, -Inf), logα)
    accept = logu < logα
    return ifelse.(accept, u′, u), ifelse(accept, Φ′, Φu), ifelse.(accept, g′, gu), logα,
        ifelse.(accept, a′, a), ifelse.(accept, c′, c)
end

"""
    DILISampler(fg, p::DILIProposal; δr = 1.0, δc = 1.0, target_accept = 0.5, split = false)
    DILISampler(k::DILIKernels, s::LikelihoodSubspace; langevin_complement = false, kwargs...)

Chain state and tuning for [`dili_step`](@ref) with `fg(u) = (Φ(u), ∇Φ(u))` on the host, or
with the device kernels of `k` (proposal and acceptance compiled once, separately from the
potential; positions, step sizes and random numbers are runtime inputs). The step sizes are
`c ⋅ (δr, δc)`, where `log c` follows a Robbins–Monro recursion toward `target_accept` during
warmup and is frozen for sampling. With `split`, each iteration is a subspace step `(δr, 0)`
followed by a complement step `(0, δc)`, each tuned by its own factor (`tuner`, `tuner_c`).
The recursions' state is the `tuner`/`tuner_c` keywords, which a new sampler can take over.
"""
struct DILISampler{F, S, P}
    fg::F
    step::S
    proposal::P
    δr::Float64
    δc::Float64
    target_accept::Float64
    tuner::MoveTuner
    split::Bool
    tuner_c::MoveTuner
end

function DILISampler(
        fg, p::DILIProposal;
        δr::Real = 1.0, δc::Real = 1.0, target_accept::Real = 0.5, tuner::MoveTuner = MoveTuner(1.0),
        split::Bool = false, tuner_c::MoveTuner = MoveTuner(1.0)
    )
    step = (args...) -> dili_step(fg, p, args...)
    return _dili_sampler(fg, step, p, δr, δc, target_accept, tuner, split, tuner_c)
end

function _dili_sampler(fg, step, p, δr, δc, target_accept, tuner, split, tuner_c)
    (δr >= 0 && δc >= 0) || throw(ArgumentError("δr and δc must be non-negative, got $δr and $δc"))
    (δr > 0 || δc > 0) || throw(ArgumentError("δr and δc cannot both be zero"))
    (split && (δr == 0 || δc == 0)) && throw(ArgumentError("a split sampler needs δr > 0 and δc > 0"))
    0 < target_accept < 1 || throw(ArgumentError("target_accept must be in (0, 1), got $target_accept"))
    return DILISampler(fg, step, p, Float64(δr), Float64(δc), Float64(target_accept), tuner, split, tuner_c)
end

function DILISampler(
        k::DILIKernels, s::LikelihoodSubspace;
        langevin_complement::Bool = false, δr::Real = 1.0, δc::Real = 1.0, target_accept::Real = 0.5,
        tuner::MoveTuner = MoveTuner(1.0), split::Bool = false, tuner_c::MoveTuner = MoveTuner(1.0)
    )
    s.n == dimension(k.tpost) ||
        throw(DimensionMismatch("the subspace has dimension $(s.n), the posterior $(dimension(k.tpost))"))
    hp = DILIProposal(s; langevin_complement)
    p = DILIProposal(_device(hp.V), _device(hp.λ), _device(hp.mask), langevin_complement)
    fg = u -> potential_gradient(k, u)
    # (Vᵀu, Vᵀ∇Φ) of the last state the step returned, keyed by that state's device array:
    # the chain passes it back unchanged, so each step makes three single-vector products
    # with V (on this hardware batching them into one product is slower).
    cache = Ref{Any}(nothing)
    step = function (u, Φu, gu, δr, δc, ξr, ξ, logu)
        num = Reactant.ConcreteRNumber
        du, dg = _device(u), _device(gu)
        cc = cache[]
        a, c = (!isnothing(cc) && cc[1] === du && cc[2] === dg) ? (cc[3], cc[4]) :
            _linalg_kernel(_subspace_coords, k, :subspace_coords, p, du, dg)
        u′, a′ = _linalg_kernel(dili_propose, k, :dili_propose, p, du, dg, a, c, num(δr), num(δc), _device(ξr), _device(ξ))
        Φ′, g′ = fg(u′)
        u″, Φ″, g″, logα, a″, c″ = _linalg_kernel(
            dili_accept, k, :dili_accept, p, du, num(Φu), dg, u′, a, a′, c, num(Φ′), g′, num(δr), num(δc), num(logu)
        )
        cache[] = (u″, g″, a″, c″)
        return u″, _host_scalar(Φ″), g″, _host_scalar(logα)
    end
    return _dili_sampler(fg, step, p, δr, δc, target_accept, tuner, split, tuner_c)
end

_subspace_coords(p::DILIProposal, u, g) = (p.V' * u, p.V' * g)

# The model is not traced into the step programs: the potential runs as its own kernel.
function _linalg_kernel(f, k::DILIKernels, name::Symbol, p::DILIProposal, args...)
    c = get!(k.compiled, (name, size(p.V), p.langevin_complement)) do
        Reactant.@compile sync = true f(p, args...)
    end
    return c(p, args...)
end

step_sizes(s::DILISampler) = (exp(s.tuner.logscale) * s.δr, exp((s.split ? s.tuner_c : s.tuner).logscale) * s.δc)

"""
    dili_start(s::DILISampler, u0) -> (u, Φ, g)

The chain state at `u0`: the position with its potential and gradient. Errors if the
potential is not finite.
"""
function dili_start(s::DILISampler, u0)
    n = size(s.proposal.V, 1)
    length(u0) == n || throw(DimensionMismatch("u0 has length $(length(u0)), the proposal $n"))
    Φu, gu = s.fg(u0)
    isfinite(Φu) || error("the potential at u0 is $Φu")
    return (u0, Φu, gu)
end

"""
    dili_advance!(onstep, s::DILISampler, state, phase, nsteps; rng = Random.default_rng())
        -> (; state, logα, accepted, scale, seconds, logα_c, accepted_c, scale_c)

Make `nsteps` steps from `state = (u, Φ, g)` in `phase` (`:warmup` tunes the step sizes,
`:sampling` does not) and return the final state with, per step, `logα`, the acceptance, the
step-size factor `scale` it used and its wall time. For a split sampler `logα`, `accepted`
and `scale` belong to the subspace steps and `logα_c`, `accepted_c`, `scale_c` to the
complement steps (empty otherwise). `onstep(t, state, accepted)` runs after step `t`, with
`accepted` the fraction of its moves accepted.
"""
function dili_advance!(
        onstep, s::DILISampler, state, phase::Symbol, nsteps::Integer;
        rng::Random.AbstractRNG = Random.default_rng()
    )
    phase in (:warmup, :sampling) || throw(ArgumentError("phase must be :warmup or :sampling, got $phase"))
    n, r = size(s.proposal.V)
    u, Φu, gu = state
    logαs = Vector{Float64}(undef, nsteps)
    accepted = falses(nsteps)
    scale = Vector{Float64}(undef, nsteps)
    seconds = Vector{Float64}(undef, nsteps)
    nc = s.split ? nsteps : 0
    logαs_c = Vector{Float64}(undef, nc)
    accepted_c = falses(nc)
    scale_c = Vector{Float64}(undef, nc)
    function move(δr, δc, tuner)
        logu = log(rand(rng))
        u, Φu, gu, logα = s.step(u, Φu, gu, δr, δc, randn(rng, r), randn(rng, n), logu)
        acc = logu < logα
        record!(tuner, phase, isnan(logα) ? 0.0 : min(1.0, exp(logα)), acc, s.target_accept)
        return logα, acc
    end
    for t in 1:nsteps
        t0 = time()
        scale[t] = exp(s.tuner.logscale)
        δr, δc = step_sizes(s)
        if s.split
            scale_c[t] = exp(s.tuner_c.logscale)
            logαs[t], accepted[t] = move(δr, 0.0, s.tuner)
            logαs_c[t], accepted_c[t] = move(0.0, δc, s.tuner_c)
        else
            logαs[t], accepted[t] = move(δr, δc, s.tuner)
        end
        seconds[t] = time() - t0
        onstep(t, (u, Φu, gu), s.split ? (accepted[t] + accepted_c[t]) / 2 : accepted[t])
    end
    return (; state = (u, Φu, gu), logα = logαs, accepted, scale, seconds, logα_c = logαs_c, accepted_c, scale_c)
end

"""
    dili_sample(s::DILISampler, u0, nwarmup, nsamples; rng = Random.default_rng(), record = Array)
        -> (; draws, logα, accepted, nnan)

Run `nwarmup` tuning steps then `nsamples` steps from `u0`. `draws[i] = record(u)` after
sampling step `i`; `logα` and `accepted` cover every step, warmup first. A proposal with a
NaN `logα` (a NaN potential or gradient, e.g. from overflow in the model) is rejected;
`nnan = (; warmup, sampling)` counts them and a nonzero count is logged. Errors if the
potential at `u0` is not finite.
"""
function dili_sample(
        s::DILISampler, u0, nwarmup::Integer, nsamples::Integer;
        rng::Random.AbstractRNG = Random.default_rng(), record = Array
    )
    w = dili_advance!((_...) -> nothing, s, dili_start(s, u0), :warmup, nwarmup; rng)
    draws = Vector{Any}(undef, nsamples)
    r = dili_advance!((t, st, _) -> (draws[t] = record(first(st))), s, w.state, :sampling, nsamples; rng)
    nnan = (; warmup = count(isnan, w.logα), sampling = count(isnan, r.logα))
    _log_nan(nnan)
    return (; draws = [d for d in draws], logα = vcat(w.logα, r.logα), accepted = vcat(w.accepted, r.accepted), nnan)
end

_log_nan(nnan) = (nnan.warmup + nnan.sampling > 0) &&
    @info "DILI: rejected $(nnan.warmup) warmup and $(nnan.sampling) sampling proposals with a NaN logα"
