# Metropolis–Hastings moves run between the Reactant NUTS chunks through Comrade's
# `between_chunks` hook: builders of Comrade `AbstractMove`s by kind (see `MoveSpec`).

_sky_metadata(post) = post.skymodel.metadata

# Log of the per-wavenumber factor `A_k(ρ) · rtnrm(ρ)` by which `genfield` multiplies the
# white coefficients of a field with spectral parameters `ρs`: `A_k` is the amplitude
# spectrum (`_log_raw_amplitude`) and `rtnrm = (Σ_k A_k² · dk)^(-1/2)`, with `k2` the squared
# wavenumbers of the plan and `dk` its cell area over 2π.
function _log_amplitude(base, ρs::Tuple, k2, dk)
    la = _log_raw_amplitude(base, ρs, k2)
    return la .- log(sum(exp.(2 .* la)) * dk) / 2
end

# `A_k = (1 + Σₙ (ρₙ² k²)ⁿ)^(-1/2)`
function _log_raw_amplitude(::MarkovRF, ρs, k2)
    q = one.(k2)
    for n in eachindex(ρs)
        q = q .+ (ρs[n]^2 .* k2) .^ n
    end
    return log.(q) ./ -2
end

# `A_k = (1 + ℓ² k²)^(-α/4)`
_log_raw_amplitude(::Matern, (ℓ, α), k2) = log1p.(ℓ^2 .* k2) .* (-α / 4)

# The Hartley transform `genfield` applies, `real(F v) + imag(F v)` with `F` the forward
# DFT; applying it twice multiplies by `length(v)`.
function _hartley(v)
    F = Comrade.VLBISkyModels.FFTW.fft(complex.(v))
    return real.(F) .+ imag.(F)
end

# Squared wavenumbers of the random-field plan and its cell area over 2π.
function _plan_wavenumbers(plan)
    return plan.kx .^ 2 .+ (plan.ky .^ 2)', step(plan.kx) * step(plan.ky) / (2π)
end

function _srf_base(post, kind)
    md = _sky_metadata(post)
    (hasproperty(md, :base) && md.base isa StationaryBase) ||
        error("move \"$kind\" needs a stationary random-field sky model (Markov RF, order < 0, or Matérn, order = 0)")
    return md.base
end

# Non-centered sky fields: arrays `X` with a scalar scale `σX` (the rule `rescale_fields`
# uses).
_scaled_fields(sky) = [
    k for k in keys(sky)
        if sky[k] isa AbstractArray && get(sky, Symbol(:σ, k), nothing) isa Real
]

# The path of sky field `field`, whose latent coordinates must be its white coefficients:
# the field moves rescale or shift them with Jacobians written for that case.
function _white_field(view, field)
    path = (:sky, field)
    r = Comrade.coords(view, path)
    y = randn(Random.Xoshiro(1), dimension(view.tbase))
    vec(Comrade.value(view, y, path)) == y[r] || error(
        "sky field $field is not an identity leaf of the $(_space_label(view)) space; " *
            "the moves need its latent coordinates to be the white coefficients themselves"
    )
    return path
end

_space_label(view) = isnothing(Comrade.space(view)) ? "flat" : "StdNormal"

# Wavenumber magnitude of each white coefficient of a stationary random sky field (its
# Fourier mode), in units of the longest baseline of the data.
function _field_wavenumbers(post)
    md = _sky_metadata(post)
    plan = md.base.plan
    # `plan.kx` is π × the DFT frequency in cycles per pixel
    pix = abs(step(md.grid.X))
    umax = maximum(d -> maximum(hypot.(Comrade.datatable(d).baseline.U, Comrade.datatable(d).baseline.V)), post.data)
    return hypot.(plan.kx, plan.ky') ./ (π * pix * umax)
end

# 1.0 at the white coefficients of field `f` that a field move compensates, 0.0 elsewhere:
# every coefficient for `band = Inf`, else those with wavenumber at most `band` times the
# longest baseline.
function _band_mask(post, θ, f, band, kind)
    isinf(band) && return ones(size(θ.sky[f]))
    _srf_base(post, kind)
    kb = _field_wavenumbers(post)
    size(kb) == size(θ.sky[f]) ||
        error("move \"$kind\": field $f is not on the random-field plan grid $(size(kb))")
    m = Float64.(kb .<= band)
    any(isone, m) || error("move \"$kind\": band_limit = $band keeps no coefficient of field $f")
    return m
end

function _selected(spec::MoveSpec, available, what)
    isnothing(spec.params) && return available
    bad = setdiff(spec.params, String.(available))
    isempty(bad) || error(
        "move \"$(spec.kind)\": params $(bad) are not among the $what $(String.(available))"
    )
    return [a for a in available if String(a) in spec.params]
end

_reshaped(v, w) = reshape(vec(w), size(v))

# --- builders by kind ---------------------------------------------------------------------

function _flux_gain_moves(post, view, θ, spec)
    isnothing(spec.params) ||
        error("move \"flux_gain\" takes no params; it trades sky.flux.ftot against the gain log-amplitudes")
    (haskey(θ.sky, :flux) && haskey(θ.sky.flux, :ftot)) || error(
        "move \"flux_gain\" needs a sampled total flux (a two-value [flux] ftot in the sky " *
            "config); the sky parameters are $(keys(θ.sky))"
    )
    # A per-station constant lg1μ carries a common shift for the cost of its own prior; the
    # lg1 chain would have to shift every integration of every site.
    gains = if haskey(θ.instrument, :lg1μ)
        (:instrument, :lg1μ)
    else
        haskey(θ.instrument, :lg1) || error(
            "move \"flux_gain\" needs a gain log-amplitude `lg1μ` or `lg1`; the instrument " *
                "parameters are $(keys(θ.instrument))"
        )
        _is_chain(θ.instrument.lg1) || error(
            "move \"flux_gain\" needs `lg1` to be a Gauss–Markov chain with fitted hyperparameters"
        )
        (:instrument, :lg1)
    end
    return [
        Comrade.flux_gain_move(
            view; flux = (:sky, :flux, :ftot), gains,
            initial_scale = something(spec.initial_scale, 0.05)
        ),
    ]
end

"""
    _field_scale_moves(post, view, θ, spec)

`"field_scale"` moves: trade each non-centered sky field `X` against its scale `σX`: the
latent coordinate of `σX` moves by `u` and each compensated coefficient `X_k → X_k · σX / σX′`,
with `log|det ∂x′/∂x| = n_c log(σX / σX′)` for `n_c` compensated coefficients.

With `spec.band_limit = Inf` every coefficient is compensated; the image depends on `σX .* X`
only, so the likelihood is unchanged and the move is accepted on the prior alone. A finite
`band_limit` (stationary random fields only) compensates only the coefficients with
wavenumber up to `band_limit` times the longest baseline. The rest keep their values, so the
image changes at those wavenumbers and the move is accepted on the full posterior; in exchange
the scale's conditional width given the coefficients is near `1/√(2n_c)` instead of
`1/√(2n)`. The default initial scale is `2.4 / √(2n_c)`.
"""
function _field_scale_moves(post, view, θ, spec)
    fields = _selected(spec, _scaled_fields(θ.sky), "non-centered sky fields")
    isempty(fields) && error(
        "move \"field_scale\" found no non-centered field (an array `X` with a scalar " *
            "scale `σX`) among the sky parameters $(keys(θ.sky))"
    )
    return map(fields) do f
        path = _white_field(view, f)
        spath = (:sky, Symbol(:σ, f))
        logratio(x, x′, ctx) = log(Comrade.value(ctx.view, x, spath) / Comrade.value(ctx.view, x′, spath))
        mask = _band_mask(post, θ, f, spec.band_limit, "field_scale")
        key = Symbol(:field_scale_mask_, f)
        nc = sum(mask)
        Comrade.CompensatedMove(
            "field_scale[$f]", view, spath, path,
            (vX, x, x′, ctx) -> vX .* exp.(_reshaped(vX, getfield(ctx, key)) .* logratio(x, x′, ctx));
            logdet = (vX, x, x′, ctx) -> nc * logratio(x, x′, ctx),
            initial_scale = something(spec.initial_scale, 2.4 / sqrt(2nc)),
            invariant = isinf(spec.band_limit), context = NamedTuple{(key,)}((mask,)),
            traceable = true
        )
    end
end

"""
    _rho_field_moves(post, view, θ, spec)

`"rho_field"` moves: trade one spectral parameter `ρₙ` of a stationary random field `X` (a
Markov RF correlation length, or the Matérn outer scale `ℓ = ρ₁` or slope `α = ρ₂`) against its
white coefficients: the latent coordinate of `ρₙ` moves by `u` and each compensated
coefficient is multiplied by `A_k(ρ) rtnrm(ρ) / (A_k(ρ′) rtnrm(ρ′))`, the ratio of the
factors `genfield` applies to it, with
`log|det ∂x′/∂x| = Σ_k log(A_k(ρ) rtnrm(ρ)) − log(A_k(ρ′) rtnrm(ρ′))` over the compensated
coefficients. Each term of each field is its own move.

With `spec.band_limit = Inf` every coefficient is compensated, so the field and the
likelihood are unchanged and the move is accepted on the prior alone. A finite `band_limit`
compensates only the coefficients with wavenumber up to `band_limit` times the longest
baseline and is accepted on the full posterior, as for `"field_scale"`.
"""
function _rho_field_moves(post, view, θ, spec)
    base = _srf_base(post, "rho_field")
    k2, dk = _plan_wavenumbers(base.plan)
    k2 = collect(k2)
    withrho = [f for f in _scaled_fields(θ.sky) if get(θ.sky, Symbol(:ρ, f), nothing) isa Tuple]
    fields = _selected(spec, withrho, "stationary random fields")
    isempty(fields) && error(
        "move \"rho_field\" found no field `X` with a scale `σX` and spectral parameters " *
            "`ρX` among the sky parameters $(keys(θ.sky))"
    )
    return mapreduce(vcat, fields) do f
        path = _white_field(view, f)
        rpath = (:sky, Symbol(:ρ, f))
        nterms = length(θ.sky[rpath[2]])
        length(Comrade.coords(view, rpath)) == nterms ||
            error("the spectral parameters ρ$f do not have one latent coordinate each")
        la(x, ctx) = _log_amplitude(base.ps, Comrade.value(ctx.view, x, rpath), ctx.k2, dk)
        mask = _band_mask(post, θ, f, spec.band_limit, "rho_field")
        key = Symbol(:rho_field_mask_, f)
        dla(x, x′, ctx) = getfield(ctx, key) .* (la(x, ctx) .- la(x′, ctx))
        map(1:nterms) do n
            Comrade.CompensatedMove(
                "rho_field[$f,$n]", view, rpath, path,
                (vX, x, x′, ctx) -> vX .* _reshaped(vX, exp.(dla(x, x′, ctx)));
                index = n, logdet = (vX, x, x′, ctx) -> sum(dla(x, x′, ctx)),
                initial_scale = something(spec.initial_scale, 0.05),
                invariant = isinf(spec.band_limit), context = merge((; k2), NamedTuple{(key,)}((mask,))),
                traceable = true
            )
        end
    end
end

"""
    _mean_field_moves(post, view, θ, spec)

`"mean_field"` moves: trade one mean-model parameter against the log-intensity field `a` of the PolExp stationary
random-field sky model: the parameter's latent coordinate moves by `u`, changing the mean image
`m → m′`, and the white coefficients of `a` move by the inverse of the linear map from
coefficients to `σa · δa`, applied to `log m − log m′`. Then `m′ exp(σa δa′) = m exp(σa δa)`
pixelwise and Stokes I, Q, U and V, the flux normalization and the centering are all
unchanged. The coefficient shift depends on the other coordinates only, so
`log|det ∂x′/∂x| = 0`.
"""
function _mean_field_moves(post, view, θ, spec)
    post.skymodel.f === _polexp_srf_sky || error(
        "move \"mean_field\" needs the PolExp stationary random-field sky model " *
            "(polrep = \"PolExp\", order ≤ 0)"
    )
    means = keys(get(θ.sky, :mean, NamedTuple()))
    isempty(means) && error("move \"mean_field\" needs a mean model with free parameters")
    params = _selected(spec, collect(means), "mean-model parameters")
    base = _srf_base(post, "mean_field")
    k2, dk = _plan_wavenumbers(base.plan)
    k2 = collect(k2)
    path = _white_field(view, :a)
    function logmean(x, ctx)
        md = _sky_metadata(ctx.view.tbase.lpost)
        return log.(baseimage(make_mean(md.meanmodel, md.grid, Comrade.value(ctx.view, x, (:sky, :mean)))))
    end
    compensate = function (va, x, x′, ctx)
        σ = Comrade.value(ctx.view, x, (:sky, :σa))
        la = _log_amplitude(base.ps, Comrade.value(ctx.view, x, (:sky, :ρa)), ctx.k2, dk)
        Δ = _hartley((logmean(x, ctx) .- logmean(x′, ctx)) ./ σ) .* exp.(.-la) ./ sqrt(length(la))
        return va .+ _reshaped(va, Δ)
    end
    return map(params) do p
        mpath = (:sky, :mean, p)
        length(Comrade.coords(view, mpath)) == 1 ||
            error("mean-model parameter $p does not have one latent coordinate")
        Comrade.CompensatedMove(
            "mean_field[$p]", view, mpath, path, compensate;
            initial_scale = something(spec.initial_scale, 0.01), context = (; k2), traceable = true
        )
    end
end

function _phase_sheet_moves(post, view, θ, spec)
    isnothing(spec.initial_scale) ||
        error("move \"phase_sheet\" takes discrete ±2π steps; remove initial_scale")
    terms = _selected(spec, collect(phase_chain_terms(θ)), "real-line phase chains")
    isempty(terms) && error(
        "move \"phase_sheet\" needs a real-line Gauss–Markov phase chain among $(PHASE_CHAIN_TERMS); the model has none"
    )
    return [Comrade.PhaseSheetMove(post, Tuple(terms); space = Comrade.space(view))]
end

"""
    _chain_hyper_moves(post, view, θ, spec)

`"chain_hyper"` moves: `Comrade.chain_hyper_moves` for each instrument Gauss–Markov chain
term with fitted hyperparameters, one move per term and hyperparameter field.
"""
function _chain_hyper_moves(post, view, θ, spec)
    available = [t for t in keys(θ.instrument) if _is_chain(θ.instrument[t]) && !isempty(θ.instrument[t].hyperparams)]
    terms = _selected(spec, available, "Gauss–Markov chains with fitted hyperparameters")
    isempty(terms) && error(
        "move \"chain_hyper\" needs an instrument Gauss–Markov chain with fitted " *
            "hyperparameters; the instrument parameters are $(keys(θ.instrument))"
    )
    return mapreduce(vcat, terms) do t
        Comrade.chain_hyper_moves(view, (:instrument, t); initial_scale = something(spec.initial_scale, 0.1))
    end
end

const _MOVE_BUILDERS = Dict(
    "flux_gain" => _flux_gain_moves,
    "field_scale" => _field_scale_moves,
    "rho_field" => _rho_field_moves,
    "mean_field" => _mean_field_moves,
    "phase_sheet" => _phase_sheet_moves,
    "chain_hyper" => _chain_hyper_moves,
)

"""
    build_moves(post::VLBIPosterior, specs, θ0; space = nothing, output = nothing) -> Comrade.MoveSet

The moves the [`MoveSpec`](@ref)s `specs` describe for `post` in the latent space `space`
(`nothing` for flat, or `ProbabilityTransports.StdNormal()`), as a `Comrade.MoveSet`, the
`between_chunks` hook of the Reactant NUTS sampler. A spec expands to one move per field
(`"field_scale"`), per field and spectral parameter (`"rho_field"`), or per mean-model
parameter (`"mean_field"`); `"flux_gain"` is `Comrade.flux_gain_move` on `sky.flux.ftot`
and `instrument.lg1`, `"phase_sheet"` is one `Comrade.PhaseSheetMove` over the
real-line phase chains (`$(PHASE_CHAIN_TERMS)`), and `"chain_hyper"` expands to one
`Comrade.ChainHyperMove` per chain term and hyperparameter field. Each move takes its spec's `rounds` and
`target_accept`; `output` is where the `MoveSet` writes its statistics.

Errors if a move does not apply to the model, or if at `θ0` (a constrained parameter point
of `post`) a move that should leave the likelihood unchanged does not.
"""
function build_moves(post::VLBIPosterior, specs, θ0; space = nothing, output = nothing)
    θ0 = Comrade.Adapt.adapt(Array, θ0)
    view = Comrade.CoordinateView(post, space)
    moves, rounds, target_accept = Comrade.AbstractMove[], Int[], Float64[]
    for spec in specs
        haskey(_MOVE_BUILDERS, spec.kind) ||
            error("unknown move kind \"$(spec.kind)\". Allowed: $(collect(MOVE_KINDS))")
        ms = _MOVE_BUILDERS[spec.kind](post, view, θ0, spec)
        append!(moves, ms)
        append!(rounds, fill(spec.rounds, length(ms)))
        append!(target_accept, fill(spec.target_accept, length(ms)))
    end
    return Comrade.MoveSet(post, moves; space, θ0, rounds, target_accept, output)
end
