# Partially non-centered spectral-parameter moves: one spectral parameter of a stationary sky
# field moves while the realized field stays fixed at a fixed pixel set and the field's white
# coefficients move minimally elsewhere.

# Raw Stokes I of the PolExp raster at `θ` (before the pulse and the re-centering shift).
function _raw_stokes_i(md, θ)
    δ = map((:a, :b, :c, :d)) do X
        getproperty(θ.sky, Symbol(:σ, X)) .* genfield(
            StationaryRandomField(field_spectrum(md.base.ps, getproperty(θ.sky, Symbol(:ρ, X))), md.base.plan),
            getproperty(θ.sky, X)
        )
    end
    m = parent(parent(baseimage(make_mean(md.meanmodel, md.grid, θ.sky.mean))))
    return m .* exp.(δ[1]) .* cosh.(sqrt.(δ[2] .^ 2 .+ δ[3] .^ 2 .+ δ[4] .^ 2))
end

# Rows of the map from white coefficients to the field at pixels `S`, without the amplitudes:
# genfield is real(F v) + imag(F v) of v = e^{la} ⊙ w, divided by √N.
function _hartley_rows(dims, S)
    nx, ny = dims
    HS = zeros(length(S), nx * ny)
    for (r, p) in enumerate(S), (j, k) in enumerate(CartesianIndices(dims))
        φ = 2π * ((p[1] - 1) * (k[1] - 1) / nx + (p[2] - 1) * (k[2] - 1) / ny)
        HS[r, j] = cos(φ) - sin(φ)
    end
    return HS
end

"""
    PNCFieldMove

A random-walk move of one spectral parameter (latent coordinate `ishift`) of the sky field
whose white coefficients are the latent coordinates `wcoords`. With `G(t) = σ/√N · H_S ·
diag(e^{la(t)})` the map from the white coefficients to the field at the pixels `S` and `t`
the parameter's latent coordinate, the coefficients follow the flow `dw/dt = −Gᵀ (G Gᵀ)⁻¹ Ġ w`,
the smallest change that keeps the field at `S` fixed. The flow is integrated by `nsub`
implicit-midpoint steps, which are exactly time-reversible; the generator has rank `|S|`, so
each step is a Woodbury update and its log-determinant comes from `|S| × |S|` matrices. The
field off `S` changes, so the move is accepted on the full density. `H_S` and the squared
wavenumbers are read from `ctx` (`pnc_HS`, `k2`), so the proposal is traceable.
"""
struct PNCFieldMove{R <: Tuple, S <: Tuple, P, C <: NamedTuple} <: Comrade.AbstractMove
    name::String
    ishift::Int
    rpath::R
    wcoords::Vector{Int}
    σpath::S
    ps::P
    dk::Float64
    nsub::Int
    initial_scale::Float64
    context::C
end

Comrade.step_kind(m::PNCFieldMove) = Comrade.RandomWalk(m.initial_scale)
Comrade.move_name(m::PNCFieldMove) = m.name
Comrade.is_invariant(::PNCFieldMove) = false
Comrade.context_data(m::PNCFieldMove, view) = m.context
Comrade.traceable(::PNCFieldMove) = true

function Comrade.propose(m::PNCFieldMove, x, u, ctx)
    h = u / m.nsub
    ld = zero(h)
    for _ in 1:m.nsub
        x, l = _pnc_substep(m, x, h, ctx)
        ld = ld + l
    end
    return x, ld
end

# One implicit-midpoint step of size `h` of the flow of `m` from `x`, and its log-determinant.
function _pnc_substep(m::PNCFieldMove, x, h, ctx)
    HS = ctx.pnc_HS
    s = size(HS, 1)
    Is = Matrix{Float64}(I, s, s)
    shifted(t) = Comrade._with_coordinate(x, m.ishift, Comrade._rget(x, m.ishift) + t)
    la(t) = vec(_log_amplitude(m.ps, Comrade.value(ctx.view, shifted(t), m.rpath), ctx.k2, m.dk))
    c = Comrade.value(ctx.view, x, m.σpath) / sqrt(size(HS, 2))
    w = x[m.wcoords]
    G = c .* HS .* exp.(la(h / 2))'
    Ġ = G .* ((la(h / 2 + 1.0e-5) .- la(h / 2 - 1.0e-5)) ./ 2.0e-5)'
    Vt = cholesky(Symmetric(G * G')) \ Ġ
    VU = Vt * G'
    rhs = w .- (h / 2) .* (G' * (Vt * w))
    w′ = rhs .- (h / 2) .* (G' * ((Is .+ (h / 2) .* VU) \ (Vt * rhs)))
    ld = logabsdet(Is .- (h / 2) .* VU)[1] - logabsdet(Is .+ (h / 2) .* VU)[1]
    x′ = shifted(h)
    x′[m.wcoords] = w′
    return x′, ld
end

"""
    PNCOrbitMove

Draws the next point of the orbit of `x` under one step of size `h` of the flow of a
[`PNCFieldMove`](@ref) (`flow`), rather than proposing one: with `T` that step, the orbit
points `xₖ = Tᵏ x` have weights `π(xₖ) |det ∂xₖ/∂x|`. A window of `K` consecutive orbit points
is placed uniformly among those that contain `x`, the weights are evaluated there, and one
point is drawn in proportion to them. This leaves `π` invariant (a generalized Gibbs step
along the orbit; Liu & Sabatti 2000), and the window spans `K h` in the spectral parameter's
latent coordinate. The step is entry `hidx` of the context array `pnc_orbit_h`, so it can
change between calls without recompiling. `target(x, ctx)` is the log density, by default
that of the base space of `ctx.view`.

    PNCOrbitMove(name, flow, h, K, target)

A single orbit move with step `h`.
"""
struct PNCOrbitMove{M <: PNCFieldMove, F, C <: NamedTuple} <: Comrade.AbstractMove
    name::String
    flow::M
    hidx::Int
    K::Int
    target::F
    context::C
end

PNCOrbitMove(name, flow, h::Real, K, target) =
    PNCOrbitMove(name, flow, 1, K, target, merge(flow.context, (; pnc_orbit_h = [Float64(h)])))

Comrade.step_kind(m::PNCOrbitMove) = Comrade.OrbitStep(m.K)
Comrade.move_name(m::PNCOrbitMove) = m.name
Comrade.is_invariant(::PNCOrbitMove) = false
Comrade.context_data(m::PNCOrbitMove, view) = m.context
Comrade.traceable(::PNCOrbitMove) = true

_base_logdensity(x, ctx) = Comrade.logdensityof(ctx.view.tbase, x)

# The step `o + v` places `x` at window position `o` (0-based) and draws with the uniform `v`.
# Loops start from copies: a traced loop may reuse the buffer of the array it starts from.
function Comrade.propose(m::PNCOrbitMove, x, step, ctx)
    h = Comrade._rget(ctx.pnc_orbit_h, m.hidx)
    o = floor(Int, step)
    v = step - o
    y = copy(x)
    @trace track_numbers = false for _ in 1:o
        y, _ = _pnc_substep(m.flow, y, -h, ctx)
    end
    start = copy(y)
    lw = fill!(similar(x, m.K), 0)
    ld = zero(h) * zero(eltype(x))
    @trace track_numbers = false for k in 1:m.K
        Comrade.ComradeBase.rsetindex!(lw, m.target(y, ctx) + ld, k)
        y, l = _pnc_substep(m.flow, y, h, ctx)
        ld = ld + l
    end
    lw = ifelse.(isnan.(lw), -Inf, lw)
    p = exp.(lw .- maximum(lw))
    j = sum(cumsum(p) .< v * sum(p))
    z = copy(start)
    @trace track_numbers = false for _ in 1:j
        z, _ = _pnc_substep(m.flow, z, h, ctx)
    end
    return z, ifelse(j == o, -Inf, Inf)
end

"""
    _pnc_field_moves(post, view, θ, spec)

`"pnc_field"` moves: one [`PNCFieldMove`](@ref) per spectral parameter of each selected
stationary random field (default: the PolExp polarization fields `b`, `c`, `d`), holding the
field fixed at the pixels where the raw Stokes I of `θ` exceeds 1e-2 of its peak.
"""
function _pnc_field_moves(post, view, θ, spec; kind = "pnc_field", default_fields = [:b, :c, :d])
    base = _srf_base(post, kind)
    post.skymodel.f === _polexp_srf_sky ||
        error("move \"$kind\" needs the PolExp stationary random-field sky model")
    md = _sky_metadata(post)
    k2, dk = _plan_wavenumbers(base.plan)
    k2 = collect(k2)
    fields = isnothing(spec.params) ? default_fields : _selected(spec, [:a, :b, :c, :d], "PolExp fields")
    Iraw = _raw_stokes_i(md, θ)
    S = findall(>(1.0e-2 * maximum(Iraw)), Iraw)
    HS = _hartley_rows(size(k2), S)
    @info "$kind: field held at $(length(S)) pixels (raw Stokes I above 1e-2 of its peak)"
    return mapreduce(vcat, fields) do f
        wcoords = collect(Comrade.coords(view, _white_field(view, f)))
        rpath = (:sky, Symbol(:ρ, f))
        rc = collect(Comrade.coords(view, rpath))
        map(eachindex(rc)) do n
            PNCFieldMove(
                "pnc_field[$f,$n]", rc[n], rpath, wcoords, (:sky, Symbol(:σ, f)), base.ps,
                dk, 8, Float64(something(spec.initial_scale, 0.05)), (; k2, pnc_HS = HS)
            )
        end
    end
end

"""
    _pnc_orbit_moves(post, view, θ, spec)

`"pnc_orbit"` moves: one [`PNCOrbitMove`](@ref) per spectral parameter of each selected PolExp
field (default: `b` and `c`), along the flow of the matching `"pnc_field"` move, with orbit
step `h = initial_scale` (default 0.1) to start from and a window of 32 orbit points.
"""
function _pnc_orbit_moves(post, view, θ, spec)
    h = Float64(something(spec.initial_scale, 0.1))
    flows = _pnc_field_moves(post, view, θ, spec; kind = "pnc_orbit", default_fields = [:b, :c])
    hs = fill(h, length(flows))
    return map(enumerate(flows)) do (i, f)
        PNCOrbitMove(
            replace(f.name, "pnc_field" => "pnc_orbit"), f, i, 32, _base_logdensity,
            merge(f.context, (; pnc_orbit_h = hs))
        )
    end
end

"""
    PNCAdaptation(moves, post, θ0, steps; target = 0.65)

A `between_chunks` hook that runs the [`Comrade.MoveSet`](@ref) `moves`, whose PNC moves were
built at the constrained point `θ0` of `post`, and adapts them during warmup:

  - it keeps the mean raw Stokes I image of the draws it sees, and at each warmup step in
    `steps` holds the PNC moves' fields fixed at the `M` brightest pixels of that mean
    instead, `M` the number held at `θ0`, then restarts the mean;
  - after each call it moves the log step of each [`PNCOrbitMove`](@ref) toward the fraction
    `target` of its draws leaving the current point, with gain `1/n^0.6` after `n` calls
    and the step kept in `[1e-3, 1]`.

Sampling keeps the last held set and steps.
"""
mutable struct PNCAdaptation{M, P}
    moves::M
    post::P
    held::Vector{CartesianIndex{2}}
    steps::Vector{Int}
    total::Matrix{Float64}
    n::Int
    orbit::Vector{Int}
    calls::Int
    target::Float64
end

function PNCAdaptation(moves, post, θ0, steps; target::Real = 0.65)
    0 < target < 1 || throw(ArgumentError("target must lie in (0, 1), got $target"))
    I0 = _raw_stokes_i(_sky_metadata(post), θ0)
    held = findall(>(1.0e-2 * maximum(I0)), I0)
    size(moves.ctx.pnc_HS, 1) == length(held) || throw(
        ArgumentError("the PNC moves hold $(size(moves.ctx.pnc_HS, 1)) pixels, but θ0 gives $(length(held))")
    )
    orbit = findall(m -> m isa PNCOrbitMove, collect(moves.moves))
    return PNCAdaptation(moves, post, held, collect(Int, steps), zero(I0), 0, orbit, 0, Float64(target))
end

function (h::PNCAdaptation)(state, tpost, info, rng)
    warmup = info.phase === :warmup
    warmup && !isempty(h.steps) && _observe_held!(h, state, info)
    moved = [h.moves.stats[j].counts[:warmup][2] for j in h.orbit]
    state = h.moves(state, tpost, info, rng)
    if warmup && !isempty(h.orbit)
        h.calls += 1
        hs = copy(h.moves.ctx.pnc_orbit_h)
        for (i, j) in pairs(h.orbit)
            rate = (h.moves.stats[j].counts[:warmup][2] - moved[i]) / h.moves.rounds[j]
            k = h.moves.moves[j].hidx
            hs[k] = clamp(hs[k] * exp((rate - h.target) / h.calls^0.6), 1.0e-3, 1.0)
        end
        Comrade.update_context!(h.moves, :pnc_orbit_h, hs)
        h.calls % 10 == 0 && @info "PNC orbit steps after warmup step $(info.step): " *
            join(("$(Comrade.move_name(h.moves.moves[j])) $(round(hs[h.moves.moves[j].hidx]; sigdigits = 3))" for j in h.orbit), ", ")
    end
    return state
end

function _observe_held!(h::PNCAdaptation, state, info)
    z = collect(Float64, vec(Array(state.position)))
    x = isnothing(info.pre) ? z : Comrade._affine_fwd(Comrade._hostify(info.pre), z)
    h.total .+= _raw_stokes_i(_sky_metadata(h.post), Comrade.transform(h.moves.ctx.view.tbase, x))
    h.n += 1
    info.step in h.steps || return h
    Imean = h.total ./ h.n
    held = CartesianIndices(Imean)[partialsortperm(vec(Imean), 1:length(h.held); rev = true)]
    kept = length(intersect(h.held, held)) / length(held)
    Comrade.update_context!(h.moves, :pnc_HS, _hartley_rows(size(Imean), held))
    @info "PNC moves at warmup step $(info.step): field held at the $(length(held)) brightest " *
        "pixels of the mean raw Stokes I over $(h.n) draws ($(round(Int, 100 * kept))% held before)"
    h.held = held
    fill!(h.total, 0)
    h.n = 0
    return h
end

# The `between_chunks` hook for `moves`: a `PNCAdaptation` when there are PNC moves, refreshing
# the held pixels at the warmup refits of `adaptor` (if any) and adapting the orbit steps.
function _moves_hook(moves, post, θ0, adaptor, nadapt, chunk)
    (isnothing(moves) || !haskey(moves.ctx, :pnc_HS)) && return moves
    steps = Comrade.metric_refit_steps(adaptor, nadapt, chunk)
    (isempty(steps) && !any(m -> m isa PNCOrbitMove, moves.moves)) && return moves
    return PNCAdaptation(moves, post, θ0, steps)
end
