# Instrument models, written with Comrade's `@instrument`/`@jones` macros. Each TOML gain
# or leakage scheme is one `@instrument` definition: the `~` lines inside the `@jones` block
# declare exactly the parameters the parameterization reads (they sit on adjacent lines, so
# the old registry invariant — "the params tuple must match the `x.<field>` accesses" — is
# now enforced by proximity and by the macro itself).
#
# Every definition takes a single `priors` keyword: a NamedTuple of `ArrayPrior`s built from
# the instrument TOML (`_build_array_prior` in assemble.jl). A gain scheme alone is a
# complete `InstrumentModel`; a leakage scheme is a JonesD-only piece that
# `assemble_instrument` composes with the gain via a `JonesSandwich` (with the feed-rotation
# term), merging the two prior NamedTuples.

@inline sandwich_withfrcal(g, d, r) = adjoint(r) * g * d * r
@inline sandwich(g, d, r) = g * d * r

# --- gain schemes ------------------------------------------------------------------------

#     gain(; priors)
#
# Gain model for the phases and amplitudes of the R and L feeds, using a gain-ratio
# decomposition: the first feed is the reference and the second is the reference times a
# (hierarchically parameterized) gain ratio.
@instrument function gain(; priors)
    return @jones begin
        lg1 ~ priors.lg1
        gp1 ~ priors.gp1
        lgratμ ~ priors.lgratμ
        lgratσ ~ priors.lgratσ
        lgrat ~ priors.lgrat
        gprat ~ priors.gprat
        gpratμ ~ priors.gpratμ
        g1 = exp(lg1 + 1im * gp1)
        g2 = g1 * exp((lgratμ + lgratσ * lgrat) + 1im * (gprat + gpratμ))
        return JonesG((g1, g2))
    end
end

#     gain_centered(; priors)
#
# Same as [`gain`](@ref) but with the gain ratio centered on zero (no fitted ratio mean).
# Useful for data where the ratio has nominally been corrected.
@instrument function gain_centered(; priors)
    return @jones begin
        lg1 ~ priors.lg1
        gp1 ~ priors.gp1
        lgrat ~ priors.lgrat
        gprat ~ priors.gprat
        g1 = exp(complex(lg1, gp1))
        g2 = g1 * exp(complex(lgrat, gprat))
        return JonesG((g1, g2))
    end
end

#     gain_scanjitter(; priors)
#
# Like [`gain`](@ref) but the feed-1 phase is split into a scan-segmented anchor (`gp1`,
# typically flat) plus a tightly-priored integration-level jitter (`gpj`). The tight jitter
# prior keeps intra-scan visibility-phase evolution informative about the sky (per-integ
# free phases reduce the phase information to closure-only, leaving smooth sky modes on a
# near-degenerate ridge with the gains), while still absorbing residual post-fringe-fit
# phase drift within a scan.
@instrument function gain_scanjitter(; priors)
    return @jones begin
        lg1 ~ priors.lg1
        gp1 ~ priors.gp1
        gpj ~ priors.gpj
        lgratμ ~ priors.lgratμ
        lgratσ ~ priors.lgratσ
        lgrat ~ priors.lgrat
        gprat ~ priors.gprat
        gpratμ ~ priors.gpratμ
        g1 = exp(lg1 + 1im * (gp1 + gpj))
        g2 = g1 * exp((lgratμ + lgratσ * lgrat) + 1im * (gprat + gpratμ))
        return JonesG((g1, g2))
    end
end

#     gain_2timescale(; priors)
#
# Multi-timescale feed-1 gain. The log-amplitude is the sum of an across-scan term (`lgs`) and
# a subscan term (`lg1`). The phase is the sum of THREE terms: a wrapped absolute offset
# (`gp0`, a circular/von-Mises prior that carries the full-circle level), plus zero-mean
# across-scan (`gps`) and subscan (`gpj`) fluctuations. The phase needs the separate wrapped
# offset because the likelihood only sees `exp(iφ)`: giving the full-circle level a proper
# wrapped prior avoids the 2π multimodality that a plain (unwrapped) Gauss-Markov level would
# create, while the `gps`/`gpj` terms stay small and hence effectively unwrapped — give them
# `init = { kind = "fixed", value = 0.0 }` in the TOML so their level does not trade off
# against `gp0`. (If you want the phases modeled on the circle instead — a `WrappedBrownian`
# walk started uniform, which absorbs the offset itself and makes `gp0` redundant — use
# [`gain_wrappedphase`](@ref) rather than bolting it onto this scheme.)
# The amplitude has no periodicity, so `lgs` can carry its own level directly (no offset term).
# Terms are identifiable by well-separated correlation times (fix the subscan τ short) and by
# segmentation. Gain-ratio terms (`lgrat*`, `gprat*`) are single-timescale. Priors/segmentation
# for each term are set in the instrument TOML; this scheme only fixes how they combine.
@instrument function gain_2timescale(; priors)
    return @jones begin
        lgs ~ priors.lgs       # amplitude: across-scan (level + slow drift)
        lg1 ~ priors.lg1       # amplitude: subscan (fast within-scan)
        gp0 ~ priors.gp0       # phase: wrapped absolute offset (von Mises)
        gps ~ priors.gps       # phase: across-scan drift (zero-mean, pinned at scan 1)
        gpj ~ priors.gpj       # phase: subscan drift (zero-mean, pinned at its first time)
        lgratμ ~ priors.lgratμ
        lgratσ ~ priors.lgratσ
        lgrat ~ priors.lgrat
        gprat ~ priors.gprat
        gpratμ ~ priors.gpratμ
        g1 = exp((lgs + lg1) + 1im * (gp0 + gps + gpj))
        g2 = g1 * exp((lgratμ + lgratσ * lgrat) + 1im * (gprat + gpratμ))
        return JonesG((g1, g2))
    end
end

#     gain_gaussmarkov(; priors)
#
# Single-timescale (scan-level) Gauss-Markov gain, for SCAN-AVERAGED data where there is no
# subscan structure to model — the across-scan half of `gain_2timescale` with the subscan
# terms (`lg1` subscan / `gpj`) removed. Feed-1 gains are:
#   amplitude:  lg1μ (per-station constant offset/mean, TrackSeg) + lg1 (scan-segmented,
#               zero-mean Gauss-Markov drift, given `init = { kind = "fixed", value = 0.0 }`
#               in the TOML). Pinning scan 1 = 0 lets lg1μ unambiguously own the per-station
#               level — i.e. we fit one constant amplitude offset per station plus a smooth
#               scan-to-scan drift about it.
#   phase:      gp0 (WRAPPED von Mises absolute offset, TrackSeg) + gps (scan-segmented,
#               zero-mean Gauss-Markov drift, likewise pinned at its first scan). The phase
#               needs the separate wrapped offset because the likelihood only sees exp(iφ): a
#               proper wrapped prior on the full-circle level avoids the 2π multimodality a
#               plain unwrapped level would create, while the pinned gps stays small and
#               effectively unwrapped. [`gain_wrappedphase`](@ref) is the modern alternative:
#               one `WrappedBrownian` chain started uniform, wrapped itself, which absorbs
#               the offset and so drops gp0 entirely.
# Gain-ratio terms are single-timescale: a TrackSeg mean (`lgratμ`, `gpratμ`) plus a scan-level
# Gauss-Markov deviation (`lgrat`, `gprat`). Unlike `gain_2timescale`, the amplitude ratio has
# NO separate hierarchical scale (`lgratσ`) — the Gauss-Markov `lgrat` already carries its own
# (fitted) marginal σ, so a `lgratσ` factor would double-count the scatter.
# Priors/segmentation for each term are set in the instrument TOML; this scheme only fixes
# how they combine.
@instrument function gain_gaussmarkov(; priors)
    return @jones begin
        lg1μ ~ priors.lg1μ     # amplitude: per-station constant offset/mean (track)
        lg1 ~ priors.lg1       # amplitude: scan-segmented drift (zero-mean, pinned at scan 1)
        gp0 ~ priors.gp0       # phase: wrapped absolute offset (von Mises, track)
        gps ~ priors.gps       # phase: scan-segmented drift (zero-mean, pinned at scan 1)
        lgratμ ~ priors.lgratμ
        lgrat ~ priors.lgrat
        gprat ~ priors.gprat
        gpratμ ~ priors.gpratμ
        g1 = exp(complex((lg1μ + lg1), (gp0 + gps)))
        g2 = g1 * exp(complex((lgratμ + lgrat), (gprat + gpratμ)))
        return JonesG((g1, g2))
    end
end

#     gain_wrappedphase(; priors)
#
# Gauss-Markov gain whose PHASES are a single wrapped random walk instead of a sum of
# real-line terms:
#   amplitude:  lg1 — ONE Gauss-Markov chain, the whole feed-1 log-amplitude. A stationary
#               `OrnsteinUhlenbeck` reverts to its own (fixed) mean, so with `μ = 0` the
#               prior says "gain ≈ nominal, with σ scatter correlated over τ" — which is
#               exactly what a-priori amplitude calibration asserts. No separate offset term.
#   phase:      gp1 — ONE chain, meant to be a `WrappedBrownian` process started uniform on
#               the circle (`init` defaults to that for a wrapped process).
#   ratio:      lgratμ (per-track mean) + lgrat (Gauss-Markov residual about it, carrying its
#               own fitted σ, so there is no separate `lgratσ` scale); gpratμ (per-track
#               gpratμ (per-track WRAPPED offset) + gprat (`WrappedOrnsteinUhlenbeck`).
# The two phase terms are treated differently ON PURPOSE. gp1 is atmospheric and wanders
# through many radians, so it must live on the circle: a `WrappedBrownian` started uniform.
# gprat is instrumental: stable up to slow drift, so it wants a process that is circular AND
# mean-reverting. A free wrapped walk has no restoring force and can drift to track the smooth
# e^{2iφ(t)} field-rotation modulation that identifies the leakage terms; a stationary process
# protects the d-terms in a way a Brownian one cannot. `WrappedOrnsteinUhlenbeck` is both, and
# its σ — rather than an unbounded walk — sets how far the ratio phase may stray.
# It does NOT replace `gpratμ`. Its stationary marginal is WN(μ, σ²) with μ fixed at 0 and not
# fittable, so it pins the ratio phase near zero, whereas each station's R-L offset is an
# arbitrary constant anywhere on the circle. σ cannot stand in for that: the process is only
# valid for σ well below π. So gpratμ carries the level (flat, wrapped) and gprat the drift
# about it — the circular analogue of lgratμ + lgrat, and for the same reason: the process
# cannot fit its own mean.
# The ratio keeps a fitted mean where the feed-1 amplitude does not, and the asymmetry is the
# point: a-priori calibration already puts the feed-1 gain near nominal, so shrinking it
# toward 0 is right, whereas the R-L offset is a real uncalibrated instrumental constant that
# needs a free per-station mean. `OrnsteinUhlenbeck` cannot supply one itself — its `mu` is
# fixed, not fittable — which is why lgratμ is a separate term.
# This is the phase counterpart of the amplitude logic, and the reason it needs its own
# scheme: the offset terms `gp0`/`gpratμ` that `gain_2timescale` and `gain_gaussmarkov` carry
# exist only because an unwrapped (OU) phase chain cannot represent the full-circle level
# without manufacturing 2π-shifted modes. A `WrappedBrownian` chain is exactly 2π-periodic
# and its stationary circular law IS uniform, so a uniform start absorbs that level itself —
# keeping a separate offset would just make the two redundant. For the same reason there is
# only one phase drift term: Brownian motion has no stationary marginal, so two additive
# wrapped walks would share one degenerate slow component rather than separating timescales
# the way two OU chains with well-separated τ do.
# Priors/segmentation for each term are set in the instrument TOML; this scheme only fixes
# how they combine.
@instrument function gain_wrappedphase(; priors)
    return @jones begin
        lg1 ~ priors.lg1       # amplitude: single mean-reverting chain (μ = 0 ⇒ nominal gain)
        gp1 ~ priors.gp1       # phase: single wrapped walk (carries level AND drift)
        lgratμ ~ priors.lgratμ # ratio amplitude: per-station mean (OU cannot fit its own)
        lgrat ~ priors.lgrat   # ratio amplitude: residual about that mean
        gpratμ ~ priors.gpratμ # ratio phase: per-station WRAPPED offset (arbitrary, on circle)
        gprat ~ priors.gprat   # ratio phase: mean-reverting circular drift about it
        g1 = exp(complex(lg1, gp1))
        g2 = g1 * exp(complex((lgratμ + lgrat), (gpratμ + gprat)))
        return JonesG((g1, g2))
    end
end

#     gain_offsetphase(; priors)
#
# `gain_wrappedphase` with the feed-1 phase SPLIT into a per-track circular offset and a
# per-stamp residual, the same decomposition the ratio phase uses (gpratμ + gprat):
#   amplitude:  lg1 — as in `gain_wrappedphase`.
#   phase:      gp1μ (per-track WRAPPED offset, one circular value per site) + gp1 (the
#               per-stamp phase relative to it — typically an iid near-uniform
#               DiagonalVonMises with `init = { kind = "fixed", value = 0.0 }`).
#   ratio:      identical to `gain_wrappedphase`.
# The split targets sampler geometry, not statistics: the posterior's soft direction is a
# near-common motion of all of a site's phases (trading against image structure), and here
# that motion is a single circular coordinate per site (gp1μ) instead of a coherent
# rotation of every per-stamp phase. Identifiability needs two pins, both set in the TOML:
# gp1's `init` fixes each site's first stamp so the offset cannot trade against a common
# shift of the residuals, and a refant on gp1μ fixes one site's offset, which anchors the
# global phase level the likelihood cannot see (a common shift of every site cancels in
# g_i·conj(g_j) on the parallel hands).
@instrument function gain_offsetphase(; priors)
    return @jones begin
        lg1 ~ priors.lg1       # amplitude: single mean-reverting chain (μ = 0 ⇒ nominal gain)
        gp1μ ~ priors.gp1μ     # phase: per-station WRAPPED offset (arbitrary, on circle)
        gp1 ~ priors.gp1       # phase: per-stamp value about it (first stamp pinned via init)
        lgratμ ~ priors.lgratμ # ratio amplitude: per-station mean (OU cannot fit its own)
        lgrat ~ priors.lgrat   # ratio amplitude: residual about that mean
        gpratμ ~ priors.gpratμ # ratio phase: per-station WRAPPED offset (arbitrary, on circle)
        gprat ~ priors.gprat   # ratio phase: mean-reverting circular drift about it
        g1 = exp(complex(lg1, gp1μ + gp1))
        g2 = g1 * exp(complex((lgratμ + lgrat), (gpratμ + gprat)))
        return JonesG((g1, g2))
    end
end

#     gain_hier(; priors)
#
# Hierarchical gain model: the feed-1 amplitude and the feed-2 gain ratio are each given by a
# mean, a standard deviation, and a standardized random variable. The phase scatter is not
# fit due to wrapping concerns.
@instrument function gain_hier(; priors)
    return @jones begin
        lg1μ ~ priors.lg1μ
        lg1σ ~ priors.lg1σ
        lg1 ~ priors.lg1
        gp1 ~ priors.gp1
        lgratμ ~ priors.lgratμ
        lgratσ ~ priors.lgratσ
        lgrat ~ priors.lgrat
        gpratμ ~ priors.gpratμ
        gpratσ ~ priors.gpratσ
        gprat ~ priors.gprat
        g1 = exp(complex(lg1μ + lg1σ * lg1, gp1))
        g2 = g1 * exp(complex(lgratμ + lgratσ * lgrat, gpratμ + gpratσ * gprat))
        return JonesG((g1, g2))
    end
end

#     gain_noratio(; priors)
#
# Gain model with no feed ratio: `g1 = g2 = exp(complex(lg, gp))`.
@instrument function gain_noratio(; priors)
    return @jones begin
        lg ~ priors.lg
        gp ~ priors.gp
        g = exp(complex(lg, gp))
        return JonesG((g, g))
    end
end

#     singlegain(; priors)
#
# A single complex gain for total-intensity (Stokes I) fitting.
@instrument function singlegain(; priors)
    return @jones begin
        lg ~ priors.lg
        gp ~ priors.gp
        return SingleStokesGain(exp(complex(lg, gp)))
    end
end

# --- leakage schemes ---------------------------------------------------------------------

#     leakage_simple(; priors)
#
# Leakage given by a single complex number per feed.
@instrument function leakage_simple(; priors)
    return @jones begin
        d1re ~ priors.d1re
        d1im ~ priors.d1im
        d2re ~ priors.d2re
        d2im ~ priors.d2im
        return JonesD((complex(d1re, d1im), complex(d2re, d2im)))
    end
end

#     leakage_hier(; priors)
#
# Hierarchical leakage: each feed's real/imaginary parts are given by a mean, a standard
# deviation, and a standardized random variable.
@instrument function leakage_hier(; priors)
    return @jones begin
        d1reμ ~ priors.d1reμ
        d1reσ ~ priors.d1reσ
        d1re ~ priors.d1re
        d1imμ ~ priors.d1imμ
        d1imσ ~ priors.d1imσ
        d1im ~ priors.d1im
        d2reμ ~ priors.d2reμ
        d2reσ ~ priors.d2reσ
        d2re ~ priors.d2re
        d2imμ ~ priors.d2imμ
        d2imσ ~ priors.d2imσ
        d2im ~ priors.d2im
        dR = complex(d1reμ + d1reσ * d1re, d1imμ + d1imσ * d1im)
        dL = complex(d2reμ + d2reσ * d2re, d2imμ + d2imσ * d2im)
        return JonesD((dR, dL))
    end
end

# --- scheme registries + required-parameter probing ----------------------------------------

# TOML scheme name → @instrument constructor. `nothing` means no leakage.
const GAIN_SCHEMES = Dict{String, Any}(
    "gain" => gain,
    "gain_scanjitter" => gain_scanjitter,
    "gain_2timescale" => gain_2timescale,
    "gain_gaussmarkov" => gain_gaussmarkov,
    "gain_wrappedphase" => gain_wrappedphase,
    "gain_offsetphase" => gain_offsetphase,
    "gain_centered" => gain_centered,
    "gain_hier" => gain_hier,
    "gain_noratio" => gain_noratio,
    "singlegain" => singlegain,
)

const LEAKAGE_SCHEMES = Dict{String, Any}(
    "none" => nothing,
    "leakage_simple" => leakage_simple,
    "leakage_hier" => leakage_hier,
)

const SEGMENTATIONS = Dict{String, Any}(
    "integ" => IntegSeg(),
    "scan" => ScanSeg(),
    "track" => TrackSeg(),
)

# Records every `priors.<name>` access an @instrument constructor makes, so the required
# parameter set is derived from the definition itself instead of a hand-maintained tuple.
struct PriorProbe
    keys::Vector{Symbol}
end
Base.getproperty(p::PriorProbe, s::Symbol) = (push!(getfield(p, :keys), s); nothing)

"""
    required_params(ctor) -> Tuple{Vararg{Symbol}}

The instrument-prior parameter names an `@instrument` constructor reads from its `priors`
keyword (probed by constructing the model against a recording stand-in).
"""
function required_params(ctor)
    p = PriorProbe(Symbol[])
    ctor(; priors = p)
    return Tuple(getfield(p, :keys))
end
required_params(::Nothing) = ()
