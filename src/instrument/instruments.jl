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
