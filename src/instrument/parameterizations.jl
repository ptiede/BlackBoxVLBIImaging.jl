# Gain and leakage parameterizations. Each one is registered, immediately below its
# definition, under the TOML name that selects it (`[gain]`/`[leakage]` `scheme` in the
# instrument TOML). The `params` tuple in the registration MUST list exactly the fields the
# parameterization destructures from `x` — they become the required `[priors.<param>]`
# entries and the keys of the instrument prior NamedTuple (see assemble.jl).

"""
    gain(x)

Simple gain model for the phases and the amplitudes of the R and L feeds.
We use a gain ratio decomposition where the first feed is is the reference and
the second feed is the reference multiplied by a gain ratio. The gain ratio is

g1 = exp(complex(lgR, gpR))
g2 = g1 * exp(complex(lgratμ + lgratσ * lgrat , gprat + gpratμ))

"""
@inline function gain(x)
    g1 = exp(x.lg1 + 1im * x.gp1)
    lgrat = x.lgratμ + x.lgratσ * x.lgrat
    gprat = x.gprat + x.gpratμ
    g2 = g1 * exp(lgrat + 1im * gprat)
    return g1, g2
end
register_gain_scheme!(
    "gain", gain; kind = :jones,
    params = (:lg1, :gp1, :lgratμ, :lgratσ, :lgrat, :gprat, :gpratμ),
)

"""
    gain_centered(x)

Same as `gain` but with the gain ratio centered on the mean, i.e. gpratμ is not added to gprat.
This is useful for data where the ratio has nominally been corrected
"""
@inline function gain_centered(x)
    g1 = exp(complex(x.lg1, x.gp1))
    g2 = g1 * exp(complex(x.lgrat, x.gprat))
    return g1, g2
end
register_gain_scheme!(
    "gain_centered", gain_centered; kind = :jones,
    params = (:lg1, :gp1, :lgrat, :gprat),
)

"""
    gain_hier(x)

A hierarchical gain model where the gain amplitude for feed 1 is given by a mean, std dev, and random variable
and the gain amplitude for feed 2 is given by the gain for feed 1 multiplied by a gain ratio which is given by a mean,
std dev, and random variable.

The phases are modeled similarly but the scatter is not fit due to concerns with wrapping
"""
function gain_hier(x)
    lg1 = x.lg1μ + x.lg1σ * x.lg1
    g1 = exp(complex(lg1, x.gp1))
    lgrat = x.lgratμ + x.lgratσ * x.lgrat
    gprat = x.gpratμ + x.gpratσ * x.gprat
    g2 = g1 * exp(complex(lgrat, gprat))
    return g1, g2
end
register_gain_scheme!(
    "gain_hier", gain_hier; kind = :jones,
    params = (:lg1μ, :lg1σ, :lg1, :gp1, :lgratμ, :lgratσ, :lgrat, :gpratμ, :gpratσ, :gprat),
)

"""
    gain_noratio(x)

A gain model with no ratios

g1 = g2 = exp(complex(lg, gp))
"""
@inline function gain_noratio(x)
    g = exp(complex(x.lg, x.gp))
    return g, g
end
register_gain_scheme!("gain_noratio", gain_noratio; kind = :jones, params = (:lg, :gp))

"""
    singlegain(x)

A single gain for total intensity
"""
@inline function singlegain(x)
    return exp(complex(x.lg, x.gp))
end
register_gain_scheme!("singlegain", singlegain; kind = :single, params = (:lg, :gp))

# "none" disables the leakage Jones matrix entirely (no parameterization, no priors).
register_leakage_scheme!("none", nothing; params = ())

"""
    leakage_simple(x)

A simple leakage model where the leakage is given by a single complex number for each feed.
"""
@inline function leakage_simple(x)
    dR = complex(x.d1re, x.d1im)
    dL = complex(x.d2re, x.d2im)
    return dR, dL
end
register_leakage_scheme!(
    "leakage_simple", leakage_simple;
    params = (:d1re, :d1im, :d2re, :d2im),
)

"""
    leakage_hier(x)

A hierarchical leakage model where the leakage for each feed is given by a mean, std dev, and random variable.
"""
@inline function leakage_hier(x)
    dR = complex(x.d1reμ + x.d1reσ * x.d1re, x.d1imμ + x.d1imσ * x.d1im)
    dL = complex(x.d2reμ + x.d2reσ * x.d2re, x.d2imμ + x.d2imσ * x.d2im)
    return dR, dL
end
register_leakage_scheme!(
    "leakage_hier", leakage_hier;
    params = (:d1reμ, :d1reσ, :d1re, :d1imμ, :d1imσ, :d1im, :d2reμ, :d2reσ, :d2re, :d2imμ, :d2imσ, :d2im),
)

@inline sandwich_withfrcal(g, d, r) = adjoint(r) * g * d * r
@inline sandwich(g, d, r) = g * d * r
