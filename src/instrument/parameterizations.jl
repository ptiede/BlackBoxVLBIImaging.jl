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


"""
    gainhier(x)

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

"""
    gain_noratio(x)

A gain model with no ratios

g1 = g2 = exp(complex(lg, gp))
"""
@inline function gain_noratio(x)
    g = exp(complex(x.lg, x.gp))
    return g, g
end

"""
    singlegain(x)

A single gain for total intensity
"""
@inline function singlegain(x)
    return exp(complex(x.lg, x.gp))
end

"""
    leakage_simple(x)

A simple leakage model where the leakage is given by a single complex number for each feed.
"""
@inline function leakage_simple(x)
    dR = complex(x.d1re, x.d1im)
    dL = complex(x.d2re, x.d2im)
    return dR, dL
end


"""
    leakage_hier(x)

A hierarchical leakage model where the leakage for each feed is given by a mean, std dev, and random variable.
"""
@inline function leakage_hier(x)
    dR = complex(x.d1reμ + x.d1reσ * x.d1re, x.d1imμ + x.d1imσ * x.d1im)
    dL = complex(x.d2reμ + x.d2reσ * x.d2re, x.d2imμ + x.d2imσ * x.d2im)
    return dR, dL
end

@inline sandwich_withfrcal(g, d, r) = adjoint(r) * g * d * r
@inline sandwich(g, d, r) = g * d * r






