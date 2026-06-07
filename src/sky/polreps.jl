# Polarization representations and random-field base markers used by `ImagingModel`.
#
# These small marker types select, via dispatch, how the stochastic image fluctuations
# are represented (`make_image` in imagingmodel.jl) and which priors are generated
# (`genimgprior` in skyprior.jl).

abstract type PolRep end
abstract type PolModel <: PolRep end

"""Full polarization via the Poincaré-sphere parameterization."""
struct Poincare <: PolModel end

"""Full polarization via the matrix-exponential (`PolExp`) parameterization."""
struct PolExp <: PolModel end

"""Total-intensity (Stokes I only) imaging."""
struct TotalIntensity <: PolRep end

"""Matérn-process random-field base (`order == 0`)."""
struct Matern end

"""Markov random field of explicit order `N` (`order < 0` selects `MarkovRF(abs(order))`)."""
struct MarkovRF{N} end
MarkovRF(n::Int) = MarkovRF{n}()

"""Non-centered Markov transform wrapping a base random field (`order > 1`)."""
struct NonCenteredMRF{B}
    base::B
end

"""A standardized stationary random field paired with its FFT plan."""
struct SRF{PS, P}
    ps::PS
    plan::P
end
