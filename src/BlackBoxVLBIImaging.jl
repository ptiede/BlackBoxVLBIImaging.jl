module BlackBoxVLBIImaging

using Reexport
@reexport using Comrade
using CairoMakie
# Imported unconditionally so ComradeReactantExt (the device posterior + ReactantNUTS
# sampler) is always available when the fitting strategy selects the Reactant sampler.
using Reactant
using VLBIImagePriors
using Distributions
using StaticArrays
using StatsFuns: logistic
using Enzyme
using LinearAlgebra
using Random
using Printf
using Serialization
using TOML
using FINUFFT
using AdvancedHMC
using Optimization
using OptimizationOptimisers: Adam, AdamW
using OptimizationOptimJL: LBFGS
import Optimisers
using LogDensityProblems
using FillArrays
using VLBIFiles
using DataFrames
using CSV
using StructArrays
using Accessors: @set
using ConstructionBase
using BenchmarkTools

# `VLBISkyModels` is re-exported by Comrade but its module binding is not; alias it so the
# many `VLBISkyModels.foo` references in the sky-model code resolve.
const VLBISkyModels = Comrade.VLBISkyModels

# --- sky models ------------------------------------------------------------------------
include("sky/polreps.jl")
include("sky/imagingmodel.jl")
include("sky/meanmodels.jl")
include("sky/skyprior.jl")

# --- instrument parameterizations (needed by the scheme registry) ----------------------
include("instrument/parameterizations.jl")

# --- data layer ------------------------------------------------------------------------
include("data/arraytable.jl")
include("data/dlist.jl")
include("data/corrpol.jl")
include("data/flagtable.jl")
include("data/dataloader.jl")

# --- generic instrument assembler ------------------------------------------------------
include("instrument/schemes.jl")
include("instrument/distspec.jl")
include("instrument/assemble.jl")

# --- TOML config layer -----------------------------------------------------------------
include("config/common.jl")
include("config/sky_config.jl")
include("config/instrument_config.jl")
include("config/data_config.jl")
include("config/fitting_config.jl")

# --- imaging pipeline ------------------------------------------------------------------
include("pipeline/output.jl")
include("pipeline/reactant_opt.jl")
include("pipeline/imager.jl")
include("pipeline/run.jl")

function __init__()
    # Parallelism comes from Julia threads (and the NUFFT), so pin BLAS to one thread.
    LinearAlgebra.BLAS.set_num_threads(1)
    return nothing
end

# sky models
export ImagingModel, skyprior, centroid
export Poincare, PolExp, TotalIntensity, Matern, MarkovRF, NonCenteredMRF
export GaussMean, DblRingMean, DblRingWBkgd, TBlobMean, JetGauss, GaussBkgdMean, MimgPlusBkg
# data layer
export read_flagtable, parse_flagtable, apply_flagtable, corr_polbasis
export read_dlist, read_array_table, build_data_uvfits, build_data_dlist
# instrument assembler
export assemble_instrument, parse_dist
# config layer
export build_sky_config, build_instrument_config, build_data_config, build_fitting_config
export FittingStrategy
# pipeline
export comrade_imager, best_image, reactant_opt, load_chain_and_post, saveimgs, image_from_toml

end
