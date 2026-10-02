using BlackBoxVLBIImaging
using Test
using TOML
using Random
using LinearAlgebra
using Statistics: cov, mean
using Distributions: LogNormal, Normal, cdf, logpdf, quantile
import Enzyme

const EXDIR = normpath(joinpath(@__DIR__, "..", "examples"))
exconfig(f) = TOML.parsefile(joinpath(EXDIR, f))

@testset "BlackBoxVLBIImaging.jl" begin

    @testset "distribution spec parser" begin
        # parse_dist emits the Reactant-friendly VLBI* variants (also CPU-compatible); the
        # VLBI* constructors return AffineDistribution-wrapped Std* distributions.
        @test occursin("StdNormal", string(typeof(parse_dist(Dict("dist" => "Normal", "args" => [0.0, 0.4])))))
        @test occursin("StdExponential", string(typeof(parse_dist(Dict("dist" => "Exponential", "args" => [0.2])))))
        # VLBITruncated builds a ProbabilityTransports.Truncated since Comrade 0.11.33
        d = parse_dist(Dict("dist" => "Normal", "args" => [0.0, 1.0], "lower" => 0.0))
        @test occursin("Truncated", string(typeof(d)))
        @test parse_dist(Dict("dist" => "DiagonalVonMises", "args" => [0.0, 3.14159])) isa DiagonalVonMises
        @test parse_dist(Dict("dist" => "AngularProjectedNormal", "args" => [0.0, 0.0808])) isa
            BlackBoxVLBIImaging.PT.AngularProjectedNormal
        @test parse_dist(Dict("dist" => "WrappedNormal", "args" => [0.0, 2.44])) isa
            BlackBoxVLBIImaging.PT.WrappedNormal
        # a bounded LogNormal is the truncated log-normal
        dln = parse_dist(Dict("dist" => "LogNormal", "args" => [2.66, 0.764], "upper" => 48.0))
        ref = BlackBoxVLBIImaging.Distributions.truncated(LogNormal(2.66, 0.764); upper = 48.0)
        @test logpdf(dln, 20.0) - logpdf(ref, 20.0) ≈ logpdf(dln, 3.0) - logpdf(ref, 3.0) atol = 1.0e-12
        @test_throws ErrorException parse_dist(Dict("dist" => "Nonsense", "args" => [1.0]))
        @test_throws ErrorException parse_dist(Dict("args" => [1.0]))               # missing 'dist'
        @test_throws ErrorException parse_dist(Dict("dist" => "Normal", "arg" => [1.0]))  # typo'd key
    end

    @testset "scheme registry" begin
        # required params are probed from the @instrument definitions themselves
        for (k, ctor) in BlackBoxVLBIImaging.GAIN_SCHEMES
            @test !isempty(BlackBoxVLBIImaging.required_params(ctor))
        end
        @test BlackBoxVLBIImaging.required_params(BlackBoxVLBIImaging.LEAKAGE_SCHEMES["none"]) == ()
        @test BlackBoxVLBIImaging.required_params(BlackBoxVLBIImaging.GAIN_SCHEMES["gain"]) ==
            (:lg1, :gp1, :lgratμ, :lgratσ, :lgrat, :gprat, :gpratμ)
        @test BlackBoxVLBIImaging.required_params(BlackBoxVLBIImaging.LEAKAGE_SCHEMES["leakage_simple"]) ==
            (:d1re, :d1im, :d2re, :d2im)
        @test BlackBoxVLBIImaging.required_params(BlackBoxVLBIImaging.LEAKAGE_SCHEMES["leakage_disk"]) ==
            (:d1re, :d1im, :d2re, :d2im)
        @test BlackBoxVLBIImaging.unitdisk(0.0 + 0.0im) == 0
        @test abs(BlackBoxVLBIImaging.unitdisk(0.2 + 0.0im)) ≈ 0.2 / sqrt(1.04)
        @test all(z -> abs(BlackBoxVLBIImaging.unitdisk(z)) < 1, (3.0 + 4.0im, -1.0e3im, 30.0 + 0.0im))
        @test BlackBoxVLBIImaging.required_params(BlackBoxVLBIImaging.GAIN_SCHEMES["gain_offsetphase"]) ==
            (:lg1, :gp1μ, :gp1, :lgratμ, :lgrat, :gpratμ, :gprat)
    end

    @testset "instrument assembler" begin
        intm = build_instrument_config(exconfig("instrument_mixed.toml"))
        @test intm isa InstrumentModel
        # a missing required prior must error (closed-schema validation)
        cfg = exconfig("instrument_mixed.toml")
        delete!(cfg["priors"], "lg1")
        @test_throws ErrorException assemble_instrument(cfg)
        # unknown gain scheme must error
        cfg2 = exconfig("instrument_mixed.toml")
        cfg2["gain"]["scheme"] = "nope"
        @test_throws ErrorException assemble_instrument(cfg2)
    end

    @testset "iid first-stamp pin (init)" begin
        vm = Dict{String, Any}("dist" => "DiagonalVonMises", "args" => [0.0, 3.14159])
        sp = BlackBoxVLBIImaging._site_prior(
            Dict{String, Any}(
                "seg" => "integ", "dist" => vm,
                "init" => Dict{String, Any}("kind" => "fixed", "value" => 0.0),
            ), "test"
        )
        @test sp isa IIDSitePrior
        @test sp.init == FixedInit(0.0)
        # no init -> nothing (the default, back-compatible path)
        @test isnothing(
            BlackBoxVLBIImaging._site_prior(
                Dict{String, Any}("seg" => "integ", "dist" => vm), "test"
            ).init
        )
        # only a fixed pin is meaningful for iid
        @test_throws ErrorException BlackBoxVLBIImaging._site_prior(
            Dict{String, Any}(
                "seg" => "integ", "dist" => vm,
                "init" => Dict{String, Any}("kind" => "uniform"),
            ), "test"
        )
        @test_throws ArgumentError IIDSitePrior(
            ScanSeg(), parse_dist(vm); init = UniformInit()
        )
    end

    @testset "gauss-markov instrument priors" begin
        # parse_process: fixed hyperparameters vs fitted (distribution-spec) hyperparameters
        pf = parse_process(Dict("kind" => "OrnsteinUhlenbeck", "sigma" => 0.1, "tau" => 2.0))
        @test pf isa OrnsteinUhlenbeck
        @test isempty(Comrade.hyperprior(pf))            # all fixed -> no fitted hyperparams
        pfit = parse_process(Dict(
            "kind" => "ou",
            "sigma" => Dict("dist" => "Exponential", "args" => [0.1]),
            "tau" => Dict("dist" => "InverseGamma", "args" => [3.0, 6.0]),
        ))
        @test keys(Comrade.hyperprior(pfit)) == (:σ, :τ)
        @test_throws ErrorException parse_process(Dict("kind" => "Nope", "sigma" => 1.0, "tau" => 1.0))
        @test_throws ErrorException parse_process(Dict("kind" => "ou", "tau" => 1.0))       # missing sigma
        @test_throws ErrorException parse_process(Dict("kind" => "ou", "sigma" => 1.0, "tau" => 1.0, "mu" => Dict()))

        # WrappedBrownian: the circular process for phases, parameterized by the coherence
        # time tau (hours), fixed or fitted.
        wb = parse_process(Dict("kind" => "WrappedBrownian", "tau" => 1.0))
        @test wb isa WrappedBrownian
        @test isempty(Comrade.hyperprior(wb))
        wbfit = parse_process(Dict("kind" => "wb", "tau" => Dict("dist" => "InverseGamma", "args" => [1.0, 0.2])))
        @test keys(Comrade.hyperprior(wbfit)) == (:τ,)
        @test_throws ErrorException parse_process(Dict("kind" => "wb"))                     # missing tau
        @test_throws ErrorException parse_process(Dict("kind" => "wb", "tau" => 1.0, "sigma" => 1.0))  # OU key
        # `D` was the pre-tau spelling: rejected with a conversion hint, not silently accepted
        @test_throws ErrorException parse_process(Dict("kind" => "wb", "D" => 2.0))
        # BrownianMotion: the real-line random walk, parameterized by the diffusion coefficient D
        bm = parse_process(Dict("kind" => "BrownianMotion", "D" => 0.5))
        @test bm isa BrownianMotion
        @test isempty(Comrade.hyperprior(bm))
        bmfit = parse_process(Dict("kind" => "bm", "D" => Dict("dist" => "Exponential", "args" => [1.0])))
        @test keys(Comrade.hyperprior(bmfit)) == (:D,)
        @test_throws "missing 'D'" parse_process(Dict("kind" => "bm"))
        @test_throws ErrorException parse_process(Dict("kind" => "bm", "D" => 0.5, "tau" => 1.0))

        # WrappedOrnsteinUhlenbeck: circular AND stationary (the mean-reverting phase prior)
        wou = parse_process(Dict("kind" => "WrappedOrnsteinUhlenbeck", "sigma" => 0.3, "tau" => 12.0))
        @test wou isa WrappedOrnsteinUhlenbeck
        @test Comrade.is_wrapped(wou) && Comrade.isstationary(wou)
        @test isempty(Comrade.hyperprior(wou))
        woufit = parse_process(
            Dict(
                "kind" => "wou",
                "sigma" => Dict("dist" => "Exponential", "args" => [0.3]),
                "tau" => Dict("dist" => "InverseGamma", "args" => [2.0, 24.0]),
            )
        )
        @test keys(Comrade.hyperprior(woufit)) == (:σ, :τ)
        @test_throws ErrorException parse_process(Dict("kind" => "wou", "tau" => 1.0))   # no sigma
        @test_throws ErrorException parse_process(Dict("kind" => "wou", "sigma" => 0.3, "tau" => 1.0, "mu" => Dict()))

        # init default keys off STATIONARITY, not wrappedness: a stationary wrapped process
        # starts in its own WN(μ, σ²) marginal, and only the unbounded wrapped walk starts
        # uniform. Comrade accepts UniformInit for either, so a wrongly-defaulted init here
        # would be silent rather than an error.
        @test parse_init(nothing, wou, "test") isa StationaryInit
        @test parse_init(nothing, wb, "test") isa UniformInit
        @test parse_init(nothing, parse_process(Dict("kind" => "ou", "sigma" => 1.0, "tau" => 1.0)), "test") isa StationaryInit

        # initial priors: the default is each process's own stationary law (uniform on the
        # circle for a wrapped process), and the other kinds come from a string or a table
        @test parse_init(nothing, wb, "test") isa UniformInit
        @test parse_init(nothing, pf, "test") isa StationaryInit
        @test parse_init("uniform", wb, "test") isa UniformInit
        @test parse_init(Dict("kind" => "fixed", "value" => 0.0), wb, "test") == FixedInit(0.0)
        @test parse_init(Dict("kind" => "gaussian", "mu" => 1.0, "sigma" => 2.0), pf, "test") ==
            GaussianInit(1.0, 2.0)
        @test_throws ErrorException parse_init("bogus", wb, "test")
        @test_throws ErrorException parse_init("fixed", wb, "test")               # needs a value
        @test_throws ErrorException parse_init(Dict("value" => 0.0), wb, "test")  # missing kind
        @test_throws ErrorException parse_init(Dict("kind" => "fixed", "val" => 0.0), wb, "test")
        @test_throws ErrorException parse_init(0.0, wb, "test")
        # Comrade rejects the (init, process) combinations that do not exist
        @test_throws ArgumentError GaussMarkovSitePrior(IntegSeg(), wb; init = parse_init("stationary", wb, "test"))

        # a gaussmarkov default (fitted hyperparams) with an iid per-site override still builds
        cfg = exconfig("instrument_mixed.toml")
        cfg["priors"]["lg1"] = Dict{String, Any}(
            "kind" => "gaussmarkov", "seg" => "integ",
            "process" => Dict{String, Any}(
                "kind" => "OrnsteinUhlenbeck",
                "sigma" => Dict{String, Any}("dist" => "Exponential", "args" => [0.4]),
                "tau" => 2.0,
            ),
            "overrides" => Dict{String, Any}("SM" => Dict{String, Any}(
                "kind" => "iid", "seg" => "integ",
                "dist" => Dict{String, Any}("dist" => "Normal", "args" => [0.0, 0.5]),
            )),
        )
        cfg["priors"]["lgrat"] = Dict{String, Any}(
            "kind" => "gaussmarkov", "seg" => "scan", "centered" => true,
            "process" => Dict{String, Any}("kind" => "ou", "sigma" => 0.5, "tau" => 4.0),
        )
        @test build_instrument_config(cfg) isa InstrumentModel

        # refant kinds: Multi names several sites, for a parameter carrying more than one
        # gauge freedom
        @test BlackBoxVLBIImaging._parse_refant(nothing) === NoReference()
        @test BlackBoxVLBIImaging._parse_refant(Dict{String, Any}("kind" => "SEFD", "val" => 0.0)) isa
            SEFDReference
        @test BlackBoxVLBIImaging._parse_refant(
            Dict{String, Any}("kind" => "Single", "site" => "AA", "val" => 0.0)
        ) isa SingleReference
        rm = BlackBoxVLBIImaging._parse_refant(
            Dict{String, Any}("kind" => "Multi", "sites" => ["AA", "LM"], "val" => 0.0)
        )
        @test rm isa MultiReference
        @test rm.sites == [:AA, :LM]
        @test rm.value == 0.0
        @test_throws "requires a 'sites' list" BlackBoxVLBIImaging._parse_refant(
            Dict{String, Any}("kind" => "Multi")
        )
        @test_throws "'sites' to be a list" BlackBoxVLBIImaging._parse_refant(
            Dict{String, Any}("kind" => "Multi", "sites" => "AA")
        )
        @test_throws "None, SEFD, Single, Multi" BlackBoxVLBIImaging._parse_refant(
            Dict{String, Any}("kind" => "Manny", "sites" => ["AA"])
        )
        cfgm = exconfig("instrument_mixed.toml")
        cfgm["priors"]["gp1"]["refant"] =
            Dict{String, Any}("kind" => "Multi", "sites" => ["AA", "LM"], "val" => 0.0)
        @test build_instrument_config(cfgm) isa InstrumentModel

        # gauge = "phase" marks a parameter as one summand of the station phase, and the
        # top-level gaugefix says what to do when the summands leave it under-determined
        cfgg = exconfig("instrument_mixed.toml")
        cfgg["priors"]["gp1"]["gauge"] = "phase"
        intg = build_instrument_config(cfgg)
        @test intg isa InstrumentModel
        @test intg.prior.gp1.gauge === :phase
        @test intg.gaugefix === :error
        cfgg["gaugefix"] = "pin"
        @test build_instrument_config(cfgg).gaugefix === :pin
        cfgg["gaugefix"] = "ignore"
        @test_throws "Allowed: error, pin" build_instrument_config(cfgg)
        delete!(cfgg, "gaugefix")
        cfgg["priors"]["gp1"]["gauge"] = "amplitude"
        @test_throws "Allowed: none, phase" build_instrument_config(cfgg)

        # a WrappedBrownian phase chain (the correlated replacement for the iid phase = true
        # cumulative walk), with a per-site override that carries its own init
        cfgw = exconfig("instrument_mixed.toml")
        cfgw["priors"]["gp1"] = Dict{String, Any}(
            "kind" => "gaussmarkov", "seg" => "integ",
            "process" => Dict{String, Any}(
                "kind" => "WrappedBrownian",
                "tau" => Dict{String, Any}("dist" => "InverseGamma", "args" => [1.0, 0.2]),
            ),
            "refant" => Dict{String, Any}("kind" => "SEFD", "val" => 0.0),
        )
        cfgw["priors"]["gprat"] = Dict{String, Any}(
            "kind" => "gaussmarkov", "seg" => "integ",
            "process" => Dict{String, Any}("kind" => "WrappedBrownian", "tau" => 20.0),
            "init" => Dict{String, Any}("kind" => "fixed", "value" => 0.0),
            "overrides" => Dict{String, Any}("SM" => Dict{String, Any}(
                "kind" => "gaussmarkov", "seg" => "integ",
                "process" => Dict{String, Any}("kind" => "wb", "tau" => 2.0e4),
                "init" => Dict{String, Any}("kind" => "fixed", "value" => 0.0),
            )),
        )
        @test build_instrument_config(cfgw) isa InstrumentModel

        # a VonMisesProcess ratio-phase chain: the smooth-drift circular process takes
        # Comrade's NonCentered default when the TOML has no `centered`, and the shorthand
        # still selects Centered
        cfgv = exconfig("instrument_mixed.toml")
        cfgv["priors"]["gprat"] = Dict{String, Any}(
            "kind" => "gaussmarkov", "seg" => "scan",
            "process" => Dict{String, Any}(
                "kind" => "VonMisesProcess",
                "sigma" => Dict{String, Any}("dist" => "Exponential", "args" => [0.3]),
                "tau" => 24.0,
            ),
            "init" => Dict{String, Any}("kind" => "fixed", "value" => 0.0),
        )
        imv = build_instrument_config(cfgv)
        @test imv isa InstrumentModel
        let d = imv.prior.gprat.default_dist
            @test d.process isa BlackBoxVLBIImaging.Comrade.VonMisesProcess
            @test d.param isa BlackBoxVLBIImaging.Comrade.NonCentered
        end
        cfgv["priors"]["gprat"]["centered"] = true
        @test build_instrument_config(cfgv).prior.gprat.default_dist.param isa
            BlackBoxVLBIImaging.Comrade.Centered
        # ... while the shortest-arc process rejects the non-centered shorthand
        cfgv["priors"]["gprat"]["centered"] = false
        cfgv["priors"]["gprat"]["process"]["kind"] = "wou"
        @test_throws ArgumentError build_instrument_config(cfgv)

        # the shipped example TOMLs are valid (instrument_gaussmarkov.toml plus the
        # per-observation configs, whose phases are WrappedBrownian chains)
        @test build_instrument_config(exconfig("instrument_gaussmarkov.toml")) isa InstrumentModel
        for dir in filter(isdir, readdir(EXDIR; join = true))
            f = joinpath(dir, "instrument.toml")
            isfile(f) || continue
            @test build_instrument_config(TOML.parsefile(f)) isa InstrumentModel
        end

        # phase = true is incompatible with a gaussmarkov site prior
        cfgp = exconfig("instrument_mixed.toml")
        cfgp["priors"]["gp1"]["kind"] = "gaussmarkov"
        cfgp["priors"]["gp1"]["process"] = Dict{String, Any}("kind" => "ou", "sigma" => 0.1, "tau" => 2.0)
        @test_throws ErrorException assemble_instrument(cfgp)

        # unknown site-prior kind, and a gaussmarkov entry missing its process, both error
        cfgk = exconfig("instrument_mixed.toml")
        cfgk["priors"]["lg1"]["kind"] = "bogus"
        @test_throws ErrorException assemble_instrument(cfgk)
        cfgm = exconfig("instrument_mixed.toml")
        cfgm["priors"]["lg1"]["kind"] = "gaussmarkov"      # has dist but no process
        @test_throws ErrorException assemble_instrument(cfgm)
    end

    @testset "config key validation" begin
        # the original footgun: frcal placed below a [section] header nests under it
        cfg = exconfig("instrument_mixed.toml")
        cfg["gain"]["frcal"] = true
        @test_throws ErrorException assemble_instrument(cfg)
        # unknown keys error at every level of the instrument config
        mutations = (
            c -> c["fcral"] = true,                                        # top level
            c -> c["gain"]["sheme"] = "gain",                              # [gain]
            c -> c["priors"]["lg1"]["sg"] = "integ",                       # prior entry
            c -> c["priors"]["lg1"]["overrides"]["LM"]["phase"] = true,    # override entry
            c -> c["priors"]["gp1"]["refant"]["value"] = 0.0,              # refant spec
            c -> c["priors"]["lg1"]["dist"]["arg"] = [1.0],                # dist spec
        )
        for mutate! in mutations
            c = exconfig("instrument_mixed.toml")
            mutate!(c)
            @test_throws ErrorException assemble_instrument(c)
        end
        # priors unused by the chosen schemes warn but still build
        c = exconfig("instrument_mixed.toml")
        c["priors"]["lgoops"] = Dict{String, Any}(
            "seg" => "track", "dist" => Dict{String, Any}("dist" => "Normal", "args" => [0.0, 1.0])
        )
        intm = @test_logs (:warn, r"unused") match_mode = :any build_instrument_config(c)
        @test intm isa InstrumentModel
        # the sky, fitting, data, and flag configs reject unknown keys too
        s = exconfig("image.toml")
        s["grid"]["fox"] = 100.0
        @test_throws ErrorException build_sky_config(s)
        f = exconfig("fitting.toml")
        f["sampler"]["nsamples"] = 100
        @test_throws ErrorException build_fitting_config(f)
        d = exconfig("data.toml")
        d["path"] = Dict{String, Any}()
        @test_throws ErrorException build_data_config(d)   # errors before touching the file
        @test_throws ErrorException parse_flagtable(Dict{String, Any}("site" => ["AA"]))
    end

    @testset "polarization-basis corrections" begin
        flags = parse_flagtable(Dict{String, Any}("corr_polbasis" => Any[
            Dict("site" => "AA", "R" => "Y", "L" => "X"),
            Dict("site" => "LM", "X" => "L", "Y" => "R"),
            "HAY",
        ]))
        @test flags.corr_polbasis[1] == (; site = :AA, mapping = Pair{Any, Any}[RPol() => YPol(), LPol() => XPol()])
        @test flags.corr_polbasis[2] == (; site = :LM, mapping = Pair{Any, Any}[XPol() => LPol(), YPol() => RPol()])
        @test flags.corr_polbasis[3] == (; site = :HAY, mapping = Pair{Any, Any}[RPol() => YPol(), LPol() => XPol()])
        # `corpol` (the per-feed relabeler) is internal; only `corr_polbasis` is exported
        @test BlackBoxVLBIImaging.corpol(RPol(), flags.corr_polbasis[1].mapping) == YPol()
        @test_throws ErrorException BlackBoxVLBIImaging.corpol(XPol(), flags.corr_polbasis[1].mapping)
        @test_throws ErrorException parse_flagtable(Dict("corr_polbasis" => Any[
            Dict("site" => "AA", "R" => "Y", "L" => "linear"),
        ]))
    end

    @testset "fitting config" begin
        s = build_fitting_config(exconfig("fitting.toml"))
        @test s isa FittingStrategy
        @test s.noise_schedule == [0.05, 0.025, 0.0]
        @test s.opt_method == "Adam"
        @test s.nsample == 10_000
        @test !s.use_reactant
        @test isnothing(s.start)
        # `checkpoint` is accepted as an alias for `sample_checkpoint` (which wins when both
        # are set, so drop the explicit key first)
        cfg = exconfig("fitting.toml")
        delete!(cfg["run"], "sample_checkpoint")
        cfg["run"]["checkpoint"] = 7
        @test build_fitting_config(cfg).sample_checkpoint == 7
        # unknown optimizer must error
        cfg2 = exconfig("fitting.toml")
        cfg2["optimizer"]["method"] = "SGD"
        @test_throws ErrorException build_fitting_config(cfg2)

        @test isempty(s.fix_scales)
        cfg3 = exconfig("fitting.toml")
        cfg3["optimizer"]["fix_scales"] = ["σb", "σc", "σd"]
        @test build_fitting_config(cfg3).fix_scales == [:σb, :σc, :σd]
        cfg3["optimizer"]["fix_scales"] = "σb"
        @test_throws "must be a list of sky parameter names" build_fitting_config(cfg3)
        cfg3["optimizer"]["fix_scales"] = ["σb", "σb"]
        @test_throws "lists a name twice" build_fitting_config(cfg3)

        @test isempty(s.moves)
        cfg4 = exconfig("fitting.toml")
        cfg4["run"]["use_reactant"] = true
        cfg4["sampler"]["moves"] = ["flux_gain", "field_scale"]
        @test build_fitting_config(cfg4).moves == ["flux_gain", "field_scale"]
        cfg4["sampler"]["moves"] = ["flux_gain", "gain_phase"]
        @test_throws "unknown sampler.moves [\"gain_phase\"]" build_fitting_config(cfg4)
        cfg4["sampler"]["moves"] = ["flux_gain", "flux_gain"]
        @test_throws "lists a move twice" build_fitting_config(cfg4)
        cfg4["sampler"]["moves"] = "flux_gain"
        @test_throws "must be a list of move names" build_fitting_config(cfg4)
        cfg4["sampler"]["moves"] = ["rho_field", "mean_field", "phase_offset"]
        @test build_fitting_config(cfg4).moves_per_chunk == 1
        cfg4["sampler"]["moves_per_chunk"] = 10
        @test build_fitting_config(cfg4).moves_per_chunk == 10
        for bad in (0, -1, 1.5, "3")
            cfg4["sampler"]["moves_per_chunk"] = bad
            @test_throws "sampler.moves_per_chunk must be an integer ≥ 1" build_fitting_config(cfg4)
        end
        cfg4["sampler"]["moves_per_chunk"] = 2
        cfg4["sampler"]["moves"] = String[]
        @test_throws "moves_per_chunk is set but sampler.moves lists no moves" build_fitting_config(cfg4)
        delete!(cfg4["sampler"], "moves_per_chunk")
        cfg4["sampler"]["moves"] = ["field_scale"]
        cfg4["run"]["use_reactant"] = false
        @test_throws "needs the Reactant sampler" build_fitting_config(cfg4)

        @test s.latent_space == "flat"
        @test isnothing(BlackBoxVLBIImaging.latent_space(s))
        cfg5 = exconfig("fitting.toml")
        cfg5["run"]["latent_space"] = "stdnormal"
        s5 = build_fitting_config(cfg5)
        @test BlackBoxVLBIImaging.latent_space(s5) isa BlackBoxVLBIImaging.PT.StdNormal
        cfg5["run"]["use_reactant"] = true
        cfg5["run"]["latent_space"] = "cube"
        @test_throws "unknown run.latent_space 'cube'" build_fitting_config(cfg5)
        cfg5["run"]["latent_space"] = "stdnormal"
        cfg5["sampler"]["moves"] = ["flux_gain"]
        @test_throws "cannot run with run.latent_space" build_fitting_config(cfg5)
        delete!(cfg5["sampler"], "moves")
        cfg5["precondition"] = Dict{String, Any}("refit_schedule" => "stan")
        @test build_fitting_config(cfg5).precond_refit_schedule == "stan"
        @test BlackBoxVLBIImaging._metric_adaptor(build_fitting_config(cfg5)).carry === :none
        cfg5["precondition"]["refit_carry"] = "rescale"
        ad = BlackBoxVLBIImaging._metric_adaptor(build_fitting_config(cfg5))
        @test ad isa Comrade.FisherLowRank && ad.carry === :rescale
        cfg5["precondition"]["refit_carry"] = "keep"
        @test_throws "set precondition.seed_transport" build_fitting_config(cfg5)
        cfg5["precondition"]["seed_transport"] = "t.jls"
        @test BlackBoxVLBIImaging._metric_adaptor(build_fitting_config(cfg5)).carry === :keep
        cfg5["precondition"]["refit_carry"] = "yes"
        @test_throws "precondition.refit_carry must be" build_fitting_config(cfg5)
        delete!(cfg5, "precondition")
        cfg5["sampler"]["moves"] = ["phase_sheet"]
        @test build_fitting_config(cfg5).moves == ["phase_sheet"]
        cfg5["sampler"]["moves"] = ["phase_sheet", "flux_gain"]
        @test_throws "act on the flat latent space" build_fitting_config(cfg5)
        cfg5["sampler"]["moves"] = ["phase_sheet", "mean_field"]
        @test build_fitting_config(cfg5).moves == ["phase_sheet", "mean_field"]
        cm = BlackBoxVLBIImaging.ChainedMoves(((s, _...) -> s + 1, (s, _...) -> 2s))
        @test cm(3, nothing, nothing, nothing) == 8
        cfg5["sampler"]["moves"] = ["phase_sheet"]
        cfg5["run"]["latent_space"] = "flat"
        @test_throws "\"phase_sheet\" needs run.latent_space = \"stdnormal\"" build_fitting_config(cfg5)

        @test isnothing(s.dili)
        function dilicfg(d = Dict{String, Any}())
            cfg = exconfig("fitting.toml")
            delete!(cfg, "sampler")
            delete!(cfg["run"], "sample_checkpoint")
            cfg["run"]["use_reactant"] = true
            cfg["run"]["latent_space"] = "stdnormal"
            cfg["dili"] = d
            return cfg
        end
        dc = build_fitting_config(dilicfg()).dili
        @test all(f -> getfield(dc, f) == getfield(DILIConfig(), f), fieldnames(DILIConfig))
        dc = build_fitting_config(
            dilicfg(Dict{String, Any}("nsample" => 20, "thin" => 4, "refine_at" => [0.25, 0.5], "pin" => ["sky.σa"], "subspace" => "s.jls", "langevin_complement" => true))
        ).dili
        @test (dc.nsample, dc.thin, dc.refine_at, dc.pin, dc.subspace, dc.subspace_draws) == (20, 4, [0.25, 0.5], ["sky.σa"], "s.jls", nothing)
        @test dc.langevin_complement
        @test_throws "unknown key(s) [\"nsamples\"] in [dili]" build_fitting_config(dilicfg(Dict{String, Any}("nsamples" => 3)))
        bad = dilicfg()
        bad["run"]["use_reactant"] = false
        @test_throws "[dili] needs run.use_reactant = true" build_fitting_config(bad)
        bad = dilicfg()
        bad["run"]["latent_space"] = "flat"
        @test_throws "[dili] needs run.latent_space = \"stdnormal\"" build_fitting_config(bad)
        bad = dilicfg()
        bad["sampler"] = Dict{String, Any}("nsample" => 10)
        @test_throws "[sampler] configures NUTS" build_fitting_config(bad)
        bad = dilicfg()
        bad["run"]["sample_checkpoint"] = 10
        @test_throws "run.sample_checkpoint configures NUTS" build_fitting_config(bad)
        for (d, msg) in (
                ("nwarmup" => -1, "dili.nwarmup must be an integer ≥ 0"),
                ("thin" => true, "dili.thin must be an integer ≥ 1"),
                ("rank" => 2.5, "dili.rank must be an integer ≥ 1"),
                ("step_subspace" => 0, "dili.step_subspace must be positive"),
                ("step_complement" => -0.1, "dili.step_complement must be non-negative"),
                ("target_accept" => 1.0, "dili.target_accept must be in (0, 1)"),
                ("refine_at" => [0.5, 0.5], "dili.refine_at must be increasing fractions in (0, 1)"),
                ("refine_at" => [1.0], "dili.refine_at must be increasing fractions in (0, 1)"),
                ("pin" => "sky.σa", "dili.pin must be a list of strings"),
                ("pin" => ["sky.σa", "sky.σa"], "dili.pin lists a path twice"),
                ("langevin_complement" => 1, "dili.langevin_complement must be true or false"),
            )
            @test_throws msg build_fitting_config(dilicfg(Dict{String, Any}(d)))
        end
        @test_throws "dili.nsample = 3 is less than dili.thin = 4" build_fitting_config(dilicfg(Dict{String, Any}("nsample" => 3, "thin" => 4)))
        for (nw, ra) in ((0, [0.5]), (10, [0.01]), (10, [0.2, 0.24]), (10, [0.97]))
            @test_throws "they must be distinct and strictly between 0 and dili.nwarmup" build_fitting_config(dilicfg(Dict{String, Any}("nwarmup" => nw, "refine_at" => ra)))
        end
        @test_throws "are exclusive" build_fitting_config(dilicfg(Dict{String, Any}("subspace" => "a.jls", "subspace_draws" => "run")))
    end

    @testset "likelihood-informed subspace" begin
        BB = BlackBoxVLBIImaging
        rng = Random.Xoshiro(7)
        n = 60
        lowrank(λ) = (V = Matrix(qr(randn(rng, n, length(λ))).Q)[:, eachindex(λ)]; (V * Diagonal(λ) * V', V))
        λ0 = [40.0, 9.0, 3.0, 0.5, 0.05]
        A, V0 = lowrank(λ0)
        s = BB.likelihood_subspace(v -> A * v, n; rank = 8, threshold = 0.1, rng)
        @test s.λ ≈ λ0[1:4] rtol = 1.0e-10
        @test abs.(s.V' * V0[:, 1:4]) ≈ I atol = 1.0e-8
        @test s.V' * s.V ≈ I
        @test length(s) == 4

        @test_throws "raise rank" BB.likelihood_subspace(v -> A * v, n; rank = 2, threshold = 0.1, rng)
        @test_throws "max_basis_bytes" BB.likelihood_subspace(v -> A * v, n; rank = 8, max_basis_bytes = 100, rng)
        @test_throws "oversample must be at least 1" BB.likelihood_subspace(v -> A * v, n; rank = 8, oversample = 0, rng)
        @test_throws "free must be distinct" BB.likelihood_subspace(v -> A * v, n; rank = 8, free = [1, 1], rng)

        # restricted to free coordinates: the eigenpairs of the principal submatrix
        free = collect(1:2:n)
        sf = BB.likelihood_subspace(v -> A * v, n; rank = 8, free, threshold = 0.1, rng)
        Ef = eigen(Symmetric(A[free, free]); sortby = -)
        @test sf.λ ≈ filter(>=(0.1), Ef.values) rtol = 1.0e-10
        @test sf.free == free

        # averaged over draws: the eigenpairs of the mean operator
        B, _ = lowrank([20.0, 2.0, 0.3])
        avg = BB.averaged_operator((M, v) -> M * v, [A, B])
        sa = BB.likelihood_subspace(avg, n; rank = 10, threshold = 0.1, rng)
        Ea = eigen(Symmetric((A + B) / 2); sortby = -)
        @test sa.λ ≈ filter(>=(0.1), Ea.values) rtol = 1.0e-10
        @test_throws "at least one draw" BB.averaged_operator((M, v) -> M * v, [])

        path = joinpath(mktempdir(), "subspace.jls")
        BB.save_subspace(path, sa)
        sl = BB.load_subspace(path)
        @test sl.λ == sa.λ && sl.V == sa.V && sl.free == sa.free
        BB.Serialization.serialize(path, 1.0)
        @test_throws "not a LikelihoodSubspace" BB.load_subspace(path)
    end

    @testset "DILI step" begin
        BB = BlackBoxVLBIImaging
        rng = Random.Xoshiro(11)

        # linear-Gaussian likelihood with the exact subspace: every proposal is accepted
        n, m = 30, 5
        A, y = 3 .* randn(rng, m, n), randn(rng, m)
        fglin(u) = (r = A * u .- y; (sum(abs2, r) / 2, A' * r))
        sub = BB.likelihood_subspace(v -> A' * (A * v), n; rank = 8, threshold = 0.1, rng)
        @test length(sub) == m
        plin = BB.DILIProposal(sub)
        u = randn(rng, n)
        for lc in (false, true)
            pl = BB.DILIProposal(sub; langevin_complement = lc)
            logαs = map(1:100) do _
                δr, δc = exp.(randn(rng, 2))
                BB.dili_step(fglin, pl, u, fglin(u)..., δr, δc, randn(rng, m), randn(rng, n), -Inf)[4]
            end
            @test maximum(abs, logαs) < 1.0e-9
        end

        # nonlinear 4-D toy, one pinned coordinate: logα against the explicit density ratio
        free = [1, 2, 3]
        Φt(u) = (u[1] + u[2]^2 - 1)^2 + (u[3] * u[1] - 0.3 + u[4])^2 / 2
        function ∇Φt(u)
            d1, d2 = u[1] + u[2]^2 - 1, u[3] * u[1] - 0.3 + u[4]
            return [2 * d1 + d2 * u[3], 4 * d1 * u[2], d2 * u[1], d2]
        end
        fgt(u) = (Φt(u), ∇Φt(u))
        v, λ = normalize([1.0, 0.5, -0.3]), 2.5
        st = BB.LikelihoodSubspace([λ], reshape(v, 3, 1), free, 4, 0.1)
        pt = BB.DILIProposal(st)
        W = nullspace(reshape(v, 1, 3))
        logπ(u) = -Φt(u) - sum(abs2, u[free]) / 2
        function logq(u′, u, δr, δc, κ)
            (sr, βr), (sc, βc) = BB._cn(δr), BB._cn(δc)
            γ = 1 / (1 + λ)
            a, a′ = v' * u[free], v' * u′[free]
            g = ∇Φt(u)[free]
            gr = v' * g - λ * a
            w, w′ = W' * u[free], W' * u′[free]
            qr_ = δr > 0 ? -(a′ - sr * a + (1 - sr) * γ * gr)^2 / (2βr^2 * γ) : 0.0
            qc = δc > 0 ? -sum(abs2, w′ .- sc .* w .+ κ * (1 - sc) .* (W' * g)) / (2βc^2) : 0.0
            return qr_ + qc
        end
        for lc in (false, true), _ in 1:200
            p = BB.DILIProposal(st; langevin_complement = lc)
            u = randn(rng, 4)
            δr, δc = 3 .* rand(rng, 2)
            u′, _, _, logα = BB.dili_step(fgt, p, u, fgt(u)..., δr, δc, randn(rng, 1), randn(rng, 4), -Inf)
            @test u′[4] == u[4]
            @test logα ≈ logπ(u′) + logq(u, u′, δr, δc, lc) - logπ(u) - logq(u′, u, δr, δc, lc) atol = 1.0e-8 rtol = 1.0e-12
        end
        # one block frozen: δr = 0 leaves the subspace coordinate, δc = 0 the complement
        for lc in (false, true), (fr, fc) in ((0, 1), (1, 0)), _ in 1:100
            p = BB.DILIProposal(st; langevin_complement = lc)
            u = randn(rng, 4)
            δr, δc = fr .* 3 .* rand(rng), fc .* 3 .* rand(rng)
            u′, _, _, logα = BB.dili_step(fgt, p, u, fgt(u)..., δr, δc, randn(rng, 1), randn(rng, 4), -Inf)
            fr == 0 && @test v' * u′[free] ≈ v' * u[free] atol = 1.0e-12
            fc == 0 && @test W' * u′[free] ≈ W' * u[free] atol = 1.0e-12
            @test logα ≈ logπ(u′) + logq(u, u′, δr, δc, lc) - logπ(u) - logq(u′, u, δr, δc, lc) atol = 1.0e-8 rtol = 1.0e-12
        end
        # the propose/accept algebra against the direct projections
        let p = BB.DILIProposal(st; langevin_complement = true), u = randn(rng, 4)
            ξr, ξ = randn(rng, 1), randn(rng, 4)
            g = ∇Φt(u)
            u′, a′ = BB.dili_propose(p, u, g, p.V' * u, p.V' * g, 0.7, 0.4, ξr, ξ)
            @test a′ ≈ p.V' * u′
        end
        pinf = BB.dili_step(u -> (Inf, zero(u)), pt, u, fgt(u)..., 1.0, 1.0, [0.0], zeros(4), -Inf)
        @test pinf[4] == -Inf && pinf[1] == u

        # stationarity: chains started from exact draws (rejection from the prior) stay
        # at the target; KS at p ≈ 0.001. The same test rejects a step that omits the
        # subspace proposal densities, or the complement Langevin terms, from logα (KS ≈ 0.15).
        Φs(u) = 2 * (u[1] + u[2]^2 - 1)^2 + (u[3] * u[1] - 0.5)^2
        function ∇Φs(u)
            d1, d2 = u[1] + u[2]^2 - 1, u[3] * u[1] - 0.5
            return [4 * d1 + 2 * d2 * u[3], 8 * d1 * u[2], 2 * d2 * u[1], 0.0, 0.0]
        end
        fgs(u) = (Φs(u), ∇Φs(u))
        function exact_draws(rng, N)
            out = Vector{Vector{Float64}}()
            while length(out) < N
                u = randn(rng, 5)
                rand(rng) < exp(-Φs(u)) && push!(out, u)
            end
            return out
        end
        Vs = Matrix(qr(randn(Random.Xoshiro(1), 5, 2)).Q)[:, 1:2]
        ss = BB.LikelihoodSubspace([4.0, 1.0], Vs, collect(1:5), 5, 0.1)
        ps = BB.DILIProposal(ss)
        N = 4000
        function run_chains(p, δc; split = false)
            return map(exact_draws(Random.Xoshiro(200), N)) do u
                srng = Random.Xoshiro(hash(u))
                Φu, gu = fgs(u)
                for _ in 1:10
                    for (δr′, δc′) in (split ? ((3.0, 0.0), (0.0, δc)) : ((3.0, δc),))
                        u, Φu, gu, _ = BB.dili_step(fgs, p, u, Φu, gu, δr′, δc′, randn(srng, 2), randn(srng, 5), log(rand(srng)))
                    end
                end
                u
            end
        end
        ref = exact_draws(Random.Xoshiro(100), N)
        function ks2(x, y)
            xs, ys = sort(x), sort(y)
            return maximum(t -> abs(searchsortedlast(xs, t) / length(xs) - searchsortedlast(ys, t) / length(ys)), vcat(x, y))
        end
        chains = run_chains(ps, 1.0)
        chainsl = run_chains(BB.DILIProposal(ss; langevin_complement = true), 1.0)
        chainss = run_chains(BB.DILIProposal(ss; langevin_complement = true), 1.0; split = true)
        for f in (u -> u[1], u -> u[2], u -> u[3], Φs)
            @test ks2(f.(chains), f.(ref)) < 1.95 * sqrt(2 / N)
            @test ks2(f.(chainsl), f.(ref)) < 1.95 * sqrt(2 / N)
            @test ks2(f.(chainss), f.(ref)) < 1.95 * sqrt(2 / N)
        end

        # tuning toward target_accept during warmup, then frozen
        smp = BB.DILISampler(fgs, ps; target_accept = 0.6)
        res = BB.dili_sample(smp, randn(rng, 5), 5000, 20000; rng, record = u -> u[1])
        @test length(res.draws) == 20000 && length(res.logα) == 25000
        @test mean(res.accepted[5001:end]) ≈ 0.6 atol = 0.05
        δ = BB.step_sizes(smp)
        BB.dili_sample(smp, randn(rng, 5), 0, 10; rng)
        @test BB.step_sizes(smp) == δ

        # the linear-Gaussian posterior mean (A'A + I) \ A'y
        slin = BB.DILISampler(fglin, plin; δr = 2.0, δc = 2.0)
        rlin = BB.dili_sample(slin, zeros(n), 100, 4000; rng)
        @test all(rlin.accepted)
        @test mean(rlin.draws) ≈ (A' * A + I) \ (A' * y) atol = 0.1

        # a split sampler tunes each block to the target on its own
        ssp = BB.DILISampler(fgs, ps; target_accept = 0.6, split = true)
        st0 = BB.dili_start(ssp, randn(rng, 5))
        w = BB.dili_advance!((_...) -> nothing, ssp, st0, :warmup, 3000; rng)
        rs = BB.dili_advance!((_...) -> nothing, ssp, w.state, :sampling, 20000; rng)
        @test mean(rs.accepted) ≈ 0.6 atol = 0.05
        @test mean(rs.accepted_c) ≈ 0.6 atol = 0.05
        @test BB.step_sizes(ssp)[1] != BB.step_sizes(ssp)[2]
        @test length(rs.logα_c) == 20000 && isempty(BB.dili_advance!((_...) -> nothing, smp, w.state, :sampling, 3; rng).logα_c)
        @test_throws "cannot both be zero" BB.DILISampler(fgs, ps; δr = 0.0, δc = 0.0)
        @test_throws "must be non-negative" BB.DILISampler(fgs, ps; δr = -1.0)
        @test_throws "a split sampler needs δr > 0 and δc > 0" BB.DILISampler(fgs, ps; δc = 0.0, split = true)
        @test_throws "target_accept must be in (0, 1)" BB.DILISampler(fgs, ps; target_accept = 1.0)
        # NaN potentials at proposals are rejected and counted; a NaN logα leaves the tuner finite
        snan = BB.DILISampler(u -> (u[1] < 0 ? NaN : 0.0, zero(u)), ps; δr = 1.0e6, δc = 1.0e6)
        rnan = BB.dili_sample(snan, ones(5), 5, 5; rng)
        @test rnan.nnan.warmup + rnan.nnan.sampling == count(isnan, rnan.logα) > 0
        @test all(d -> d[1] >= 0, rnan.draws)
        @test all(isfinite, BB.step_sizes(snan))
        @test_throws "the potential at u0 is NaN" BB.dili_sample(snan, -ones(5), 1, 1; rng)
        @test_throws DimensionMismatch BB.dili_sample(smp, zeros(4), 1, 1)
    end

    @testset "sky config + grid snapping" begin
        skym, imgdata = build_sky_config(exconfig("image.toml"))
        @test skym isa SkyModel
        @test isnothing(imgdata)
        # the @sky model evaluates: draw from the prior and render the map
        x = rand(Random.default_rng(), Comrade.NamedDist(skym.prior))
        m = skym.f(x, skym.metadata)
        img = intensitymap(m, skym.grid)
        @test all(isfinite, stokes(img, :I))
        # the documented template builds too (order 2 → NonCenteredMRF, snapped grid)
        skym2, _ = build_sky_config(exconfig("image.toml"))
        @test skym2 isa SkyModel
        # NonCenteredMRF (order 2) wants nx such that nx+1 is a product of small primes
        nx, ny = BlackBoxVLBIImaging.snap_grid_size(NonCenteredMRF(GMRF), 2, 64, 64)
        @test (nx == 63) && (ny == 63)
        # GMRF order 1 snaps to a product of small primes
        @test BlackBoxVLBIImaging.snap_grid_size(GMRF, 1, 23, 23) == (24, 24)
    end

    @testset "image centering switch" begin
        # `metadata.center` is the Val the @sky body dispatches its re-centering on. A small
        # grid keeps the builds cheap; the grid plays no part in the centering choice.
        function centercfg(; kwargs...)
            cfg = exconfig("image.toml")
            cfg["grid"]["nx"] = 24
            cfg["grid"]["ny"] = 24
            cfg["model"]["order"] = 1
            for (k, v) in kwargs
                cfg["model"][String(k)] = v
            end
            return cfg
        end

        # no `center` key: the Bkgd mean model asks for re-centering
        skyd, imgdata = build_sky_config(centercfg())
        @test skyd.metadata.center === Val(true)
        @test isnothing(imgdata)

        # `center` states the choice outright
        @test build_sky_config(centercfg(center = false))[1].metadata.center === Val(false)
        @test build_sky_config(centercfg(center = true))[1].metadata.center === Val(true)

        # the centroid regularization turns re-centering off; `center = false` agrees with it
        skyc, cregdata = build_sky_config(centercfg(creg = true))
        @test skyc.metadata.center === Val(false)
        @test !isnothing(cregdata)
        skycf, cregdataf = build_sky_config(centercfg(creg = true, center = false))
        @test skycf.metadata.center === Val(false)
        @test !isnothing(cregdataf)

        # ...and cannot be combined with re-centering
        @test_throws "already pins the centroid" build_sky_config(
            centercfg(creg = true, center = true)
        )

        # `center_power` is the intensity weighting of the centroid either mechanism pins
        @test build_sky_config(centercfg(center = true))[1].metadata.center_power == 1
        @test build_sky_config(centercfg(center = true, center_power = 2))[1].metadata.center_power == 2
        @test build_sky_config(centercfg(creg = true, center_power = 2))[1].metadata.center_power == 2
        @test_throws "must be a real number >= 1" build_sky_config(
            centercfg(center = true, center_power = 0.5)
        )
        # nothing pins the position, so the power would be silently ignored
        @test_throws "nothing pins it" build_sky_config(
            centercfg(center = false, center_power = 2)
        )

        # A centered model puts the weighted centroid of its image at the origin, the plain
        # centroid elsewhere. Showing it needs an interpolating pulse — a delta-pulse render on
        # the model's own grid returns the raster itself, so the shift only reaches the
        # visibilities — and what is left of the I² centroid is the interpolation of a pixelized
        # raster, a fraction of a pixel.
        skyp, _ = build_sky_config(centercfg(center = true, center_power = 2, pulse = "bspline3"))
        xp = rand(Random.Xoshiro(42), Comrade.NamedDist(skyp.prior))
        imgp = intensitymap(skyp.f(xp, skyp.metadata), skyp.grid)
        px = rad2μas(pixelsizes(skyp.grid).X)
        c2 = rad2μas.(power_centroid(imgp, 2))
        c1 = rad2μas.(centroid(imgp))
        @test maximum(abs, c2) < 0.1 * px
        @test maximum(abs, c1) > maximum(abs, c2)
    end

    @testset "intensity-power-weighted centroid" begin
        # A compact blob at the origin plus a faint wide background offset along X: the center of
        # light rides along with the background, the I² centroid stays on the blob.
        gs = imagepixels(μas2rad(400.0), μas2rad(400.0), 64, 64)
        blob = modify(Gaussian(), Stretch(μas2rad(10.0)))
        bkgd(off) = modify(
            Gaussian(), Stretch(μas2rad(100.0)), Shift(μas2rad(off), 0.0), Renormalize(0.3)
        )
        img0 = intensitymap(blob + bkgd(0.0), gs)
        imgo = intensitymap(blob + bkgd(120.0), gs)

        # p = 1 goes through `centroid` itself, so it agrees to the last bit
        @test power_centroid(img0, 1) === centroid(img0)
        @test power_centroid(imgo, 1) === centroid(imgo)

        d1 = rad2μas.(centroid(imgo) .- centroid(img0))
        d2 = rad2μas.(power_centroid(imgo, 2) .- power_centroid(img0, 2))
        @test d1[1] > 10
        @test abs(d2[1]) < 0.05 * abs(d1[1])
        @test maximum(abs, rad2μas.(power_centroid(imgo, 2))) < 1.0

        # Differentiable in the raster: Enzyme reverse mode against a central difference along a
        # random direction. Single-pixel difference quotients are not a usable reference here —
        # derivatives in the faint wings are ~1e-13 and the quotient there is mostly round-off.
        f2(b) = first(power_centroid(IntensityMap(b, gs), 2))
        b0 = collect(baseimage(imgo))
        gr = Enzyme.gradient(Enzyme.Reverse, f2, b0)[1]
        @test all(isfinite, gr)
        v = randn(Random.Xoshiro(7), size(b0))
        h = 1.0e-6 * maximum(b0)
        @test sum(gr .* v) ≈ (f2(b0 .+ h .* v) - f2(b0 .- h .* v)) / (2h) rtol = 1.0e-4
    end

    @testset "Markov RF correlation-length prior" begin
        # A small order-3 Markov RF grid: three correlation lengths per field, nx = ny = 32
        # already FFT-friendly so the prior bounds are the configured pixel counts.
        function markovcfg(; kwargs...)
            cfg = exconfig("image.toml")
            cfg["grid"]["nx"] = 32
            cfg["grid"]["ny"] = 32
            cfg["model"]["order"] = -3
            for (k, v) in kwargs
                cfg["model"][String(k)] = v
            end
            return cfg
        end

        # default: uniform on [0.1, max(nx, ny)] pixels for every term
        skyu, _ = build_sky_config(markovcfg())
        @test length(skyu.prior.ρa) == 3
        for ρ in skyu.prior.ρa
            @test minimum(ρ) == 0.1
            @test maximum(ρ) == 32.0
        end

        # lognormal: term 1 about half the larger grid dimension with log-sd 1.0, terms
        # n ≥ 2 about the data beam in pixels with log-sd 0.7
        skyl, _ = build_sky_config(markovcfg(rho_prior = "lognormal"))
        beam_px = μas2rad(20.0) / step(skyl.grid.X)
        @test logpdf(skyl.prior.ρa[1], 7.0) ≈ logpdf(LogNormal(log(16.0), 1.0), 7.0)
        for n in 2:3
            @test logpdf(skyl.prior.ρa[n], 7.0) ≈ logpdf(LogNormal(log(beam_px), 0.7), 7.0)
        end
        @test skyl.prior.ρb == skyl.prior.ρc == skyl.prior.ρd == skyl.prior.ρa

        # the Stokes-I Markov constructor takes the same option (its field is `ρs`)
        skyi, _ = build_sky_config(markovcfg(polrep = "TotalIntensity", rho_prior = "lognormal"))
        @test logpdf(skyi.prior.ρs[1], 7.0) ≈ logpdf(LogNormal(log(16.0), 1.0), 7.0)

        # the sampler's unconstrained coordinate of a log-normal ρ is log ρ
        t = BlackBoxVLBIImaging.PT.transport_node(skyl.prior.ρa[1], BlackBoxVLBIImaging.PT.TVFlat())
        @test BlackBoxVLBIImaging.TV.transform(t, [log(7.0)]) ≈ 7.0
        @test only(BlackBoxVLBIImaging.TV.inverse(t, 7.0)) ≈ log(7.0)

        # the model still evaluates at a draw from the log-normal prior
        pr = Comrade.NamedDist(skyl.prior)
        x = rand(Random.default_rng(), pr)
        @test isfinite(logpdf(pr, x))
        @test all(isfinite, stokes(intensitymap(skyl.f(x, skyl.metadata), skyl.grid), :I))

        # an unknown family, and a family set where there is no ρ to put it on
        @test_throws "unknown rho_prior 'loggaussian'" build_sky_config(
            markovcfg(rho_prior = "loggaussian")
        )
        @test_throws "rho_prior sets the correlation-length prior" build_sky_config(
            markovcfg(rho_prior = "lognormal", order = 1)
        )
    end

    @testset "sky prior overrides" begin
        function overridecfg(overrides)
            cfg = exconfig("image.toml")
            cfg["grid"]["nx"] = 32
            cfg["grid"]["ny"] = 32
            cfg["model"]["order"] = -3
            cfg["model"]["rho_prior"] = "lognormal"
            cfg["overrides"] = overrides
            return cfg
        end

        # a scalar override (σa) pins a narrow truncated Normal around the given value
        skyσ, _ = build_sky_config(
            overridecfg(Dict("σa" => Dict("dist" => "Normal", "args" => [2.5, 1.0e-3], "lower" => 0.0)))
        )
        @test logpdf(skyσ.prior.σa, 2.5) ≈ logpdf(Normal(2.5, 1.0e-3), 2.5) atol = 1.0e-6
        @test logpdf(skyσ.prior.σa, 2.5) - logpdf(skyσ.prior.σa, 2.51) > 40  # narrow: steep falloff

        # an array override (ρa) replaces each Markov-order term with its own narrow LogNormal,
        # matching markov_rho_prior's own log ρ unconstrained coordinate
        ρvals = (7.0, 3.0, 1.5)
        skyρ, _ = build_sky_config(
            overridecfg(
                Dict(
                    "ρa" => [
                        Dict("dist" => "LogNormal", "args" => [log(v), 1.0e-3]) for v in ρvals
                    ],
                )
            )
        )
        @test length(skyρ.prior.ρa) == 3
        for (ρ, v) in zip(skyρ.prior.ρa, ρvals)
            @test logpdf(ρ, v) ≈ logpdf(LogNormal(log(v), 1.0e-3), v) atol = 1.0e-6
            @test logpdf(ρ, v) - logpdf(ρ, 1.01 * v) > 40  # narrow: steep falloff
        end
        # the untouched fields keep their default lognormal ρ prior
        @test skyρ.prior.ρb != skyρ.prior.ρa

        # an unknown override name fails fast
        @test_throws "does not match any prior entry" build_sky_config(
            overridecfg(Dict("σnope" => Dict("dist" => "Normal", "args" => [1.0, 1.0])))
        )
    end

    @testset "data product from polrep" begin
        # Stokes-I sky models fit complex visibilities; polarized models fit coherencies.
        @test BlackBoxVLBIImaging.data_product(TotalIntensity()) === Visibilities
        @test BlackBoxVLBIImaging.data_product(PolExp()) === Coherencies
        @test BlackBoxVLBIImaging.data_product(Poincare()) === Coherencies
        # sky_polrep is the single parse point shared by the sky and data paths.
        @test BlackBoxVLBIImaging.sky_polrep(
            Dict("model" => Dict("polrep" => "TotalIntensity"))
        ) isa TotalIntensity
        @test BlackBoxVLBIImaging.sky_polrep(Dict{String, Any}()) isa PolExp  # default
        # dlist is polarized coherencies by construction; TotalIntensity is unsupported and
        # must error (before/independent of touching a real file).
        @test_throws ErrorException build_data_dlist(
            "nope.dlist", "nope.array"; polrep = TotalIntensity()
        )
    end

    @testset "optional array file" begin
        # dlist builds its antenna table from the array file, so it is required: omitting it
        # (array = nothing) errors.
        @test_throws ErrorException build_data_dlist("nope.dlist", nothing)
        dcfg_dl = Dict{String, Any}(
            "paths" => Dict{String, Any}("file" => "nope.dlist", "path_mode" => "cwd"),
            "data" => Dict{String, Any}("format" => "dlist"),
        )
        @test_throws ErrorException build_data_config(dcfg_dl)
        # uvfits does not need an array (it is only used for feed-rotation overrides). A config
        # with no array must get PAST path validation and fail only when the data file is
        # loaded — proving 'array' is optional rather than required up front.
        dcfg_uv = Dict{String, Any}(
            "paths" => Dict{String, Any}("file" => "definitely_missing.uvfits", "path_mode" => "cwd"),
            "data" => Dict{String, Any}("format" => "uvfits"),
        )
        err = try
            build_data_config(dcfg_uv)
            nothing
        catch e
            e
        end
        @test err !== nothing
        @test !occursin("array", sprint(showerror, err))
    end

    @testset "uvfits convention switches" begin
        # `conjugate` / `ignore_feed_labels` are accepted [data] keys: a config using them must
        # get past key validation and fail only on the missing file.
        dcfg = Dict{String, Any}(
            "paths" => Dict{String, Any}("file" => "definitely_missing.uvfits", "path_mode" => "cwd"),
            "data" => Dict{String, Any}("format" => "uvfits", "conjugate" => true, "ignore_feed_labels" => true),
        )
        err = try
            build_data_config(dcfg)
            nothing
        catch e
            e
        end
        @test err !== nothing
        @test !occursin("unknown", lowercase(sprint(showerror, err)))
        @test_throws ErrorException build_data_config(Dict{String, Any}(
            "paths" => Dict{String, Any}("file" => "x.uvfits", "path_mode" => "cwd"),
            "data" => Dict{String, Any}("conjugat" => true),
        ))

        # End-to-end on a mixed-feed, opposite-convention file when it is available locally.
        domfile = "/mnt/ptiede/Research/EHT2026/data/DomData/calibration/2026-03-13/calibrated_M87_Jy.uvfits"
        if isfile(domfile)
            # the raw file marks ALMA linear, which the extractor refuses
            @test_throws ErrorException build_data_uvfits(domfile, nothing)
            raw = build_data_uvfits(domfile, nothing; ignore_feed_labels = true)
            conj_ = build_data_uvfits(domfile, nothing; ignore_feed_labels = true, conjugate = true)
            @test all(Comrade.measurement(conj_) .== adjoint.(Comrade.measurement(raw)))
            @test all(Comrade.noise(conj_) .== transpose.(Comrade.noise(raw)))
            @test all(pb -> pb == (CirBasis(), CirBasis()), datatable(arrayconfig(raw)).polbasis)
            cfg = Dict{String, Any}(
                "paths" => Dict{String, Any}("file" => domfile, "path_mode" => "cwd"),
                "data" => Dict{String, Any}("conjugate" => true, "ignore_feed_labels" => true),
                "flags" => Dict{String, Any}("corr_polbasis" => Any[Dict("site" => "AA", "R" => "X", "L" => "Y")]),
            )
            dcoh = build_data_config(cfg)
            dt = datatable(arrayconfig(dcoh))
            for r in dt
                for (s, pb) in zip(r.sites, r.polbasis)
                    @test pb == (s == :AA ? PolBasis{XPol, YPol}() : CirBasis())
                end
            end
            # Stokes I path: conjugation is a plain complex conjugate
            vi = build_data_uvfits(domfile, nothing; ignore_feed_labels = true, polrep = TotalIntensity())
            vc = build_data_uvfits(domfile, nothing; ignore_feed_labels = true, conjugate = true, polrep = TotalIntensity())
            @test Comrade.measurement(vc) == conj.(Comrade.measurement(vi))
        end
    end

    @testset "preconditioner config plumbing" begin
        # The preconditioner itself (transform, Fisher estimator, `fit_preconditioner`)
        # lives in Comrade and is tested there; this covers the TOML -> kwargs mapping.
        strat = build_fitting_config(
            Dict{String, Any}(
                "precondition" => Dict{String, Any}("pilot" => "/tmp/pilotrun", "rank" => 8),
            )
        )
        @test strat.precond_pilot == "/tmp/pilotrun"
        @test strat.precond_rank == 8
        @test strat.precond_nsamples == 2000
        @test isnothing(build_fitting_config(Dict{String, Any}()).precond_pilot)
        @test_throws ErrorException build_fitting_config(
            Dict{String, Any}("precondition" => Dict{String, Any}("minscale" => 2.0))
        )
    end

    # Integration smoke test — runs only if the workshop test data is present.
    datafile = "/home/ptiede/Harvard University Dropbox/Paul Tiede/CHWorkshop/data/3809/hops_3809_M87.apriori.uvfits"
    arrayfile = "/home/ptiede/Harvard University Dropbox/Paul Tiede/CHWorkshop/data/array.txt"
    @testset "integration: posterior + tiny optimize" begin
        if isfile(datafile) && isfile(arrayfile)
            dcfg = Dict{String, Any}(
                "paths" => Dict{String, Any}("file" => datafile, "array" => arrayfile, "path_mode" => "cwd"),
                "data" => Dict{String, Any}("format" => "uvfits", "ferr" => 0.01),
            )
            dcoh = build_data_config(dcfg)
            skym, imgdata = build_sky_config(exconfig("image.toml"))
            intm = build_instrument_config(exconfig("instrument_mixed.toml"))
            post = VLBIPosterior(skym, intm, dcoh; imgdata)
            x0 = prior_sample(Random.default_rng(), post)
            @test isfinite(logdensityof(post, x0))
            xopt, _ = comrade_opt(post, BlackBoxVLBIImaging.Adam(); initial_params = x0, maxiters = 5, g_tol = 0.1)
            @test isfinite(logdensityof(post, xopt))

            @testset "fix_scales: held optimization and field rescale" begin
                icfg = exconfig("image.toml")
                icfg["grid"]["nx"] = 32
                icfg["grid"]["ny"] = 32
                icfg["model"]["order"] = -3
                skyf, imgf = build_sky_config(icfg)
                postf = VLBIPosterior(skyf, intm, dcoh; imgdata = imgf)
                sq(x) = sqrt(mean(abs2, x))

                @test_throws "'σq' is not a scalar sky parameter" held_scale_values(postf, [:σq])
                @test_throws "'b' is not a scalar sky parameter" held_scale_values(postf, [:b])
                held = held_scale_values(postf, [:σb, :σc, :σd])
                @test held[:σd] ≈ 0.1 * quantile(Normal(), 0.75)
                @test held[:σb] ≈ held[:σc] ≈ 0.5 * quantile(Normal(), 0.75)

                # the CPU optimizer moves everything but the held scales
                xs = prior_sample(Random.Xoshiro(3), postf)
                xh, _ = BlackBoxVLBIImaging._comrade_opt_held(
                    postf, BlackBoxVLBIImaging.Adam(), held; initial_params = xs,
                    maxiters = 20, g_tol = 0.1
                )
                for (k, v) in held
                    @test xh.sky[k] ≈ v rtol = 1.0e-12
                end
                @test xh.sky.σa != xs.sky.σa
                @test xh.sky.b != xs.sky.b
                @test logdensityof(postf, xh) > logdensityof(postf, BlackBoxVLBIImaging._hold_scales(xs, held))

                # rescaling puts every field at unit rms and leaves the image where it was
                xr = rescale_fields(postf, xh)
                g = postf.skymodel.grid.imgdomain
                for k in (:a, :b, :c, :d)
                    σk = Symbol(:σ, k)
                    @test sq(xr.sky[k]) ≈ 1 rtol = 1.0e-12
                    @test xr.sky[σk] ≈ xh.sky[σk] * sq(xh.sky[k]) rtol = 1.0e-12
                end
                img(x) = baseimage(intensitymap(skymodel(postf, x), g))
                @test isapprox(img(xr), img(xh); rtol = 1.0e-12)
                # an all-zero field stays zero, its scale goes to the prior median
                xz = merge(xh, (sky = merge(xh.sky, (d = zero(xh.sky.d),)),))
                xzr = rescale_fields(postf, xz)
                @test all(iszero, xzr.sky.d)
                @test xzr.sky.σd ≈ held[:σd]
                # a model without a non-centered field
                @test_throws "no non-centered field" rescale_fields(post, x0)

                # end to end on the CPU path: two tempering stages, then the rescale
                strat = FittingStrategy(;
                    maxiters = 20, ntrials = 1, noise_schedule = [0.05, 0.0],
                    fix_scales = [:σb, :σc, :σd]
                )
                xt = mktempdir() do d
                    BlackBoxVLBIImaging._optimize_tempered(
                        joinpath(d, "t"), skyf, intm, (dcoh,), imgf, strat,
                        BlackBoxVLBIImaging.Adam(), Random.Xoshiro(4)
                    )
                end
                for k in (:a, :b, :c, :d)
                    @test sq(xt.sky[k]) ≈ 1 rtol = 1.0e-12
                end
                @test_throws "'σz' is not a scalar sky parameter" mktempdir() do d
                    BlackBoxVLBIImaging._optimize_tempered(
                        joinpath(d, "t"), skyf, intm, (dcoh,), imgf,
                        FittingStrategy(; maxiters = 2, ntrials = 1, fix_scales = [:σz]),
                        BlackBoxVLBIImaging.Adam(), Random.Xoshiro(4)
                    )
                end
            end

            @testset "symmetry moves" begin
                BB = BlackBoxVLBIImaging
                ProbProg = BB.Reactant.ProbProg
                icfg = exconfig("image.toml")
                icfg["grid"]["nx"] = 16
                icfg["grid"]["ny"] = 16
                icfg["model"]["order"] = -2
                icfg["model"]["rho_prior"] = "lognormal"
                icfg["mean"] = Dict{String, Any}("type" => "GaussBkgd", "fwhm_beams" => 2.0)
                skym, imgm = build_sky_config(icfg)
                # Gauss–Markov gain amplitudes, a per-site phase offset (gp1μ) with pinned
                # sites, and the Gauss–Markov ratio terms.
                instr = TOML.parse(
                    """
                    frcal = false
                    gaugefix = "pin"
                    [gain]
                    scheme = "gain_offsetphase"
                    [leakage]
                    scheme = "leakage_simple"
                    [priors.lg1]
                    kind = "gaussmarkov"
                    seg = "integ"
                    process = { kind = "ou", sigma = { dist = "Exponential", args = [0.2] }, tau = { dist = "InverseGamma", args = [2.0, 1.0] } }
                    [priors.lgrat]
                    kind = "gaussmarkov"
                    seg = "integ"
                    process = { kind = "ou", sigma = { dist = "Exponential", args = [0.05] }, tau = { dist = "InverseGamma", args = [2.0, 10.0] } }
                    [priors."lgratμ"]
                    seg = "track"
                    dist = { dist = "Normal", args = [0.0, 0.2] }
                    [priors."gp1μ"]
                    seg = "track"
                    dist = { dist = "DiagonalVonMises", args = [0.0, 3.141592653589793] }
                    refant = { kind = "SEFD", val = 0.0 }
                    gauge = "phase"
                    [priors.gp1]
                    seg = "integ"
                    dist = { dist = "DiagonalVonMises", args = [0.0, 3.141592653589793] }
                    init = { kind = "fixed", value = 0.0 }
                    refant = { kind = "SEFD", val = 0.0 }
                    gauge = "phase"
                    [priors.gprat]
                    kind = "gaussmarkov"
                    seg = "scan"
                    init = { kind = "fixed", value = 0.0 }
                    process = { kind = "WrappedOrnsteinUhlenbeck", sigma = { dist = "Exponential", args = [0.3], lower = 0.05 }, tau = { dist = "InverseGamma", args = [2.0, 24.0], upper = 48.0 } }
                    [priors."gpratμ"]
                    seg = "track"
                    dist = { dist = "DiagonalVonMises", args = [0.0, 3.141592653589793] }
                    [priors.d1re]
                    seg = "track"
                    dist = { dist = "Normal", args = [0.0, 0.2] }
                    [priors.d1im]
                    seg = "track"
                    dist = { dist = "Normal", args = [0.0, 0.2] }
                    [priors.d2re]
                    seg = "track"
                    dist = { dist = "Normal", args = [0.0, 0.2] }
                    [priors.d2im]
                    seg = "track"
                    dist = { dist = "Normal", args = [0.0, 0.2] }
                    """
                )
                intg = build_instrument_config(instr)
                postm = VLBIPosterior(skym, intg, dcoh; imgdata = imgm)
                tpm = asflat(postm)
                grid = postm.skymodel.grid
                θs = [prior_sample(Random.Xoshiro(k), postm) for k in 1:2]
                allnames = ["flux_gain", "field_scale", "rho_field", "mean_field", "phase_offset"]
                sm = SymmetryMoves(postm, allnames, θs[1]; rounds = 2)
                names = BB.move_name.(sm.moves)
                gpμ = θs[1].instrument[Symbol("gp1μ")]
                freesites = [s for (s, v) in zip(gpμ.sites, gpμ) if v != 0]
                @test collect(names) == [
                    "flux_gain",
                    "field_scale[a]", "field_scale[b]", "field_scale[c]", "field_scale[d]",
                    "rho_field[a,1]", "rho_field[a,2]", "rho_field[b,1]", "rho_field[b,2]",
                    "rho_field[c,1]", "rho_field[c,2]", "rho_field[d,1]", "rho_field[d,2]",
                    "mean_field[fwhm]", "mean_field[fb]",
                    ["phase_offset[$s]" for s in freesites]...,
                ]
                J = length(sm.moves)
                ustep(::BB.PhaseOffsetMove) = 0.7
                ustep(::BB.FieldScaleMove) = 0.2
                ustep(::BB.SymmetryMove) = 0.1
                stokesmap(θ) = baseimage(intensitymap(skymodel(postm, θ), grid))
                rel(a, b) = maximum(abs, a .- b) / maximum(abs, b)

                @testset "invariance, reversibility: $(BB.move_name(m)), draw $i" for (i, θ) in enumerate(θs), m in sm.moves
                    x = Comrade.inverse(tpm, θ)
                    u = ustep(m)
                    x′, ld = BB.propose(m, x, u, tpm, sm.ctx)
                    xb, ldb = BB.propose(m, x′, -u, tpm, sm.ctx)
                    @test xb ≈ x rtol = 1.0e-10
                    @test ldb ≈ -ld atol = 1.0e-9
                    θ′ = Comrade.transform(tpm, x′)
                    l0 = Comrade.loglikelihood(postm, θ)
                    l1 = Comrade.loglikelihood(postm, θ′)
                    if m isa BB.PhaseOffsetMove
                        a0 = atan(x[m.i], x[m.i + 1])
                        a1 = atan(x′[m.i], x′[m.i + 1])
                        @test rem2pi(a1 - a0 - u, RoundNearest) ≈ 0 atol = 1.0e-12
                        @test hypot(x′[m.i], x′[m.i + 1]) ≈ hypot(x[m.i], x[m.i + 1]) rtol = 1.0e-14
                        @test l1 != l0
                    else
                        @test l1 ≈ l0 rtol = 1.0e-10
                    end
                    if !(m isa BB.FluxGainMove || m isa BB.PhaseOffsetMove)
                        # the polarized image itself is unchanged, pixel by pixel
                        s0, s1 = stokesmap(θ), stokesmap(θ′)
                        for p in (:I, :Q, :U, :V)
                            @test rel(stokes(s1, p), stokes(s0, p)) < 1.0e-10
                        end
                    end
                end

                # The move is x ↦ M_u(x) and changes only the coordinates `S`, so
                # log|det ∂M/∂x| = log|det ∂M_S/∂x_S|; compare with central differences.
                @testset "log-determinant: $(BB.move_name(m))" for m in sm.moves[[1, 2, 6, 9, 14, 15, 16]]
                    x = Comrade.inverse(tpm, θs[2])
                    u = ustep(m)
                    M(y) = first(BB.propose(m, y, u, tpm, sm.ctx))
                    S = findall(M(x) .!= x)
                    @test !isempty(S)
                    Jac = zeros(length(S), length(S))
                    for (k, j) in pairs(S)
                        h = 1.0e-6 * max(1.0, abs(x[j]))
                        xp = copy(x)
                        xp[j] += h
                        xm = copy(x)
                        xm[j] -= h
                        Jac[:, k] = (M(xp)[S] .- M(xm)[S]) ./ (2h)
                    end
                    @test first(logabsdet(Jac)) ≈ last(BB.propose(m, x, u, tpm, sm.ctx)) atol = 1.0e-5
                end

                # Under a prior-only target the move kernel alone must leave the prior
                # invariant: start from exact prior draws, apply the kernel, and compare the
                # moved coordinates with fresh prior draws (two-sample Kolmogorov–Smirnov).
                # The same run with one move's Jacobian dropped must fail the comparison.
                @testset "prior stationarity of the move kernel" begin
                    ldprior(tp, z) = last(BB.PT.latent_pfwd_and_logdensity(tp.transform, vec(z)))
                    function ks(a, b)
                        grid_ = sort(vcat(a, b))
                        Fa = [count(<=(g), a) / length(a) for g in grid_]
                        Fb = [count(<=(g), b) / length(b) for g in grid_]
                        return maximum(abs, Fa .- Fb)
                    end
                    scale(::BB.FluxGainMove) = 0.5
                    scale(::BB.FieldScaleMove) = 0.1
                    scale(::BB.RhoFieldMove) = 0.5
                    scale(::BB.MeanFieldMove) = 0.05
                    scale(::BB.PhaseOffsetMove) = 1.5
                    stats(θ) = (
                        ftot = θ.sky.flux.ftot, σa = θ.sky.σa, σd = θ.sky.σd,
                        ρa1 = θ.sky.ρa[1], ρb2 = θ.sky.ρb[2],
                        fwhm = θ.sky.mean.fwhm, fb = θ.sky.mean.fb,
                        cosφ = cos(θ.instrument[Symbol("gp1μ")][findfirst(==(freesites[1]), gpμ.sites)]),
                    )
                    ndraw, R = 400, 3
                    rng = Random.Xoshiro(21)
                    run(moves) = map(1:ndraw) do _
                        z = Comrade.inverse(tpm, prior_sample(rng, postm))
                        steps = [scale(m) * randn(rng) for m in moves, _ in 1:R]
                        logu = log.(rand(rng, length(moves), R))
                        z′, logα = BB.run_moves(tpm, moves, sm.ctx, z, steps, logu; ldf = ldprior)
                        stats(Comrade.transform(tpm, z′)), logu .< logα
                    end
                    moved = run(sm.moves)
                    fresh = [stats(prior_sample(rng, postm)) for _ in 1:ndraw]
                    accept = mean(last.(moved))
                    @test all(>(0.02), accept)
                    Dcrit = 1.95 * sqrt(2 / ndraw)   # α ≈ 0.001 per statistic
                    for k in keys(first(fresh))
                        @test ks(getproperty.(first.(moved), k), getproperty.(fresh, k)) < Dcrit
                    end
                    # field_scale[a] with its Jacobian dropped drifts σa off its prior
                    nojac = (BB.FieldScaleMove(:a, sm.moves[2].coeffs, sm.moves[2].iscale, sm.moves[2].tscale),)
                    wrong = map(1:ndraw) do _
                        z = Comrade.inverse(tpm, prior_sample(rng, postm))
                        for _ in 1:R
                            u = scale(nojac[1]) * randn(rng)
                            z′, _ = BB.propose(nojac[1], z, u, tpm, sm.ctx)
                            if log(rand(rng)) < ldprior(tpm, z′) - ldprior(tpm, z)
                                z = z′
                            end
                        end
                        Comrade.transform(tpm, z).sky.σa
                    end
                    @test ks(wrong, getproperty.(fresh, :σa)) > Dcrit
                end

                # The compiled device step and the host step take the same proposals and
                # decisions on the same random numbers, through a preconditioner.
                @testset "device step matches the host step" begin
                    n = dimension(tpm)
                    prng = Random.Xoshiro(11)
                    V = Matrix(qr(randn(prng, n, 3)).Q)[:, 1:3]
                    pre = LowRankPreconditioner(randn(prng, n), exp.(0.3 .* randn(prng, n)), V, [3.0, 0.5, 1.5])
                    tph = BB.PT.transport_to(postm, pre)
                    postc = BB.ConstructionBase.setproperties(postm, (; admode = nothing))
                    rpost = Comrade.prepare_device(postc, Comrade.ComradeBase.ReactantEx())
                    devpre = Comrade._device_pre(pre)
                    tpd = BB.PT.transport_to(rpost, devpre)
                    z0 = Comrade._affine_inv(pre, Comrade.inverse(tpm, θs[1]))
                    steps = [0.3 * ustep(m) * randn(prng) for m in sm.moves, _ in 1:2]
                    logu = log.(rand(prng, J, 2))
                    zh, αh = BB.run_moves(tph, sm.moves, sm.ctx, z0, steps, logu)
                    zd, αd = BB._device_moves(sm, tpd, BB.Reactant.to_rarray(z0), steps, logu)
                    @test (logu .< αd) == (logu .< αh)
                    @test count(logu .< αh) > 0
                    # the device and host log densities agree to rounding of their magnitude
                    @test αd ≈ αh atol = 1.0e-11 * abs(logdensityof(tph, z0))
                    @test Array(zd) ≈ zh rtol = 1.0e-9
                    # same tpost, no recompile; a new one recompiles
                    c = sm.compiled[]
                    BB._device_moves(sm, tpd, zd, steps, logu)
                    @test sm.compiled[] === c

                    # a refit that overwrites the device preconditioner in place is seen by
                    # the compiled step without a recompile: it proposes and accepts in the
                    # new coordinates exactly as the host step does with the new transform
                    V2 = Matrix(qr(randn(prng, n, 3)).Q)[:, 1:3]
                    pre2 = LowRankPreconditioner(randn(prng, n), exp.(0.3 .* randn(prng, n)), V2, [0.2, 4.0, 1.3])
                    Comrade._update_device_pre!(devpre, pre2)
                    tph2 = BB.PT.transport_to(postm, pre2)
                    z2 = Comrade._affine_inv(pre2, Comrade.inverse(tpm, θs[2]))
                    zh2, αh2 = BB.run_moves(tph2, sm.moves, sm.ctx, z2, steps, logu)
                    zd2, αd2 = BB._device_moves(sm, tpd, BB.Reactant.to_rarray(z2), steps, logu)
                    @test sm.compiled[] === c
                    @test (logu .< αd2) == (logu .< αh2)
                    @test αd2 ≈ αh2 atol = 1.0e-11 * abs(logdensityof(tph2, z2))
                    @test Array(zd2) ≈ zh2 rtol = 1.0e-9

                    # the hook on the device: moves the position, records every proposal
                    state = ProbProg.MCMCState(BB.Reactant.to_rarray(z0), nothing, nothing, 0.1, nothing, nothing)
                    info = (; phase = :warmup, step = 10, total = 100, pre = BB.Comrade._transport_pre(tpd))
                    state = sm(state, tpd, info, Random.Xoshiro(12))
                    @test Array(state.position) != z0
                    @test all(t -> t.nwarmup == 2, sm.tuners)
                    @test occursin("flux_gain acc=", move_summary(sm, :warmup))
                    @test occursin("(0/0)", move_summary(sm, :sampling))
                end

                @testset "moves that do not apply are rejected up front" begin
                    @test_throws "unknown move \"gain_phase\"" SymmetryMoves(postm, ["gain_phase"], θs[1])
                    @test_throws "rounds must be at least 1" SymmetryMoves(postm, ["flux_gain"], θs[1]; rounds = 0)
                    fcfg = deepcopy(icfg)
                    fcfg["flux"]["ftot"] = [0.8]
                    skyfix, imgfix = build_sky_config(fcfg)
                    postfix = VLBIPosterior(skyfix, intg, dcoh; imgdata = imgfix)
                    @test_throws "needs a sampled total flux" SymmetryMoves(
                        postfix, ["flux_gain"], prior_sample(Random.Xoshiro(1), postfix)
                    )
                    # a first-stamp pin on lg1 cannot follow the common shift
                    pcfg = deepcopy(instr)
                    pcfg["priors"]["lg1"]["init"] = Dict("kind" => "fixed", "value" => 0.0)
                    postpin = VLBIPosterior(skym, build_instrument_config(pcfg), dcoh; imgdata = imgm)
                    @test_throws "move flux_gain changed the log-likelihood" SymmetryMoves(
                        postpin, ["flux_gain"], prior_sample(Random.Xoshiro(1), postpin)
                    )
                    # a GMRF sky has neither correlation lengths nor the PolExp Markov mean
                    gcfg = deepcopy(icfg)
                    gcfg["model"]["order"] = 1
                    delete!(gcfg["model"], "rho_prior")
                    skyg, imgg = build_sky_config(gcfg)
                    postg = VLBIPosterior(skyg, intg, dcoh; imgdata = imgg)
                    θg = prior_sample(Random.Xoshiro(1), postg)
                    @test_throws "needs a Markov RF sky model" SymmetryMoves(postg, ["rho_field"], θg)
                    @test_throws "needs the PolExp Markov RF sky model" SymmetryMoves(postg, ["mean_field"], θg)
                    postgm = VLBIPosterior(skym, build_instrument_config(exconfig("instrument_gaussmarkov.toml")), dcoh; imgdata = imgm)
                    @test_throws "needs a gain phase offset `gp1μ`" SymmetryMoves(
                        postgm, ["phase_offset"], prior_sample(Random.Xoshiro(1), postgm)
                    )
                end

                @testset "DILI kernels" begin
                    @test_throws "Cannot transport the circular distribution" BB.dili_posterior(postm)

                    # the same instrument with every prior exactly transportable to N(0, I):
                    # projected-normal phases and a real-line non-centered gprat chain
                    icfg_std = deepcopy(instr)
                    for p in ("gp1μ", "gp1", "gpratμ")
                        icfg_std["priors"][p]["dist"] = Dict("dist" => "AngularProjectedNormal", "args" => [0.0, 0.0808])
                    end
                    # log-normal τ: the InverseGamma quantile does not trace under Reactant
                    icfg_std["priors"]["lg1"]["process"]["tau"] = Dict("dist" => "LogNormal", "args" => [-0.518, 0.764])
                    icfg_std["priors"]["lgrat"]["process"]["tau"] = Dict("dist" => "LogNormal", "args" => [1.785, 0.764])
                    gprat = icfg_std["priors"]["gprat"]
                    gprat["centered"] = false
                    gprat["process"] = Dict(
                        "kind" => "ou",
                        "sigma" => Dict("dist" => "Exponential", "args" => [0.3], "lower" => 0.05),
                        "tau" => Dict("dist" => "LogNormal", "args" => [2.66, 0.764], "upper" => 48.0),
                    )
                    bm = Dict(
                        "kind" => "gaussmarkov", "seg" => "scan", "centered" => false,
                        "init" => Dict("kind" => "fixed", "value" => 0.0),
                        "process" => Dict("kind" => "bm", "D" => Dict("dist" => "Exponential", "args" => [1.0])),
                    )
                    gprat["overrides"] = Dict("SW" => bm, "AA" => deepcopy(bm))
                    posts = VLBIPosterior(skym, build_instrument_config(icfg_std), dcoh; imgdata = imgm)

                    tps = BB.dili_posterior(posts)
                    n = dimension(tps)
                    L = BB.latent_layout(tps)
                    @test last(L.instrument.d2im) == n
                    @test first(L.sky.a) == 1

                    θd = [prior_sample(Random.Xoshiro(k), posts) for k in 1:2]
                    us = [Comrade.inverse(tps, θ) for θ in θd]
                    ll(θ) = Comrade.loglikelihood(posts, θ)
                    @test ll(Comrade.transform(tps, us[1])) ≈ ll(θd[1]) rtol = 1.0e-10
                    # the prior is exactly N(0, I): log π + ½‖u‖² - log L is constant
                    c(u) = logdensityof(tps, u) + sum(abs2, u) / 2 - ll(Comrade.transform(tps, u))
                    cs = [c.(us); c(randn(Random.Xoshiro(3), n))]
                    @test all(≈(cs[1]; rtol = 1.0e-11), cs)

                    # a start point whose latent is not finite fails before sampling
                    @test BB.check_start(posts, BB.PT.StdNormal(), θd[1]) ≈ logdensityof(tps, us[1])
                    θ1 = θd[1]
                    θbad = BB.@set θ1.sky.mean.fwhm = 0.0
                    @test_throws "non-finite latent coordinates in sky.mean.fwhm" BB.check_start(posts, BB.PT.StdNormal(), θbad)

                    # real-line phase chains: the start re-wrap and the ±2π sheet move
                    xs1 = Comrade.transform(tps, us[1])
                    @test BB.phase_chain_terms(xs1) == (:gprat,)
                    xb = deepcopy(xs1)
                    Iaa = Dict(BB._site_points(xb.instrument.gprat.params))[:AA]
                    parent(xb.instrument.gprat.params)[Iaa[4:end]] .+= 2π
                    xw, nch = BB.unwrap_phase_chains(posts, xb)
                    @test nch >= 1
                    @test all(abs.(diff(parent(xw.instrument.gprat.params)[Iaa])) .<= π + 1.0e-12)
                    @test ll(xw) ≈ ll(xb) rtol = 1.0e-10
                    @test logdensityof(tps, Comrade.inverse(tps, xw)) >= logdensityof(tps, Comrade.inverse(tps, xb))
                    sm = BB.PhaseSheetMoves(posts; rounds = 3)
                    fx = sm.fixed[:gprat]
                    term, Ip = sm.points[findfirst(p -> length(p[2]) > 3 && !fx[p[2][3]], sm.points)]
                    span = BB._shift_span(fx, Ip, 3)
                    u1 = BB.sheet_proposal(sm, us[1], term, Ip, 3, 1)
                    d = parent(Comrade.transform(tps, u1).instrument.gprat.params) .- parent(xs1.instrument.gprat.params)
                    @test all(≈(2π; atol = 1.0e-9), d[span])
                    @test maximum(abs, d[setdiff(eachindex(d), span)]) < 1.0e-9
                    @test ll(Comrade.transform(tps, u1)) ≈ ll(xs1) rtol = 1.0e-10
                    @test BB.sheet_proposal(sm, u1, term, Ip, 3, -1) ≈ us[1] atol = 1.0e-10
                    # unit Jacobian: the latent shift is the same at another path with the same hyperparameters
                    rg, gn = sm.ranges[term], BB._term_node(sm.node, term)
                    u2 = copy(us[1])
                    xg = BB.PT.latent_pfwd(gn, u2[rg])
                    xg2 = BB.PT.latent_pfwd(gn, u2[rg] .+ 0.3 .* randn(Random.Xoshiro(9), length(rg)))
                    u2[rg] = BB.PT.latent_pback(gn, (; params = xg2.params, hyperparams = xg.hyperparams))
                    @test BB.sheet_proposal(sm, us[1], term, Ip, 3, 1) .- us[1] ≈
                        BB.sheet_proposal(sm, u2, term, Ip, 3, 1) .- u2 atol = 1.0e-9

                    # the mean-field move in the StdNormal space: the likelihood is unchanged, the
                    # move is reversed by the opposite step, and only the mean-model coordinate and
                    # the white coefficients of `a` change, by a shift that does not depend on `a`
                    smn = BB.SymmetryMoves(posts, ["mean_field"], θd[1]; space = BB.PT.StdNormal())
                    @test collect(BB.move_name.(smn.moves)) == ["mean_field[fwhm]", "mean_field[fb]"]
                    for m in smn.moves, u in us
                        u′, ld = BB.propose(m, u, 0.1, tps, smn.ctx)
                        @test ld == 0
                        @test first(BB.propose(m, u′, -0.1, tps, smn.ctx)) ≈ u rtol = 1.0e-10
                        @test ll(Comrade.transform(tps, u′)) ≈ ll(Comrade.transform(tps, u)) rtol = 1.0e-10
                        @test all(i -> i in m.coeffs || i == m.mean[m.imean], findall(u′ .!= u))
                        ua = copy(u)
                        ua[m.coeffs] .+= randn(Random.Xoshiro(4), length(m.coeffs))
                        @test first(BB.propose(m, ua, 0.1, tps, smn.ctx)) .- ua ≈ u′ .- u atol = 1.0e-10
                    end
                    @test_throws "in the StdNormal space only \"mean_field\" runs" BB.SymmetryMoves(
                        posts, ["flux_gain"], θd[1]; space = BB.PT.StdNormal()
                    )

                    rm = BB.ResidualMap(posts)
                    Φ = [BB.dili_potential(rm, tps, u) for u in us]
                    @test Φ[1] - Φ[2] ≈ ll(θd[2]) - ll(θd[1]) rtol = 1.0e-10

                    M = Tuple(copy.(rm.measurement))
                    N = Tuple(copy.(rm.noise))
                    M[2][3] = NaN
                    N[4][5] = Inf
                    @test length(BB.ResidualMap(M, N)) == length(rm) - 4
                    N[1][1] = 0.0
                    @test_throws "non-positive noise" BB.ResidualMap(M, N)

                    k = BB.DILIKernels(posts)
                    prng = Random.Xoshiro(21)
                    u, v, w = us[1], randn(prng, n), randn(prng, n)
                    h = 1.0e-5
                    fd(f, v) = (f(u .+ h .* v) .- f(u .- h .* v)) ./ (2h)
                    resid(u) = BB.dili_resid(rm, tps, u)
                    Jv, Jw = fd(resid, v), fd(resid, w)
                    r = resid(u)

                    @test BB.potential(k, u) ≈ Φ[1] rtol = 1.0e-9
                    Φd, g = BB.potential_gradient(k, u)
                    @test Φd ≈ Φ[1] rtol = 1.0e-9
                    @test dot(Array(g), v) ≈ dot(r, Jv) rtol = 1.0e-6
                    GNv, GNw = Array(BB.gauss_newton(k, u, v)), Array(BB.gauss_newton(k, u, w))
                    @test dot(w, GNv) ≈ dot(Jw, Jv) rtol = 1.0e-6
                    @test dot(w, GNv) ≈ dot(GNw, v) rtol = 1.0e-10

                    compiled = copy(k.compiled)
                    BB.gauss_newton(k, us[2], w)
                    @test k.compiled == compiled

                    # the subspace of the draw-averaged Gauss–Newton operator against its
                    # dense eigendecomposition
                    Id = Matrix{Float64}(I, n, n)
                    H = sum(u -> reduce(hcat, (Array(BB.gauss_newton(k, u, Id[:, j])) for j in 1:n)), us) ./ length(us)
                    Ed = eigen(Symmetric(H); sortby = -)
                    thr = (Ed.values[6] + Ed.values[7]) / 2
                    sub = BB.dili_subspace(k, us; rank = 10, threshold = thr, power = 4, rng = Random.Xoshiro(5))
                    @test length(sub) == 6
                    @test sub.λ ≈ Ed.values[1:6] rtol = 1.0e-6

                    # the compiled DILI step against the host step on the device potential
                    fgk(u) = ((Φ, g) = BB.potential_gradient(k, u); (Φ, Array(g)))
                    ds = BB.DILISampler(k, sub)
                    hp = BB.DILIProposal(sub)
                    Φ0, g0 = fgk(u)
                    ξr, ξ = randn(prng, length(sub)), randn(prng, n)
                    hu, hΦ, _, hα = BB.dili_step(fgk, hp, u, Φ0, g0, 1.0e-7, 1.0e-3, ξr, ξ, -Inf)
                    du, dΦ, dg, dα = ds.step(u, Φ0, g0, 1.0e-7, 1.0e-3, ξr, ξ, -Inf)
                    @test Array(du) ≈ hu rtol = 1.0e-10
                    @test dΦ ≈ hΦ rtol = 1.0e-9
                    @test dα ≈ hα atol = 1.0e-6 * abs(Φ0)
                    compiled = copy(k.compiled)
                    ds.step(Array(du), dΦ, Array(dg), 0.1, 0.5, ξr, ξ, 0.0)
                    @test k.compiled == compiled
                    res = BB.dili_sample(ds, u, 2, 3; rng = prng)
                    @test length(res.draws) == 3 && length(res.draws[1]) == n
                    @test_throws DimensionMismatch BB.DILISampler(k, BB.LikelihoodSubspace(sub.λ, sub.V, sub.free, n + 1, sub.threshold))
                    # the Langevin complement: its own compiled pair, same agreement with the host step
                    dsl = BB.DILISampler(k, sub; langevin_complement = true)
                    hpl = BB.DILIProposal(sub; langevin_complement = true)
                    hul, hΦl, _, hαl = BB.dili_step(fgk, hpl, u, Φ0, g0, 1.0e-7, 1.0e-7, ξr, ξ, -Inf)
                    dul, dΦl, _, dαl = dsl.step(u, Φ0, g0, 1.0e-7, 1.0e-7, ξr, ξ, -Inf)
                    @test Array(dul) ≈ hul rtol = 1.0e-10
                    @test dαl ≈ hαl atol = 1.0e-6 * abs(Φ0)
                    @test !(Array(dul) ≈ first(BB.dili_propose(hp, u, g0, hp.V' * u, hp.V' * g0, 1.0e-7, 1.0e-7, ξr, ξ)))
                    # the device step caches (Vᵀu, Vᵀ∇Φ) between steps: it follows the host step
                    # along a trajectory of accepted and rejected moves, split or not
                    for split in (false, true)
                        trng = Random.Xoshiro(5)
                        hu, hΦ, hg = u, Φ0, g0
                        tu, tΦ, tg = u, Φ0, g0
                        for _ in 1:10, (δr, δc) in (split ? ((1.0e-7, 0.0), (0.0, 1.0e-7)) : ((1.0e-7, 1.0e-7),))
                            ξr, ξ, logu = randn(trng, length(sub)), randn(trng, n), log(rand(trng))
                            hu, hΦ, hg, hα = BB.dili_step(fgk, hp, hu, hΦ, hg, δr, δc, ξr, ξ, logu)
                            tu, tΦ, tg, tα = ds.step(tu, tΦ, tg, δr, δc, ξr, ξ, logu)
                            @test (logu < hα) == (logu < tα)
                            @test Array(tu) ≈ hu rtol = 1.0e-10
                        end
                    end

                    # the sampling run: DiskStore output, pinned coordinates, subspace rebuild
                    mktempdir() do dir
                        out = joinpath(dir, "dili")
                        cfg = DILIConfig(;
                            nwarmup = 4, nsample = 6, thin = 2, stride = 2, rank = 20, power = 1,
                            threshold = thr, refine_at = [0.5], subspace_ndraws = 2, pin = ["sky.σa"],
                            step_subspace = 1.0e-6, step_complement = 1.0e-4,
                        )
                        pins = BB.pinned_coordinates(tps, cfg.pin)
                        @test pins == collect(L.sky.σa)
                        @test_throws "is not a parameter path: no \"nope\"" BB.pinned_coordinates(tps, ["sky.nope"])
                        res = BB.sample_dili(out, k, u, cfg; rng = Random.Xoshiro(4))
                        @test (res.nsamples, res.nfiles, res.stride) == (3, 2, 2)
                        ch = load_samples(out)
                        xs = Comrade.postsamples(ch)
                        @test length(xs) == 3
                        @test Comrade.samplerstats(ch).step == [2, 4, 6]
                        @test all(x -> x.sky.σa == Comrade.transform(tps, u).sky.σa, xs)
                        tr = BB.deserialize(joinpath(out, "dili_trace.jls"))
                        @test tr.phase == [fill(:warmup, 4); fill(:sampling, 6)]
                        @test length(tr.logα) == length(tr.scale) == length(tr.seconds) == 10
                        @test tr.refine_steps == [2]
                        @test isfile(joinpath(out, "subspace_0.jls")) && isfile(joinpath(out, "subspace_1.jls"))
                        @test BB.load_subspace(joinpath(out, "subspace.jls")).free == setdiff(1:n, pins)
                        @test_throws "already holds a chain" BB.sample_dili(out, k, u, cfg)
                        # a saved subspace must match the sampled coordinates
                        cfg2 = DILIConfig(; nwarmup = 0, nsample = 1, subspace = joinpath(out, "subspace.jls"))
                        @test_throws "check dili.pin" BB.sample_dili(joinpath(dir, "dili2"), k, u, cfg2)
                    end
                end
            end
        else
            @test_skip "test data not found at $datafile"
        end
    end
end
