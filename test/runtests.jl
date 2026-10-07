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

    @testset "a draw with non-finite parameters is an error" begin
        good = (sky = (σa = 1.0, ρa = (2.0, 3.0), a = ones(2, 2)), instrument = (lg = [0.1, 0.2],))
        @test BlackBoxVLBIImaging._check_finite_draw(good, "warmup step 10") === good
        bad = (sky = (σa = 1.0, σd = Inf, ρa = (2.0, NaN), a = [1.0 -Inf; 0.0 1.0]), instrument = (lg = [0.1, 0.2],))
        @test BlackBoxVLBIImaging._nonfinite_paths(bad) == [".sky.σd = Inf", ".sky.ρa[2] = NaN", ".sky.a (1 of 4 entries, e.g. -Inf)"]
        @test_throws "the warmup step 140 draw has non-finite parameters: .sky.σd = Inf" BlackBoxVLBIImaging._check_finite_draw(bad, "warmup step 140")
    end

    @testset "NUTS depth estimate from the kernel time" begin
        dn = BlackBoxVLBIImaging._depth_note
        # 8 s over 10 draws of 1 ms gradients
        @test dn(8.0, 10, 1.0e-3) == " ~lf/step=800 (depth≈9.6)"
        @test dn(8.0, 10, nothing) == ""
        @test dn(8.0, 10, 1.0e-3; fresh = true) == ""
        @test dn(8.0, 10, 1.0e-3; maxdepth = 10) == " ~lf/step=800 (depth≈9.6)"
        @test dn(8.0, 10, 0.78e-3; maxdepth = 10) == " ~lf/step=1026 (depth≈10.0), at the depth cap 10"
    end

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
        cfg4["sampler"]["moves"] = [
            Dict{String, Any}("kind" => "flux_gain"),
            Dict{String, Any}("kind" => "mean_field", "params" => ["fwhm"], "rounds" => 10, "target_accept" => 0.3, "initial_scale" => 0.02),
            Dict{String, Any}("kind" => "mean_field", "params" => ["fb"]),
        ]
        ms4 = build_fitting_config(cfg4).moves
        @test ms4[1] == MoveSpec(; kind = "flux_gain")
        @test (ms4[2].params, ms4[2].rounds, ms4[2].target_accept, ms4[2].initial_scale) == (["fwhm"], 10, 0.3, 0.02)
        @test (ms4[3].rounds, ms4[3].target_accept, ms4[3].initial_scale) == (1, 0.45, nothing)
        # the tables parse from TOML as written in the docs
        toml = TOML.parse("""
            [sampler]
            [[sampler.moves]]
            kind = "field_scale"
            params = ["a"]
            [[sampler.moves]]
            kind = "rho_field"
            [run]
            use_reactant = true
            """)
        @test [m.kind for m in build_fitting_config(toml).moves] == ["field_scale", "rho_field"]
        for (moves, msg) in (
                (["flux_gain", "field_scale"], "sampler.moves must be [[sampler.moves]] tables"),
                ("flux_gain", "sampler.moves must be [[sampler.moves]] tables"),
                ([Dict{String, Any}("kind" => "phase_offset")], "unknown move kind \"phase_offset\""),
                ([Dict{String, Any}("rounds" => 2)], "table 1 needs a kind"),
                ([Dict{String, Any}("kind" => "flux_gain", "every" => 2)], "unknown key(s) [\"every\"] in [[sampler.moves]] table 1"),
                ([Dict{String, Any}("kind" => "flux_gain", "params" => ["ftot"])], "flux_gain takes no params"),
                ([Dict{String, Any}("kind" => "rho_field", "params" => "a")], "params must be a non-empty list"),
                ([Dict{String, Any}("kind" => "rho_field", "rounds" => 0)], "rounds must be an integer ≥ 1"),
                ([Dict{String, Any}("kind" => "rho_field", "rounds" => 1.5)], "rounds must be an integer ≥ 1"),
                ([Dict{String, Any}("kind" => "rho_field", "target_accept" => 1.0)], "target_accept must lie in (0, 1)"),
                ([Dict{String, Any}("kind" => "rho_field", "initial_scale" => -1)], "initial_scale must be positive"),
                ([Dict{String, Any}("kind" => "mean_field"), Dict{String, Any}("kind" => "mean_field", "params" => ["fb"])], "kind \"mean_field\" more than once"),
                ([Dict{String, Any}("kind" => "mean_field", "params" => ["fb"]), Dict{String, Any}("kind" => "mean_field", "params" => ["fb"])], "kind \"mean_field\" more than once"),
                ([Dict{String, Any}("kind" => "phase_sheet")], "phase_sheet needs run.latent_space = \"stdnormal\""),
                ([Dict{String, Any}("kind" => "chain_hyper")], "chain_hyper needs run.latent_space = \"stdnormal\""),
            )
            c = deepcopy(cfg4)
            c["sampler"]["moves"] = moves
            @test_throws msg build_fitting_config(c)
        end
        cfg4["sampler"]["moves_per_chunk"] = 10
        @test_throws "set `rounds` in each [[sampler.moves]] table" build_fitting_config(cfg4)
        delete!(cfg4["sampler"], "moves_per_chunk")
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
        # Gauss–Newton refits
        cfg5["precondition"] = Dict{String, Any}(
            "refit_schedule" => "stan", "refit_kind" => "gauss_newton", "rank" => 600, "threshold" => 50.0
        )
        sg = build_fitting_config(cfg5)
        @test (sg.precond_refit_kind, sg.precond_rank, sg.precond_threshold, sg.precond_oversample) == ("gauss_newton", 600, 50.0, 10)
        H(x, W) = W
        ga = BlackBoxVLBIImaging._metric_adaptor(sg, H)
        @test ga isa Comrade.GaussNewtonLowRank && ga.curvature === H && ga.probes_per_draw == 610 && ga.schedule === :stan
        cfg5["precondition"]["probes_per_draw"] = 70
        @test BlackBoxVLBIImaging._metric_adaptor(build_fitting_config(cfg5), H).probes_per_draw == 70
        @test build_fitting_config(cfg5).precond_band_limit == 3.0
        @test BlackBoxVLBIImaging._metric_adaptor(build_fitting_config(cfg5), H; rows = [1, 4, 9]).rows == [1, 4, 9]
        cfg5["precondition"]["band_limit"] = Inf
        @test build_fitting_config(cfg5).precond_band_limit == Inf
        cfg5["precondition"]["band_limit"] = 0.0
        @test_throws "band_limit must be a positive multiple" build_fitting_config(cfg5)
        delete!(cfg5["precondition"], "band_limit")
        @test_throws "need a curvature function" BlackBoxVLBIImaging._metric_adaptor(sg)
        for (edit, msg) in (
                (c -> c["precondition"]["refit_kind"] = "hessian", "precondition.refit_kind must be"),
                (c -> delete!(c["precondition"], "rank"), "needs precondition.rank"),
                (c -> delete!(c["precondition"], "refit_schedule"), "needs refit_schedule or refit_at"),
                (c -> c["precondition"]["refit_carry"] = "rescale", "precondition.refit_carry applies to Fisher refits"),
                (c -> c["precondition"]["pilot"] = "run", "precondition.pilot fits a Fisher transform"),
                (c -> c["run"]["latent_space"] = "flat", "needs run.latent_space = \"stdnormal\""),
                (c -> c["run"]["use_reactant"] = false, "needs run.use_reactant = true"),
                (c -> delete!(c["precondition"], "refit_kind"), "precondition.threshold, probes_per_draw apply to refit_kind = \"gauss_newton\" only"),
            )
            c = deepcopy(cfg5)
            edit(c)
            @test_throws msg build_fitting_config(c)
        end
        delete!(cfg5, "precondition")
        cfg5["sampler"]["moves"] = [Dict{String, Any}("kind" => k) for k in ("phase_sheet", "flux_gain", "mean_field")]
        @test [m.kind for m in build_fitting_config(cfg5).moves] == ["phase_sheet", "flux_gain", "mean_field"]
        cfg5["sampler"]["moves"] = [Dict{String, Any}("kind" => "phase_sheet", "initial_scale" => 0.1)]
        @test_throws "phase_sheet takes discrete ±2π steps" build_fitting_config(cfg5)

        cfg6 = exconfig("fitting.toml")
        cfg6["dili"] = Dict{String, Any}()
        @test_throws "unknown key(s) [\"dili\"]" build_fitting_config(cfg6)
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
        # n ≥ 2 about the data beam in pixels with log-sd 0.7, each truncated to [1, 32] pixels
        skyl, _ = build_sky_config(markovcfg(rho_prior = "lognormal"))
        beam_px = μas2rad(20.0) / step(skyl.grid.X)
        @test logpdf(skyl.prior.ρa[1], 7.0) ≈ logpdf(BlackBoxVLBIImaging.Distributions.truncated(LogNormal(log(16.0), 1.0); lower = 1.0, upper = 32.0), 7.0)
        for n in 2:3
            @test logpdf(skyl.prior.ρa[n], 7.0) ≈ logpdf(BlackBoxVLBIImaging.Distributions.truncated(LogNormal(log(beam_px), 0.7); lower = 1.0, upper = 32.0), 7.0)
            @test logpdf(skyl.prior.ρa[n], 0.9) == -Inf
            @test logpdf(skyl.prior.ρa[n], 33.0) == -Inf
        end
        @test skyl.prior.ρb == skyl.prior.ρc == skyl.prior.ρd == skyl.prior.ρa

        # the Stokes-I Markov constructor takes the same option (its field is `ρs`)
        skyi, _ = build_sky_config(markovcfg(polrep = "TotalIntensity", rho_prior = "lognormal"))
        @test logpdf(skyi.prior.ρs[1], 7.0) ≈ logpdf(BlackBoxVLBIImaging.Distributions.truncated(LogNormal(log(16.0), 1.0); lower = 1.0, upper = 32.0), 7.0)

        # the flat and StdNormal transports map the whole real line onto [1, 32]
        for sp in (BlackBoxVLBIImaging.PT.TVFlat(), BlackBoxVLBIImaging.PT.StdNormal())
            t = BlackBoxVLBIImaging.PT.transport_node(skyl.prior.ρa[2], sp)
            @test BlackBoxVLBIImaging.PT.latent_pfwd(t, [-30.0]) >= 1.0
            @test BlackBoxVLBIImaging.PT.latent_pfwd(t, [30.0]) <= 32.0
            @test BlackBoxVLBIImaging.PT.latent_pfwd(t, BlackBoxVLBIImaging.PT.latent_pback(t, 2.5)) ≈ 2.5
        end

        # the model still evaluates at a draw from the log-normal prior
        pr = Comrade.NamedDist(skyl.prior)
        x = rand(Random.default_rng(), pr)
        @test isfinite(logpdf(pr, x))
        @test all(isfinite, stokes(intensitymap(skyl.f(x, skyl.metadata), skyl.grid), :I))

        # an unknown family, and a family set where there is no ρ to put it on
        @test_throws "unknown rho_prior 'loggaussian'" build_sky_config(
            markovcfg(rho_prior = "loggaussian")
        )
        @test_throws "rho_prior sets the spectral-parameter prior" build_sky_config(
            markovcfg(rho_prior = "lognormal", order = 1)
        )

        @test_throws "is not inside [50.0, 64.0]" markov_rho_prior(LogNormalRhoPrior(), skyl.grid, μas2rad(20.0), 3; lower = 50.0, upper = 64.0)
    end

    @testset "Matérn spectrum and prior" begin
        BB = BlackBoxVLBIImaging
        function materncfg(; kwargs...)
            cfg = exconfig("image.toml")
            cfg["grid"]["nx"] = 32
            cfg["grid"]["ny"] = 32
            cfg["model"]["order"] = 0
            for (k, v) in kwargs
                cfg["model"][String(k)] = v
            end
            return cfg
        end

        # (ℓ, α) is the Matérn (ρ, ν) spectrum with α = 2(ν + 1), ℓ = ρ/√(8ν); genfield
        # normalizes the spectrum, so the two give the same field
        plan = BB.StationaryRandomFieldPlan(imagepixels(1.0, 1.0, 32, 32))
        z = randn(Random.Xoshiro(3), 32, 32)
        ν = 1.5
        ℓ = 5.0
        fslope = BB.genfield(BB.StationaryRandomField(MaternSlopePS(ℓ, 2(ν + 1)), plan), z)
        fmatern = BB.genfield(BB.StationaryRandomField(BB.MaternPS(ℓ * sqrt(8ν), ν), plan), z)
        @test fslope ≈ fmatern rtol = 1.0e-10

        # the moves' amplitude factor is the one genfield applies
        k2, dk = BB._plan_wavenumbers(plan)
        la = BB._log_amplitude(Matern(), (ℓ, 2(ν + 1)), collect(k2), dk)
        @test fslope ≈ real.(BB._hartley(exp.(la) .* z)) ./ 32 rtol = 1.0e-10

        skyl, _ = build_sky_config(materncfg(rho_prior = "lognormal"))
        @test length(skyl.prior.ρa) == 2
        @test logpdf(skyl.prior.ρa[1], 7.0) ≈ logpdf(BB.Distributions.truncated(LogNormal(log(16.0), 1.0); lower = 1.0, upper = 32.0), 7.0)
        @test logpdf(skyl.prior.ρa[2], 3.0) ≈ logpdf(BB.Distributions.truncated(LogNormal(log(2.5), 0.4); lower = 1.0, upper = 8.0), 3.0)
        @test logpdf(skyl.prior.ρa[2], 0.9) == -Inf
        @test logpdf(skyl.prior.ρa[2], 8.1) == -Inf

        skyu, _ = build_sky_config(materncfg())
        @test minimum(skyu.prior.ρa[2]) == 1.0
        @test maximum(skyu.prior.ρa[2]) == 8.0

        @testset "polrep $polrep evaluates at a prior draw" for polrep in ("PolExp", "TotalIntensity", "Poincare")
            sky, _ = build_sky_config(materncfg(polrep = polrep, rho_prior = "lognormal"))
            pr = Comrade.NamedDist(sky.prior)
            x = rand(Random.Xoshiro(4), pr)
            @test isfinite(logpdf(pr, x))
            img = intensitymap(sky.f(x, sky.metadata), sky.grid)
            @test all(isfinite, polrep == "TotalIntensity" ? baseimage(img) : stokes(img, :I))
        end
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
                kinds = ("flux_gain", "field_scale", "rho_field", "mean_field")
                ms = build_moves(postm, [MoveSpec(; kind) for kind in kinds], θs[1])
                @test collect(Comrade.move_name.(ms.moves)) == [
                    "flux_gain",
                    "field_scale[a]", "field_scale[b]", "field_scale[c]", "field_scale[d]",
                    "rho_field[a,1]", "rho_field[a,2]", "rho_field[b,1]", "rho_field[b,2]",
                    "rho_field[c,1]", "rho_field[c,2]", "rho_field[d,1]", "rho_field[d,2]",
                    "mean_field[fwhm]", "mean_field[fb]",
                ]
                tpm = ms.ctx.view.tbase
                stokesmap(θ) = baseimage(intensitymap(skymodel(postm, θ), grid))
                rel(a, b) = maximum(abs, a .- b) / maximum(abs, b)

                # reversal, log-determinant against finite differences and likelihood
                # invariance at every move; prior stationarity of one move per kind
                ksmoves = ("flux_gain", "field_scale[a]", "rho_field[a,1]", "mean_field[fwhm]")
                @testset "check_move: $(Comrade.move_name(m))" for m in ms.moves
                    nprior = Comrade.move_name(m) in ksmoves ? 300 : 0
                    r = check_move(m, postm, θs; nprior, rng = Random.Xoshiro(21))
                    @test r.loglikelihood < 1.0e-8 * abs(Comrade.loglikelihood(postm, θs[1]))
                end

                @testset "the polarized image is unchanged pixel by pixel: $(Comrade.move_name(m))" for m in ms.moves[2:end]
                    x = Comrade.inverse(tpm, θs[2])
                    x′, _ = Comrade.propose(m, x, 0.1, ms.ctx)
                    s0, s1 = stokesmap(θs[2]), stokesmap(Comrade.transform(tpm, x′))
                    for p in (:I, :Q, :U, :V)
                        @test rel(stokes(s1, p), stokes(s0, p)) < 1.0e-10
                    end
                end

                # field_scale[a] with its Jacobian dropped
                fa = ms.moves[2]
                nojac = Comrade.CompensatedMove(
                    fa.name, fa.ishift, fa.block, fa.compensate, (vb, x, x′, ctx) -> 0.0, fa.initial_scale,
                    fa.invariant, fa.context, fa.traceable
                )
                @test_throws "reports logdet = 0.0" check_move(nojac, postm, θs[1:1])

                sub = build_moves(
                    postm, [MoveSpec(; kind = "rho_field", params = ["b"], rounds = 3, target_accept = 0.3)], θs[1]
                )
                @test collect(Comrade.move_name.(sub.moves)) == ["rho_field[b,1]", "rho_field[b,2]"]
                @test sub.rounds == [3, 3] && sub.target_accept == [0.3, 0.3]

                # The hook through a preconditioner makes the same proposals and decisions on
                # the host and on the device, and writes its statistics.
                @testset "MoveSet hook, host and device" begin
                    n = dimension(tpm)
                    prng = Random.Xoshiro(11)
                    V = Matrix(qr(randn(prng, n, 3)).Q)[:, 1:3]
                    pre = LowRankPreconditioner(randn(prng, n), exp.(0.3 .* randn(prng, n)), V, [3.0, 0.5, 1.5])
                    tph = BB.PT.transport_to(postm, pre)
                    postc = BB.ConstructionBase.setproperties(postm, (; admode = nothing))
                    rpost = Comrade.prepare_device(postc, Comrade.ComradeBase.ReactantEx())
                    tpd = BB.PT.transport_to(rpost, Comrade._device_pre(pre))
                    z0 = Comrade._affine_inv(pre, Comrade.inverse(tpm, θs[1]))
                    info = (; phase = :warmup, step = 10, total = 100)
                    specs = [MoveSpec(; kind, rounds = 2) for kind in kinds]
                    out = tempname()
                    msh = build_moves(postm, specs, θs[1]; output = out)
                    msd = build_moves(postm, specs, θs[1])
                    sh = msh(ProbProg.MCMCState(copy(z0), nothing, nothing, 0.1, nothing, nothing), tph, info, Random.Xoshiro(12))
                    sd = msd(ProbProg.MCMCState(BB.Reactant.to_rarray(z0), nothing, nothing, 0.1, nothing, nothing), tpd, info, Random.Xoshiro(12))
                    @test sh.position != z0
                    @test Array(sd.position) ≈ sh.position rtol = 1.0e-9
                    summary = move_summary(msh)
                    @test [m.warmup for m in move_summary(msd)] == [m.warmup for m in summary]
                    @test all(m -> m.warmup.proposed == 2, summary)
                    @test any(m -> m.warmup.accepted > 0, summary)
                    @test BB.Serialization.deserialize(out) == summary
                end

                @testset "moves that do not apply are rejected up front" begin
                    @test_throws "unknown move kind \"gain_phase\"" build_moves(postm, [MoveSpec(; kind = "gain_phase")], θs[1])
                    @test_throws "params [\"z\"] are not among the non-centered sky fields" build_moves(
                        postm, [MoveSpec(; kind = "field_scale", params = ["z"])], θs[1]
                    )
                    fcfg = deepcopy(icfg)
                    fcfg["flux"]["ftot"] = [0.8]
                    skyfix, imgfix = build_sky_config(fcfg)
                    postfix = VLBIPosterior(skyfix, intg, dcoh; imgdata = imgfix)
                    @test_throws "needs a sampled total flux" build_moves(
                        postfix, [MoveSpec(; kind = "flux_gain")], prior_sample(Random.Xoshiro(1), postfix)
                    )
                    # a first-stamp pin on lg1 cannot follow the common shift
                    pcfg = deepcopy(instr)
                    pcfg["priors"]["lg1"]["init"] = Dict("kind" => "fixed", "value" => 0.0)
                    postpin = VLBIPosterior(skym, build_instrument_config(pcfg), dcoh; imgdata = imgm)
                    @test_throws "move flux_gain changed the log-likelihood" build_moves(
                        postpin, [MoveSpec(; kind = "flux_gain")], prior_sample(Random.Xoshiro(1), postpin)
                    )
                    # a GMRF sky has neither spectral parameters nor the PolExp stationary-field mean
                    gcfg = deepcopy(icfg)
                    gcfg["model"]["order"] = 1
                    delete!(gcfg["model"], "rho_prior")
                    skyg, imgg = build_sky_config(gcfg)
                    postg = VLBIPosterior(skyg, intg, dcoh; imgdata = imgg)
                    θg = prior_sample(Random.Xoshiro(1), postg)
                    @test_throws "needs a stationary random-field sky model" build_moves(postg, [MoveSpec(; kind = "rho_field")], θg)
                    @test_throws "needs the PolExp stationary random-field sky model" build_moves(postg, [MoveSpec(; kind = "mean_field")], θg)
                end

                @testset "Matérn sky: spectral-parameter and mean-field moves" begin
                    mcfg = deepcopy(icfg)
                    mcfg["model"]["order"] = 0
                    skyw, imgw = build_sky_config(mcfg)
                    postw = VLBIPosterior(skyw, intg, dcoh; imgdata = imgw)
                    θw = [prior_sample(Random.Xoshiro(k), postw) for k in 1:2]
                    mw = build_moves(postw, [MoveSpec(; kind) for kind in ("rho_field", "mean_field")], θw[1])
                    @test collect(Comrade.move_name.(mw.moves)) == [
                        "rho_field[a,1]", "rho_field[a,2]", "rho_field[b,1]", "rho_field[b,2]",
                        "rho_field[c,1]", "rho_field[c,2]", "rho_field[d,1]", "rho_field[d,2]",
                        "mean_field[fwhm]", "mean_field[fb]",
                    ]
                    @testset "check_move: $(Comrade.move_name(m))" for m in mw.moves
                        nprior = Comrade.move_name(m) in ("rho_field[a,2]", "mean_field[fwhm]") ? 300 : 0
                        r = check_move(m, postw, θw; nprior, rng = Random.Xoshiro(21))
                        @test r.loglikelihood < 1.0e-8 * abs(Comrade.loglikelihood(postw, θw[1]))
                    end
                    tpw = mw.ctx.view.tbase
                    @testset "the polarized image is unchanged pixel by pixel: $(Comrade.move_name(m))" for m in mw.moves
                        x = Comrade.inverse(tpw, θw[2])
                        x′, _ = Comrade.propose(m, x, 0.1, mw.ctx)
                        s0 = baseimage(intensitymap(skymodel(postw, θw[2]), grid))
                        s1 = baseimage(intensitymap(skymodel(postw, Comrade.transform(tpw, x′)), grid))
                        for p in (:I, :Q, :U, :V)
                            @test rel(stokes(s1, p), stokes(s0, p)) < 1.0e-10
                        end
                    end
                end

                @testset "StdNormal posterior and Gauss–Newton kernels" begin
                    @test_throws "Cannot transport the circular distribution" BB.stdnormal_posterior(postm)

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

                    tps = BB.stdnormal_posterior(posts)
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
                    θbad = BlackBoxVLBIImaging.@set θ1.sky.mean.fwhm = 0.0
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

                    # every move kind in the StdNormal space, including the ±2π sheet move and the
                    # chain-hyperparameter moves
                    std = BB.PT.StdNormal()
                    allkinds = ("phase_sheet", kinds..., "chain_hyper")
                    sms = build_moves(posts, [MoveSpec(; kind) for kind in allkinds], θd[1]; space = std)
                    @test Comrade.move_name.(sms.moves)[[1, 2, 3, 7, 15, 16]] ==
                        ("phase_sheet", "flux_gain", "field_scale[a]", "rho_field[a,1]", "mean_field[fwhm]", "mean_field[fb]")
                    hnames = filter(startswith("chain_hyper"), Comrade.move_name.(sms.moves))
                    @test "chain_hyper[lg1.σ]" in hnames && "chain_hyper[lg1.τ]" in hnames
                    @test all(sms.free)
                    @testset "check_move (StdNormal): $(Comrade.move_name(m))" for m in sms.moves
                        nprior = Comrade.move_name(m) in ("phase_sheet", "chain_hyper[lg1.σ]", ksmoves...) ? 200 : 0
                        check_move(m, posts, θd; space = std, nprior, rng = Random.Xoshiro(22))
                    end
                    # a sheet move shifts one path by 2π from its start point up to the next fixed point
                    sheet = sms.moves[1]
                    p = findfirst(q -> length(q[4]) > 1, sheet.points)
                    _, _, Ip, free = sheet.points[p]
                    u1, _ = Comrade.propose(sheet, us[1], (p, free[2], 1), sms.ctx)
                    d = parent(Comrade.transform(tps, u1).instrument.gprat.params) .- parent(xs1.instrument.gprat.params)
                    moved = findall(>(1.0e-9) ∘ abs, d)
                    @test !isempty(moved) && all(≈(2π; atol = 1.0e-9), d[moved])
                    @test Ip[free[2]] in moved && issubset(moved, Ip)

                    rm = BB.ResidualMap(posts)
                    Φ = [sum(abs2, BB.whitened_residuals(rm, tps, u)) / 2 for u in us]
                    @test Φ[1] - Φ[2] ≈ ll(θd[2]) - ll(θd[1]) rtol = 1.0e-10

                    M = Tuple(copy.(rm.measurement))
                    N = Tuple(copy.(rm.noise))
                    M[2][3] = NaN
                    N[4][5] = Inf
                    @test length(BB.ResidualMap(M, N)) == length(rm) - 4
                    N[1][1] = 0.0
                    @test_throws "non-positive noise" BB.ResidualMap(M, N)

                    k = BB.GaussNewtonKernels(posts)
                    prng = Random.Xoshiro(21)
                    u, v, w = us[1], randn(prng, n), randn(prng, n)
                    h = 1.0e-5
                    fd(f, v) = (f(u .+ h .* v) .- f(u .- h .* v)) ./ (2h)
                    resid(u) = BB.whitened_residuals(rm, tps, u)
                    Jv, Jw = fd(resid, v), fd(resid, w)

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
                    sub = BB.gauss_newton_subspace(k, us; rank = 10, threshold = thr, power = 4, rng = Random.Xoshiro(5))
                    @test length(sub) == 6
                    @test sub.λ ≈ Ed.values[1:6] rtol = 1.0e-6

                    # the curvature function of the Gauss–Newton adaptor acts in the latent
                    # coordinates the sampler's StdNormal space uses, column by column
                    tsamp = Comrade.transport_to(posts, BB.PT.StdNormal())
                    @test Comrade.transform(tsamp, u) == Comrade.transform(tps, u)
                    curv = BB.gauss_newton_curvature(k)
                    @test curv(u, [v w]) ≈ [GNv GNw] rtol = 1.0e-12
                    # with a complete probe set the sketch is exact
                    ga = Comrade.GaussNewtonLowRank(curv; rank = n - 4, oversample = 4, threshold = thr, min_draws = 2)
                    st = Comrade.init_metric_adaptation(ga)
                    foreach(x -> Comrade.observe_draw!(ga, st, nothing, x, zero(x)), us)
                    fit = Comrade.metric_refit(ga, st)
                    @test 1 ./ fit.s .^ 2 .- 1 ≈ Ed.values[1:6] rtol = 1.0e-6

                    # directions confined to the sky-field modes within a band of the data
                    @test isnothing(BB.gauss_newton_rows(posts, Inf))
                    @test isnothing(BB.gauss_newton_rows(posts, 1.0e6))
                    rws = BB.gauss_newton_rows(posts, 0.5)
                    vs = Comrade.CoordinateView(posts, BB.PT.StdNormal())
                    fieldc = reduce(vcat, [collect(Comrade.coords(vs, (:sky, X))) for X in (:a, :b, :c, :d)])
                    @test issorted(rws) && issubset(setdiff(1:n, fieldc), rws)
                    mdp = posts.skymodel.metadata
                    km = hypot.(mdp.base.plan.kx, mdp.base.plan.ky') ./ (π * abs(step(mdp.grid.X)))
                    umaxp = maximum(hypot.(Comrade.datatable(posts.data[1]).baseline.U, Comrade.datatable(posts.data[1]).baseline.V))
                    ca = collect(Comrade.coords(vs, (:sky, :a)))
                    @test intersect(rws, ca) == ca[vec(km .<= 0.5 * umaxp)]
                    @test 0 < count(in(rws), ca) < length(ca)
                    gr = Comrade.GaussNewtonLowRank(curv; rank = 6, oversample = 4, threshold = thr, min_draws = 2, rows = rws)
                    sr = Comrade.init_metric_adaptation(gr)
                    foreach(x -> Comrade.observe_draw!(gr, sr, nothing, x, zero(x)), us)
                    fr = Comrade.metric_refit(gr, sr)
                    @test fr.V isa Comrade.RowSupportedMatrix && fr.V.rows == rws
                end
            end
        else
            @test_skip "test data not found at $datafile"
        end
    end
end
