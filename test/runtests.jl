using BlackBoxVLBIImaging
using Test
using TOML
using Random
using LinearAlgebra
using Statistics: cov

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

    @testset "low-rank preconditioner" begin
        TV = BlackBoxVLBIImaging.TV
        rng = Random.Xoshiro(42)
        n, m = 24, 3
        V = Matrix(qr(randn(rng, n, m)).Q)[:, 1:m]
        b = randn(rng, n)
        d = exp.(0.3 .* randn(rng, n))
        s = [8.0, 4.0, 2.5]
        p = LowRankPreconditioner(b, d, V, s)
        A = Diagonal(d) * (I + V * Diagonal(s .- 1) * V')

        z = randn(rng, n)
        @test BlackBoxVLBIImaging._affine_fwd(p, z) ≈ b .+ A * z
        @test BlackBoxVLBIImaging._affine_inv(p, BlackBoxVLBIImaging._affine_fwd(p, z)) ≈ z
        @test BlackBoxVLBIImaging._affine_logdet(p) ≈ first(logabsdet(A))

        @test_throws ArgumentError LowRankPreconditioner(b, d, randn(rng, n, m), s)
        @test_throws ArgumentError LowRankPreconditioner(b, -d, V, s)
        @test_throws DimensionMismatch LowRankPreconditioner(b[1:3], d, V, s)

        # rank-0: pure diagonal standardization
        p0 = LowRankPreconditioner(b, d, zeros(n, 0), Float64[])
        @test BlackBoxVLBIImaging._affine_fwd(p0, z) ≈ b .+ d .* z
        @test BlackBoxVLBIImaging._affine_logdet(p0) ≈ sum(log, d)

        # TV node over an identity inner transform: x = b + A z, constant log-Jacobian
        t = BlackBoxVLBIImaging.PreconditionedFlat(p, TV.as(Array, n))
        x, ℓ, ix = TV.transform_with(TV.LogJac(), t, z, 1)
        @test x ≈ b .+ A * z
        @test ℓ ≈ first(logabsdet(A))
        @test ix == n + 1
        @test TV.inverse(t, x) ≈ z

        # bounded inner transform: log-Jacobian is the inner's at (b + A z) plus the constant
        tb = BlackBoxVLBIImaging.PreconditionedFlat(p, TV.as(Array, TV.as𝕀, n))
        xb, ℓb, _ = TV.transform_with(TV.LogJac(), tb, z, 1)
        xi, ℓi, _ = TV.transform_with(TV.LogJac(), TV.as(Array, TV.as𝕀, n), b .+ A * z, 1)
        @test xb ≈ xi
        @test ℓb ≈ ℓi + first(logabsdet(A))
        @test TV.inverse(tb, xb) ≈ z

        # estimation: a planted correlation ridge is found and whitened away
        nd = 40
        u = normalize(randn(rng, nd))
        Σhalf = I + 9.0 * u * u'                       # sd 10 along u, 1 elsewhere
        Z = Σhalf * randn(rng, nd, 4000)
        pre = BlackBoxVLBIImaging._lowrank_from_draws(Z; rank = 4)
        @test length(pre.s) == 1                       # only the planted direction survives
        us = normalize(u ./ pre.d)                     # the ridge in standardized coordinates
        @test abs(dot(pre.V[:, 1], us)) > 0.99
        W = reduce(hcat, [BlackBoxVLBIImaging._affine_inv(pre, c) for c in eachcol(Z)])
        # The correction removes the SOFT (large-eigenvalue) directions; the standardized
        # correlation matrix also has genuinely small eigenvalues, which a top-only
        # correction leaves alone, so assert only the upper end of the spectrum.
        @test maximum(eigvals(Symmetric(cov(W; dims = 2)))) < 1.5   # planted λ ≈ 20 is gone
        @test_throws ArgumentError BlackBoxVLBIImaging._lowrank_from_draws(
            vcat(Z, zeros(1, 4000)); rank = 4
        )

        # few-draws regime (ndraws ≪ n): the spiked-model shrinkage finds the planted
        # direction but tempers its scale, and pure noise yields no correction at all.
        nf, Nf = 2000, 100
        uf = normalize(randn(rng, nf))
        Zf = (I + 7.0 * uf * uf') * randn(rng, nf, Nf)   # sd 8 along uf, 1 elsewhere
        pf = BlackBoxVLBIImaging._lowrank_from_draws(Zf; rank = 8)
        @test length(pf.s) == 1
        @test 3.0 < only(pf.s) < 8.0                     # cross-validated below the true scale 8
        @test abs(dot(pf.V[:, 1], normalize(uf ./ pf.d))) > 0.5
        pn = @test_logs (:warn, r"diagonal-only") BlackBoxVLBIImaging._lowrank_from_draws(
            randn(rng, nf, Nf); rank = 8
        )
        @test isempty(pn.s)

        # drift guard: a direction the chain trends along (burn-in) is dropped, while a
        # stationary planted spike in the same draws is kept.
        ud = normalize(randn(rng, nf))
        us = normalize(randn(rng, nf)); us .-= dot(us, ud) * ud; normalize!(us)
        Zd = randn(rng, nf, Nf) .+ 6.0 * us * randn(rng, 1, Nf)
        Zd .+= 20.0 * ud * collect(range(-1, 1, Nf))'
        pd = @test_logs (:warn, r"trending") BlackBoxVLBIImaging._lowrank_from_draws(
            Zd; rank = 8
        )
        @test all(abs.(pd.V' * normalize(ud ./ pd.d)) .< 0.3)
        @test any(abs.(pd.V' * normalize(us ./ pd.d)) .> 0.5)

        # carry (augment): a previous round's directions are kept and deflated out of
        # detection, so a refit adds new structure instead of replacing what works.
        uc = normalize(randn(rng, nf))
        uc2 = normalize(randn(rng, nf)); uc2 .-= dot(uc2, uc) * uc; normalize!(uc2)
        Zc = (I + 7.0 * uc * uc' + 5.0 * uc2 * uc2') * randn(rng, nf, Nf)
        pc1 = BlackBoxVLBIImaging._lowrank_from_draws(Zc; rank = 1)
        Zc2 = (I + 7.0 * uc * uc' + 5.0 * uc2 * uc2') * randn(rng, nf, Nf)
        pc2 = BlackBoxVLBIImaging._lowrank_from_draws(Zc2; rank = 8, carry = pc1)
        @test length(pc2.s) >= 2
        @test opnorm(pc2.V' * pc2.V - I) < 1e-8
        @test maximum(abs.(pc2.V' * normalize(uc ./ pc2.d))) > 0.5
        @test maximum(abs.(pc2.V' * normalize(uc2 ./ pc2.d))) > 0.4
        @test length(BlackBoxVLBIImaging._lowrank_from_draws(Zc2; rank = 1, carry = pc1).s) == 1

        # angle-pair whitening: (sin, cos) pairs are detected from their unit-circle
        # signature, wedge-aligned and rescaled; near-uniform (ring) pairs are skipped.
        na = 30
        Za = randn(rng, na, 400)
        θa = 0.9 .+ 0.05 .* randn(rng, 400)
        Za[21, :] .= sin.(θa); Za[22, :] .= cos.(θa)
        θr = 2π .* rand(rng, 400)
        Za[25, :] .= sin.(θr); Za[26, :] .= cos.(θr)
        prea = BlackBoxVLBIImaging._lowrank_from_draws(Za; rank = 2)
        pa = BlackBoxVLBIImaging._angle_pairs_from_draws(Za, prea)
        @test pa isa AnglePairPreconditioner
        @test pa.i1 == [21]                       # ring pair at 25:26 skipped
        za = randn(rng, na)
        @test BlackBoxVLBIImaging._affine_inv(pa, BlackBoxVLBIImaging._affine_fwd(pa, za)) ≈ za
        Aa = reduce(hcat, [
            BlackBoxVLBIImaging._affine_fwd(pa, Matrix(I, na, na)[:, k]) .-
                BlackBoxVLBIImaging._affine_fwd(pa, zeros(na)) for k in 1:na
        ])
        @test BlackBoxVLBIImaging._affine_logdet(pa) ≈ first(logabsdet(Aa))
        # true latents (radial jitter restored) whiten to ~unit isotropic in the pair
        θt = 0.9 .+ 0.05 .* randn(rng, 2000)
        rt = exp.(0.25 .* randn(rng, 2000))
        Xt = repeat(BlackBoxVLBIImaging._affine_fwd(pa, zeros(na)), 1, 2000)
        Xt[21, :] .= rt .* sin.(θt); Xt[22, :] .= rt .* cos.(θt)
        Zt = reduce(hcat, [BlackBoxVLBIImaging._affine_inv(pa, c) for c in eachcol(Xt)])
        Ct = cov(Zt[[21, 22], :]')
        @test 0.5 < Ct[1, 1] < 2.0 && 0.5 < Ct[2, 2] < 2.0
        @test abs(Ct[1, 2]) / sqrt(Ct[1, 1] * Ct[2, 2]) < 0.3

        # gradient balance: _scale_rows divides the map's response on chosen coordinates
        # by f — exactly for rows carrying no low-rank mass — through the diagonal for
        # plain coordinates and through the block rows for angle pairs.
        fb = ones(na); fb[21] = 5.0; fb[22] = 3.0; fb[15] = 7.0
        pb = BlackBoxVLBIImaging._scale_rows(pa, fb)
        Jb = reduce(hcat, [
            BlackBoxVLBIImaging._affine_fwd(pb, Matrix{Float64}(I, na, na)[:, k]) .-
                BlackBoxVLBIImaging._affine_fwd(pb, zeros(na)) for k in 1:na
        ])
        for i in (21, 22, 15)
            @test Jb[i, :] ≈ Aa[i, :] ./ fb[i]
        end
        @test BlackBoxVLBIImaging._affine_logdet(pb) ≈ first(logabsdet(Jb))
        @test BlackBoxVLBIImaging._affine_inv(pb, BlackBoxVLBIImaging._affine_fwd(pb, za)) ≈ za

        # Fisher-divergence estimator: exact whitening in the N > d regime, large
        # condition-number reduction (incl. stiff directions invisible to draw fits)
        # in the N << d regime.
        nfd = 12
        dd = [100.0, 25.0, 9.0, 1e-4, 1e-2, 0.04, 1, 1, 1, 1, 1, 1]
        Qf = Matrix(qr(randn(rng, nfd, nfd)).Q)
        Σf = Symmetric(Qf * Diagonal(dd) * Qf')
        Xf = sqrt(Σf) * randn(rng, nfd, 40) .+ randn(rng, nfd)
        Gf = -(Σf \ (Xf .- mean(Xf; dims = 2)))
        pf2 = BlackBoxVLBIImaging._fisher_lowrank(Xf, Gf; rank = 12, cutoff = 1.3)
        Af = Diagonal(pf2.d) * (I + pf2.V * Diagonal(pf2.s .- 1) * pf2.V')
        evf = eigvals(Symmetric(Af \ Matrix(Σf) / Af'))
        @test maximum(evf) / minimum(evf) < 1.5          # exact regime: fully whitened
        zf = randn(rng, nfd)
        @test BlackBoxVLBIImaging._affine_inv(pf2, BlackBoxVLBIImaging._affine_fwd(pf2, zf)) ≈ zf

        # fitting-config plumbing
        strat = build_fitting_config(
            Dict{String, Any}(
                "precondition" => Dict{String, Any}("pilot" => "/tmp/pilotrun", "rank" => 8),
            )
        )
        @test strat.precond_pilot == "/tmp/pilotrun"
        @test strat.precond_rank == 8
        @test strat.precond_nsamples == 2000
        @test isnothing(strat.precond_min_scale)
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
        else
            @test_skip "test data not found at $datafile"
        end
    end
end
