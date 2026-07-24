using BlackBoxVLBIImaging
using Test
using TOML
using Random

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

        # the shipped gauss-markov example TOML is valid
        @test build_instrument_config(exconfig("instrument_gaussmarkov.toml")) isa InstrumentModel

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
        @test corpol(RPol(), flags.corr_polbasis[1].mapping) == YPol()
        @test_throws ErrorException corpol(XPol(), flags.corr_polbasis[1].mapping)
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
