using Pkg; Pkg.activate(@__DIR__)
using Comonicon
include(joinpath(@__DIR__, "deps.jl"))


"""
    Image a VLBI obervation

# Arguments
- `file::String`: Path to the UVFITS file.

# Options
- `--ftot`: The total flux. Can either we two numbers, i.e. 0.1, 2.5 which mean it fits the
            total flux within that range, or a single number which means it fixes the total flux
            using an apriori flux estimate.
- `--outpath::String`: Path to the output directory.
- `--fovx::Float64`: Field of view in the x-direction in μas (default is `200.0`).
- `--fovy::Float64`: Field of view in the y-direction in μas (default is `200.0`).
- `--nx::Int`: Number of pixels in the x-direction (default is `63`).
- `--ny::Int`: Number of pixels in the y-direction (default is `63`).
- `--pa::Float64`: Position angle in degrees (default is `0.0`).
- `--x0::Float64`: X-coordinate of the image center in μas (default is `0.0`).
- `--y0::Float64`: Y-coordinate of the image center in μas (default is `0.0`).
- `--nsample::Int`: Number of samples for the imager (default is `20_000`).
- `--nadapt::Int`: Number of adaptation steps (default is `10_000`).
- `--maxiters::Int`: Maximum number of iterations (default is `25_000`).
- `--ntrials::Int`: Number of trials (default is `5`).
- `--ferr::Float64`: Fractional noise to add to the coherencies (default is `0.005`).
- `--flagtable::String`: Path to a TOML flag table. Supported top-level keys (all optional):
                         `corr_polbasis = ["HAY", "AA"]` — sites whose polbasis label is flipped;
                         `sites = ["AP"]` — drop all baselines touching these;
                         `baselines = [["AA","LM"]]` — drop these specific (order-independent) baselines;
                         `tranges = [[4.5, 5.2]]` — drop datums with Ti in these UT-decimal-hour ranges;
                         `uvranges = [[0.0, 0.1]]` — drop datums with uvdist in these Gλ ranges.
                         Default `""` (no-op).
- `--avg::String`: Timescale of averaging. Can either be "scan" or a number in seconds (default is `"scan"`).
- `--order::Int`: Order of the GMRF (default is `2`).
- `--mean::String`: The type of mean model to use for the sky model. Options are `GaussBkgd`, `Bkgd`, and `Gauss` (default is `GaussBkgd`).
- `--trange::String`: Time range to select data decmial hours as "start,end" (default is `nothing`).
- `--start::String`: Path to a starting image file (default is `""` which means no starting image).
- `--array::String`: Path to the array file which describes the array geometry and the feed types of the sites. See `src/arrays/` for examples.
# Flags
- `--restart::Bool`: Flag to restart the imaging process (default is `false)
- `--addgauss::Bool`: Flag to add a Gaussian component to the sky model (default is `false`).
- `--hier::Bool`: Flag to use a hierarchical gain amplitude model (default is `false`).
- `--creg::Bool`: Flag to use a centering regularization (default is `false`).
- `--polconvert::Bool`: Flag that signals polconvert has been applied to the data (default is `false`).
- `--frcal::Bool`: Flag that the data has been frcal-ed (default is `false`). Only applies to dlist files and is ignored otherwise.
"""
Comonicon.@main function main(
        file::String, array::String;
        ftot::String = "1.2",
        outpath::String = "Runs/mixpol",
        fovx::Float64 = 200.0, fovy::Float64 = 200.0,
        nx::Int = 63, ny::Int = 63,
        avg::String = "scan",
        trange::String = "nothing",
        pa::Float64 = deg2rad(0.0),
        x0::Float64 = 0.0, y0::Float64 = 0.0,
        nsample::Int = 20_000, nadapt::Int = 10_000,
        maxiters::Int = 10_000, ntrials::Int = 5,
        ferr::Float64 = 0.005,
        flagtable::String = "",
        order::Int = 2,
        mean::String = "Bkgd",
        restart::Bool = false,
        addgauss::Bool = false,
        hier::Bool = false,
        creg::Bool = false,
        start::String = "",
        polconvert::Bool = false,
        frcal::Bool = false
        )
    @info "Fitting the data: $file"
    @info "Loading the array file: $array"
    @info "Outputing to $outpath"
    @info "Field of view: ($fovx, $fovy) μas"
    @info "Image center offset: ($x0, $y0) μas"
    @info "PA of the grid $(pa) degrees"
    @info "Adding $ferr fractional error to the data"

    if order == 1
        base = GMRF
    elseif order > 1
        base = NonCenteredMRF(GMRF)
    elseif order < 0
        @info "Using Markov RF expansion of order $(abs(order))"
        base = MarkovRF(abs(order))
    elseif order == 0
        @info "Using Matern covariance for the RF"
        base = Matern()
    else
        throw(ArgumentError("Unknown order: $order"))
    end

    
    if base isa Matern || base isa MarkovRF || (base === GMRF && order == 1) 
        nx2 = nextprod((2,3,5,7), nx)
        ny2 = nextprod((2,3,5,7), ny)
        if (nx2 != nx) || (ny2 != ny)
            @warn "You are using a $base stochastic model with an image size of ($nx, $ny) which is not optimal for perf.\n" *
                  "I am changing the image size to ($nx2, $ny2) which is optimal for $base"
            nx = nx2
            ny = ny2
        end
    elseif base isa NonCenteredMRF && order > 1
        if nx == nextprod((2, 3, 5, 7), nx)
            @warn "This image size is almost optimal. Shrinking number of pixels by one in x for a speed boost"
            nx2 = nx - 1
        elseif nx+1 == nextprod((2, 3, 5, 7), nx+1)
            nx2 = nx
        else
            @warn "You are using a $base stochastic model with an image size of nx=$nx which is not optimal for perf.\n" *
                  "I am changing the nx to ($(nx2-1)) which is optimal for $base"
            nx2 = nextprod((2,3,5,7), nx+1) - 1
        end

        if ny == nextprod((2, 3, 5, 7), ny)
            @warn "This image size is almost optimal. Shrinking number of pixels by one in x for a speed boost"
            ny2 = ny - 1
        elseif ny+1 == nextprod((2, 3, 5, 7), ny+1)
            ny2 = ny
        else
            @warn "You are using a $base stochastic model with an image size of ny=$ny which is not optimal for perf.\n" *
                  "I am changing the ny to ($(ny2-1)) which is optimal for $base"
            ny2 = nextprod((2,3,5,7), ny+1) - 1
        end
        nx = nx2
        ny = ny2
    end
    @info "number of pixels: ($nx, $ny)"




    trange = trange == "nothing" ? nothing : parse.(Float64, split(trange, ","))
    mixed = !polconvert

    @info "I think the data is $(mixed ? "mixed" : "ALMA polconverted")"

    if !isnothing(trange)
        @info "Selecting data in time range: $(trange) decimal hours"
    end

    if endswith(file, ".uvf") || endswith(file, ".uvfits")
        @info "Loading a UVFITS file"
        dcoh = build_data_uvfits(file, array; avg, ferr, trange)
    elseif endswith(file, ".dlist")
        @info "Loading a dlist file"
        dcoh = build_data_dlist(file, array; avg, ferr, trange)
    else
        throw(ArgumentError("Unknown file type: $file"))
    end

    if flagtable != ""
        @info "Reading flag table from $flagtable"
        cfg = read_flagtable(flagtable)
        dcoh = apply_flagtable(dcoh, cfg)
    end

    if addgauss
        @info "Adding a over resolved Gaussian component to the sky model"
    end

    ftots = parse.(Float64, split(ftot, ","))
    if length(ftots) == 1
        @info "Using a fixed flux of $(ftots[1])"
        ftotpr = ftots[1]
    elseif length(ftots) == 2
        @info "Fitting the total flux between $(ftots[1]) and $(ftots[2])"
        ftotpr = Uniform(ftots[1], ftots[2])
    else
        throw(ArgumentError("The --ftot flag should have either one or two values while it parsed $(ftots)"))
    end

    if Threads.nthreads() > 1
        @info "Using $(Threads.nthreads()) threads for imaging"
        ex = ThreadsEx()
    else
        @info "Using a single thread for imaging"
        ex = Serial()
    end

    out = joinpath(
        outpath *
        "_mean=$(mean)_order=$(order)_hier=$(hier)" *
        "_pc=$(polconvert)_frcal=$(frcal)_creg=$(creg)_addgauss=$(addgauss)",
        basename(file)
    )
    g = imagepixels(
            μas2rad(fovx), μas2rad(fovy), nx, ny, 
            μas2rad(x0), μas2rad(y0), posang=deg2rad(pa),
            executor = ex
            )


    @info "Beamsize relative to the pixel size is $(beamsize(dcoh)/pixelsizes(g).X) pixels"
    

    if mean == "GaussBkgd"
        @info "Using a Gaussian background mean for the sky model"
        mmodel = GaussBkgdMean(g)
    elseif mean == "Bkgd"
        @info "Using a background mean for the sky model"
        mimg = intensitymap(modify(Gaussian(), Stretch(μas2rad(50.0) / fwhmfac)), g)
        mmodel = MimgPlusBkg(mimg ./ sum(mimg))
    elseif mean == "Gauss"
        @info "Using a Gaussian mean for the sky model"
        mmodel = GaussMean()
    elseif mean == "Ring"
        @info "Using a Ring mean for the sky model"
        mmodel = DblRingMean()
    elseif mean == "TBlob"
        @info "Using a T dist blob mean for the sky model"
        mmodel = TBlobMean()
    elseif mean == "JetGauss"
        @info "Using a Jet Gaussian mean for the sky model"
        mimg = intensitymap(modify(Gaussian(), Stretch(μas2rad(30.0) / fwhmfac)), g)
        mmodel = JetGauss(mimg./sum(mimg))
    else 
        throw(ArgumentError("Unknown mean type: $mean"))
    end

    if !creg
        msky = ImagingModel(PolExp(), mmodel, g, ftotpr; addgauss, base, order)
        imgdata = nothing

    else
        @info "Using a centroid regularization"
        msky = ImagingModel(PolExp(), mmodel, g, ftotpr; addgauss, base, order, center = false)
        imgdata = (Comrade.ImgNormalData(rad2μas∘centroid, SVector(0.0, 0.0), 1.0),)
    end

    skym = SkyModel(msky, skyprior(msky), g; algorithm = FINUFFTAlg(; threads = Threads.nthreads()))

    if mixed
        intm = build_instrumentmodel_mixed(hier, endswith(file, ".dlist"))
    else
        intm = build_instrumentmodel_pc(hier, endswith(file, ".dlist"), frcal)
    end


    if start != ""
        @info "Loading starting image from $start"
        startx = deserialize(start)
    else
        startx = nothing
    end


    return comrade_imager(
        out,
        skym, intm, dcoh; nsample, nadapt,
        maxiters, ntrials, restart = restart,
        start = startx, imgdata = imgdata
    )
end
