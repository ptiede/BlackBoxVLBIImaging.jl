# Data loaders for the two supported file formats. Both apply scan/fixed-interval averaging,
# optional time-range selection, and fractional-noise inflation. The Comrade data product is
# chosen from the sky model's polarization representation (`polrep`): a Stokes-I-only sky
# (`TotalIntensity`) fits complex visibilities, while a polarized sky fits coherency matrices.

# Map a polarization representation to the Comrade data-product type to extract.
data_product(::TotalIntensity) = Visibilities
data_product(::PolRep) = Coherencies

function build_data_uvfits(
        file::String, array::Union{String, Nothing} = nothing;
        avg = "scan",
        ferr::Float64 = 0.005,
        trange = nothing,
        IF = nothing,
        polrep::PolRep = PolExp(),
        conjugate::Bool = false,
        ignore_feed_labels::Bool = false,
    )

    uvd = VLBIFiles.load(VLBIFiles.UVData, file)
    ignore_feed_labels && force_circular_feed_labels!(uvd)

    if avg == "scan"
        tavg = VLBI.GapBasedScans()
    else
        tavg = VLBI.FixedTimeIntervals(parse(Float64, avg) * VLBIFiles.Unitful.u"s")
    end

    # The array file describes the antenna feed-rotation parameters; when given we apply it
    # via `reset_mounts!` after extracting the data table. It is optional: without it the
    # feed-rotation/mount metadata already in the data file is kept (fine for Stokes I; for
    # polarized fits supply an array file if the data-file mounts are wrong).
    #
    # `IF` selects a single intermediate frequency (1-based index into the sorted unique
    # frequencies). When set we keep the IFs separate (`frequency_average = false`) and filter
    # to the requested one; otherwise the default frequency-averaged band is extracted.
    product = data_product(polrep)
    dvis = extract_table(
        uvd, product(;
            time_average = tavg,
            frequency_average = isnothing(IF),
        )
    )
    if !isnothing(IF)
        ifs = sort(unique(dvis.config.datatable.Fr))
        (IF isa Integer && 1 <= IF <= length(ifs)) ||
            error("IF=$IF is invalid; IF is 1-based and the data has $(length(ifs)) IF(s) (use 1..$(length(ifs))).")
        dvis = filter(d -> d.baseline.Fr == ifs[IF], dvis)
        @info "Selected IF $IF/$(length(ifs)) (frequency = $(ifs[IF] / 1.0e9) GHz)"
    end
    if !isnothing(trange)
        dvis = filter(d -> d.baseline.Ti ∈ trange, dvis)
    end
    if isnothing(array)
        @info "No array file specified; keeping the feed-rotation/mount metadata from the data file as-is."
    else
        reset_mounts!(dvis, array)
    end

    dvis = add_fractional_noise(dvis, ferr)
    conjugate && (dvis = conjugate_data(dvis))
    return dvis
end

"""
    force_circular_feed_labels!(uvd::VLBIFiles.UVData)

Relabel every antenna's feeds as `(:R, :L)` so the file's `RR/LL/RL/LR` slots are read as-is
for every baseline. Use this when the antenna table (POLTYA/POLTYB) marks some stations
linear: VLBIFiles then translates their products to mixed labels (e.g. `YR`) that the Comrade
extractor rejects, and the labels may not even be right. After loading, declare the linear
stations with `[flags] corr_polbasis` (e.g. `{site = "AA", R = "X", L = "Y"}`), which is the
single place the feed bookkeeping is meant to live. Only valid for circular-slot files.
"""
function force_circular_feed_labels!(uvd)
    stokes = uvd.header.stokes
    all(s -> s in (:RR, :LL, :RL, :LR), stokes) || error(
        "ignore_feed_labels requires a file whose stokes axis is RR/LL/RL/LR; got $stokes"
    )
    for arr in uvd.ant_arrays, (id, ant) in pairs(arr.antennas)
        ant.poltypes == (:R, :L) && continue
        @info "ignore_feed_labels: $(ant.name) antenna-table feeds $(ant.poltypes) treated as (R, L)"
        arr.antennas[id] = VLBIFiles.VLBIData.Antenna(;
            name = ant.name, xyz = ant.xyz, mount_type = ant.mount_type,
            poltypes = (:R, :L), feed_offsets = ant.feed_offsets,
        )
    end
    return uvd
end

"""
    conjugate_data(dvis) -> dvis

Flip the Fourier-phase sign convention of every datum: uvfits written with the opposite
convention (e.g. FITS-IDI-style phases vs. the AIPS/ehtim convention Comrade assumes) store
`V(-u,-v)` under the `(u,v)` label, i.e. the Hermitian adjoint of the coherency matrix (the
complex conjugate for a Stokes I visibility). Also transposes the noise matrix so the
RL/LR entries follow their measurements. Closure phases change sign under this operation.
"""
function conjugate_data(dvis::Comrade.EHTObservationTable{T}) where {T}
    meas = map(adjoint, dvis.measurement)
    noise = map(transpose, dvis.noise)
    return Comrade.EHTObservationTable{T}(meas, noise, dvis.config)
end

"""
    repair_scan_coverage(dvis) -> dvis

Fixed-interval time averaging can put a bin-center timestamp outside the scan boundaries
recorded in the data file (data taken just before the nominal scan start average to a
center ahead of it). Scan-segmented instrument parameters look segments up with the
half-open interval `start ≤ t < stop`, so such orphan timestamps make `set_array` throw
"not found in SiteArray". Stretch the nearest scan edge to cover each orphan time.
"""
function repair_scan_coverage(dvis)
    arr = arrayconfig(dvis)
    sc = arr.scans
    start = copy(sc.start)
    stop = copy(sc.stop)
    orphans = eltype(start)[]
    for t in unique(arr[:Ti])
        any(i -> start[i] <= t < stop[i], eachindex(start, stop)) && continue
        push!(orphans, t)
        dist(i) = t < start[i] ? start[i] - t : t - stop[i]
        i = argmin(dist, eachindex(start, stop))
        if t < start[i]
            start[i] = t
        else
            # half-open upper edge: nudge past t so `t < stop` holds
            stop[i] = nextfloat(t)
        end
    end
    isempty(orphans) && return dvis
    @warn "$(length(orphans)) averaged timestamp(s) fell outside the scan table; " *
        "stretched the nearest scan edges to cover them." orphans
    newscans = StructArray((start = start, stop = stop))
    newarr = @set arr.scans = newscans
    return @set dvis.config = newarr
end

function build_data_dlist(
        file::String, array::Union{String, Nothing} = nothing;
        avg = "scan",
        ferr::Float64 = 0.005,
        trange = nothing,
        polrep::PolRep = PolExp(),
    )

    # Unlike uvfits, the array file is required for dlist: it provides the antenna table used
    # to construct the array configuration (the dlist itself carries no array metadata).
    isnothing(array) && error(
        "dlist files require an 'array' file (it provides the antenna table used to build " *
            "the array configuration)."
    )

    # dlist files are mixed-polarization coherency matrices by construction; there is no
    # Stokes-I-visibility extraction path for them.
    polrep isa TotalIntensity && error(
        "dlist files are polarized coherencies; TotalIntensity (complex-visibility) fitting " *
            "is not supported for dlist. Use a polarized polrep (PolExp/Poincare) or convert " *
            "the data to uvfits."
    )

    avg != "scan" && @warn "Only 'scan' averaging is supported for dlist files. Ignoring the --avg flag."

    dcoh0 = read_dlist(file, array)
    if !isnothing(trange)
        dcoh1 = filter(x -> trange[1] < x.baseline.Ti < trange[2], dcoh0)
    else
        dcoh1 = dcoh0
    end
    dcoh2 = add_fractional_noise(dcoh1, ferr)
    return dcoh2
end
