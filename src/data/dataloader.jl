# Coherency-data loaders for the two supported file formats. Both apply scan/fixed-interval
# averaging, optional time-range selection, and fractional-noise inflation.

function build_data_uvfits(
        file::String, array::String;
        avg = "scan",
        ferr::Float64 = 0.005,
        trange = nothing,
        IF = nothing,
    )

    uvd = VLBIFiles.load(VLBIFiles.UVData, file)

    if avg == "scan"
        tavg = VLBI.GapBasedScans()
    else
        tavg = VLBI.FixedTimeIntervals(parse(Float64, avg) * VLBIFiles.Unitful.u"s")
    end

    # The array file describes the antenna feed-rotation parameters; we apply it via
    # `reset_mounts!` after extracting the coherency table.
    #
    # `IF` selects a single intermediate frequency (1-based index into the sorted unique
    # frequencies). When set we keep the IFs separate (`frequency_average = false`) and filter
    # to the requested one; otherwise the default frequency-averaged band is extracted.
    dcoh = extract_table(
        uvd, Coherencies(;
            time_average = tavg,
            frequency_average = isnothing(IF),
        )
    )
    if !isnothing(IF)
        ifs = sort(unique(dcoh.config.datatable.Fr))
        (IF isa Integer && 1 <= IF <= length(ifs)) ||
            error("IF=$IF is invalid; IF is 1-based and the data has $(length(ifs)) IF(s) (use 1..$(length(ifs))).")
        dcoh = filter(d -> d.baseline.Fr == ifs[IF], dcoh)
        @info "Selected IF $IF/$(length(ifs)) (frequency = $(ifs[IF] / 1.0e9) GHz)"
    end
    if !isnothing(trange)
        dcoh = filter(d -> d.baseline.Ti ∈ trange, dcoh)
    end
    reset_mounts!(dcoh, array)

    dcoh = add_fractional_noise(dcoh, ferr)
    return dcoh
end

function build_data_dlist(
        file::String, array::String;
        avg = "scan",
        ferr::Float64 = 0.005,
        trange = nothing,
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
