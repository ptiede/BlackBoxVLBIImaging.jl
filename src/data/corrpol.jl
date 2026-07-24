# Polarization-basis correction. `corr_polbasis(dvis, site, mapping)` relabels each recorded
# feed basis for a single site according to source-feed => target-feed pairs. Driven by the
# data-config `corr_polbasis = [...]` list.

const DEFAULT_CORPOL_MAPPING = (RPol() => YPol(), LPol() => XPol())

function corpol(feed, mapping)
    index = findfirst(pair -> feed isa typeof(first(pair)), mapping)
    isnothing(index) && error("cannot correct polarization feed $feed: no mapping was provided")
    return last(mapping[index])
end

function corpol(::PolBasis{F1, F2}, mapping) where {F1, F2}
    corrected1 = corpol(F1(), mapping)
    corrected2 = corpol(F2(), mapping)
    return PolBasis{typeof(corrected1), typeof(corrected2)}()
end

function corr_polbasis(dcoh, site::Symbol, mapping)
    dt = map(datatable(dcoh.config)) do row
        bl = row.sites
        pb = row.polbasis
        if bl[1] == site
            row2 = @set row.polbasis = (corpol(pb[1], mapping), pb[2])
        elseif bl[2] == site
            row2 = @set row.polbasis = (pb[1], corpol(pb[2], mapping))
        else
            row2 = row
        end
        return row2
    end
    # Try to improve inference of the polbasis type
    dt2 = @set dt.polbasis = Comrade.StructArray(convert.(Tuple{PolBasis, PolBasis}, dt.polbasis))
    dt3 = Comrade.StructArray(dt2, unwrap = (T -> (T <: Tuple || T <: Comrade.AbstractBaselineDatum || T <: Comrade.SArray || T <: NamedTuple)))
    conf2 = Comrade.rebuild(dcoh.config, dt3)

    T = Comrade.EHTCoherencyDatum{eltype(real(dcoh[1].measurement)), eltype(dt3), eltype(dcoh.measurement), eltype(dcoh.noise)}
    return Comrade.EHTObservationTable{T}(dcoh.measurement, dcoh.noise, conf2)
end

corr_polbasis(dcoh, site::Symbol) = corr_polbasis(dcoh, site, DEFAULT_CORPOL_MAPPING)

ConstructionBase.constructorof(::Type{<:Comrade.EHTObservationTable{T}}) where {T} = Comrade.EHTObservationTable{T}
ConstructionBase.constructorof(::Type{<:Comrade.EHTArrayConfiguration{T}}) where {T} = Comrade.EHTArrayConfiguration
