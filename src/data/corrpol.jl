# Polarization-basis correction. `corr_polbasis(dvis, site)` flips the recorded feed basis
# for a single site (e.g. marking a station's feeds as linear when the data labeled them
# circular). Driven by the data-config `corr_polbasis = [...]` list.

corpol(::CirBasis) = PolBasis{YPol, XPol}()

function corr_polbasis(dcoh, site::Symbol)
    dt = map(datatable(dcoh.config)) do row
        bl = row.sites
        pb = row.polbasis
        if bl[1] == site
            row2 = @set row.polbasis = (corpol(pb[1]), pb[2])
        elseif bl[2] == site
            row2 = @set row.polbasis = (pb[1], corpol(pb[2]))
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

ConstructionBase.constructorof(::Type{<:Comrade.EHTObservationTable{T}}) where {T} = Comrade.EHTObservationTable{T}
ConstructionBase.constructorof(::Type{<:Comrade.EHTArrayConfiguration{T}}) where {T} = Comrade.EHTArrayConfiguration
