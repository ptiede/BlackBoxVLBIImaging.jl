# Bespoke parser for HOPS `.dlist` mixed-polarization CSV files. Builds a
# `Comrade.EHTObservationTable` of 2x2 coherency Jones matrices and their noise.
#
# NOTE: dlist files carry no source/pointing metadata, so ra/dec/mjd/source/bandwidth are
# faked here (they do not affect the likelihood for a fixed observation).

function read_dlist(fname, arrayname)
    baseline, coh, noise = getcohfield(fname)
    ra = 180.0
    dec = 90.0
    mjd = 57718
    source = :M87
    bw = 8.0e9
    tarr = read_array_table(arrayname)
    # I don't have access to scantable so fake it
    ts = unique(baseline.Ti)
    dt = minimum(diff(ts))
    start = ts .- dt / 2
    stop = ts .+ dt / 2
    sc = Table((; start, stop))
    ac = Comrade.EHTArrayConfiguration(bw, tarr, sc, mjd, ra, dec, source, :UTC, baseline)
    T = Comrade.EHTCoherencyDatum{eltype(real(coh[1])), eltype(ac.datatable), eltype(coh), eltype(noise)}
    return Comrade.EHTObservationTable{T}(coh, noise, ac)
end

function getcohfield(fname)
    df = CSV.read(fname, DataFrame; comment = "-", delim = " ", ignorerepeated = true, skipto = 3, header = 0)
    rename!(df, [:F, :T, :S1, :S2, :Pol1, :Pol2, :U, :V, :el1, :el2, :parang1, :parang2, :visre, :visim, :sigma])
    # group by the freq, time, and stations
    dfg = DataFrames.groupby(df, [:F, :T, :S1, :S2])
    U = Float64[]
    V = Float64[]

    el1 = Float64[]
    el2 = Float64[]
    par_ang1 = Float64[]
    par_ang2 = Float64[]

    C = SMatrix{2, 2, ComplexF64, 4}[]
    S = SMatrix{2, 2, Float64, 4}[]
    Ti = Float64[]
    Fr = Float64[]
    si = Tuple{Symbol, Symbol}[]

    PB = [get_polbasis(g.Pol1, g.Pol2) for g in dfg]

    for g in dfg
        push!(U, g.U[1])
        push!(V, g.V[1])
        push!(Ti, g.T[1])
        push!(Fr, g.F[1] * 1.0e9)

        push!(el1, g.el1[1])
        push!(el2, g.el2[1])
        push!(par_ang1, g.parang1[1])
        push!(par_ang2, g.parang2[1])

        push!(si, (Symbol(g.S1[1]), Symbol(g.S2[1])))
        s = MMatrix{2, 2, Float64, 4}(NaN, NaN, NaN, NaN)
        c = MMatrix{2, 2, ComplexF64, 4}(NaN, NaN, NaN, NaN)
        cind11 = findall(x -> (((x[1] == "R") || (x[1] == "X")) && ((x[2] == "R") || (x[2] == "X"))), zip(g.Pol1, g.Pol2) |> collect)
        cind21 = findall(x -> (((x[1] == "L") || (x[1] == "Y")) && ((x[2] == "R") || (x[2] == "X"))), zip(g.Pol1, g.Pol2) |> collect)
        cind12 = findall(x -> (((x[1] == "R") || (x[1] == "X")) && ((x[2] == "L") || (x[2] == "Y"))), zip(g.Pol1, g.Pol2) |> collect)
        cind22 = findall(x -> (((x[1] == "L") || (x[1] == "Y")) && ((x[2] == "L") || (x[2] == "Y"))), zip(g.Pol1, g.Pol2) |> collect)

        @assert length(cind11) <= 1 && length(cind21) <= 1 && length(cind12) <= 1 && length(cind22) <= 1 "Multiple entries for the same baseline and time. This parser assumes each line in the dlist is a unique baseline+time+freq."

        c[1, 1] = length(cind11) == 0 ? NaN : complex(g.visre[first(cind11)], g.visim[first(cind11)])
        c[2, 1] = length(cind21) == 0 ? NaN : complex(g.visre[first(cind21)], g.visim[first(cind21)])
        c[1, 2] = length(cind12) == 0 ? NaN : complex(g.visre[first(cind12)], g.visim[first(cind12)])
        c[2, 2] = length(cind22) == 0 ? NaN : complex(g.visre[first(cind22)], g.visim[first(cind22)])

        s[1, 1] = length(cind11) == 0 ? NaN : g.sigma[first(cind11)]
        s[2, 1] = length(cind21) == 0 ? NaN : g.sigma[first(cind21)]
        s[1, 2] = length(cind12) == 0 ? NaN : g.sigma[first(cind12)]
        s[2, 2] = length(cind22) == 0 ? NaN : g.sigma[first(cind22)]

        push!(C, SMatrix(c))
        push!(S, SMatrix(s))
    end

    Cstr = StructArray(C)
    Sstr = StructArray(S)

    data = StructArray{Comrade.EHTArrayBaselineDatum{eltype(U), <:eltype(PB), eltype(el1)}}(
        (; U, V, Ti, Fr, sites = si, polbasis = PB, elevation = StructArray((el1, el2)), parallactic = StructArray((par_ang1, par_ang2)))
    )

    return data, StructArray(Cstr), StructArray(Sstr)
end
