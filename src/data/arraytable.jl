# Reading the array geometry / feed table used by the dlist reader. The array file is a
# space-delimited table: site X Y Z SEFD1 SEFD2 fr_parallactic fr_elevation fr_offset(deg).

function get_polbasis(p1, p2)
    up1 = Set(unique(p1))
    up2 = Set(unique(p2))
    pb1 = (up1 == Set(("X", "Y")) ? LinBasis() : CirBasis())
    pb2 = (up2 == Set(("X", "Y")) ? LinBasis() : CirBasis())
    return pb1, pb2
end

function read_array_table(fname)
    tbl = CSV.read(fname, DataFrame; delim = " ", ignorerepeated = true)
    return StructArray(
        sites = Symbol.(tbl[:, 1]),
        X = tbl[:, 2],
        Y = tbl[:, 3],
        Z = tbl[:, 4],
        SEFD1 = tbl[:, 5],
        SEFD2 = tbl[:, 6],
        fr_parallactic = tbl[:, 7],
        fr_elevation = tbl[:, 8],
        fr_offset = deg2rad.(tbl[:, 9]),
    )
end
