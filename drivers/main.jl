using Pkg
Pkg.activate(@__DIR__)

using BlackBoxVLBIImaging

const USAGE = """
Image a VLBI observation, configured entirely from four TOML files.

The sampler (AdvancedHMC vs Reactant NUTS) is part of the fitting TOML, since selecting
Reactant also changes the optimizer path.

Usage: julia main.jl --image <path> --instrument <path> --data <path> --fitting <path>
                     [--outpath <path>] [--restart]

Options (`--name value` or `--name=value`):

  --image <path>       image/sky-model TOML (grid, mean model, flux, random-field order)
  --instrument <path>  instrument-model TOML (gain/leakage scheme + per-parameter priors)
  --data <path>        data TOML (file/array/averaging/noise + flag table)
  --fitting <path>     fitting-strategy TOML (optimizer, tempering schedule, sampler, Reactant)
  --outpath <path>     output base path for the run (default `Runs/run`)

Flags:

  --restart            resume the run from a previously serialized optimum at `--outpath`
                       instead of re-optimizing. This is a one-off action, deliberately not
                       stored in any TOML.
  -h, --help           print this message
"""

const REQUIRED = ("image", "instrument", "data", "fitting")
const OPTIONS = (REQUIRED..., "outpath")

function parse_cli(args)
    opts = Dict{String, String}()
    restart = false
    i = firstindex(args)
    while i <= lastindex(args)
        arg = args[i]
        if arg in ("-h", "--help")
            print(USAGE)
            exit(0)
        elseif arg == "--restart"
            restart = true
        elseif startswith(arg, "--")
            name, eq, value = partition(arg[3:end])
            name in OPTIONS || error("unknown option --$name\n\n$USAGE")
            if !eq
                i < lastindex(args) || error("option --$name needs a value")
                i += 1
                value = args[i]
            end
            haskey(opts, name) && error("option --$name given twice")
            opts[name] = value
        else
            error("unexpected argument '$arg'\n\n$USAGE")
        end
        i += 1
    end
    missing_opts = [n for n in REQUIRED if !haskey(opts, n)]
    isempty(missing_opts) ||
        error("missing required option(s): " * join("--" .* missing_opts, ", ") * "\n\n$USAGE")
    return opts, restart
end

# Split `name=value` into (name, true, value); a bare `name` gives (name, false, "").
function partition(s)
    k = findfirst('=', s)
    isnothing(k) && return s, false, ""
    return s[1:prevind(s, k)], true, s[nextind(s, k):end]
end

function main(args)
    opts, restart = parse_cli(args)
    return image_from_toml(
        opts["image"], opts["instrument"], opts["data"], opts["fitting"];
        outpath = get(opts, "outpath", "Runs/run"), restart
    )
end

main(ARGS)
