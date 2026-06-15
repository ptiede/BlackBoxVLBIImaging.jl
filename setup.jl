using Pkg
using TOML

# Reactant downloads a CPU-only, CUDA, or ROCm build of Reactant_jll at instantiate time,
# chosen by a Preferences entry stored under the Reactant_jll UUID. On machines without a
# usable GPU toolchain (or where the GPU artifacts simply can't be installed) we force the
# CPU-only build by setting `gpu = "none"` BEFORE instantiating, so Pkg never tries to fetch
# the GPU artifacts.
const REACTANT_JLL_UUID = "0192cb87-2b54-54ad-80e0-3be72ad8a3c0"

# Decide what to do with the Reactant GPU preference based on CLI flags / env vars:
#   :disable  -- force CPU-only (`--no-gpu`/`--cpu`, or BLACKBOX_NO_GPU truthy)
#   :reset    -- clear the pinned preference so Reactant auto-detects again (`--gpu`/`--enable-gpu`)
#   :auto     -- leave any existing preference untouched (default)
# `--no-gpu` is sticky: once written it stays in LocalPreferences.toml until `--gpu` clears it.
function gpu_mode()
    disable = any(a -> a in ("--no-gpu", "--cpu"), ARGS) ||
              lowercase(get(ENV, "BLACKBOX_NO_GPU", "")) in ("1", "true", "yes", "on")
    reset = any(a -> a in ("--gpu", "--enable-gpu"), ARGS)
    disable && reset && error("Conflicting flags: pass either --no-gpu/--cpu or --gpu, not both.")
    disable && return :disable
    reset && return :reset
    return :auto
end

# Read/update the active environment's LocalPreferences.toml, preserving everything else.
function update_reactant_pref!(mode::Symbol)
    prefs_path = joinpath(dirname(Base.active_project()), "LocalPreferences.toml")
    prefs = isfile(prefs_path) ? TOML.parsefile(prefs_path) : Dict{String,Any}()
    if mode === :disable
        section = get!(() -> Dict{String,Any}(), prefs, "Reactant_jll")
        section["gpu"] = "none"
        open(io -> TOML.print(io, prefs), prefs_path, "w")
        @info "GPU support disabled: set Reactant_jll gpu=\"none\" in $prefs_path"
    elseif mode === :reset
        section = get(prefs, "Reactant_jll", nothing)
        section isa AbstractDict && delete!(section, "gpu")
        # Drop the section entirely if it became empty, and the file if nothing is left.
        section isa AbstractDict && isempty(section) && delete!(prefs, "Reactant_jll")
        if isempty(prefs)
            isfile(prefs_path) && rm(prefs_path)
        else
            open(io -> TOML.print(io, prefs), prefs_path, "w")
        end
        @info "GPU preference reset: Reactant will auto-detect a GPU again ($prefs_path)"
    end
end

const GPU_MODE = gpu_mode()
if GPU_MODE === :disable
    @info "Setting up in CPU-only mode (no GPU/Reactant GPU artifacts will be installed)."
elseif GPU_MODE === :reset
    @info "Re-enabling GPU auto-detection (clearing any pinned CPU-only preference)."
end

Pkg.activate(@__DIR__)
update_reactant_pref!(GPU_MODE)
@info "Instantiating the BlackBoxVLBIImaging environment (this may take a moment)..."
Pkg.instantiate()

Pkg.activate(joinpath(@__DIR__, "drivers"))
update_reactant_pref!(GPU_MODE)
@info "Instantiating the drivers environment (this may take a moment)..."
Pkg.instantiate()
