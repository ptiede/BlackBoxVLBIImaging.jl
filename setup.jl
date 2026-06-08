using Pkg; Pkg.activate(@__DIR__)
@info "Instantiating the BlackBoxVLBIImaging environment (this may take a moment)..."
Pkg.instantiate()

Pkg.activate(joinpath(@__DIR__, "drivers"))
@info "Instantiating the drivers environment (this may take a moment)..."
Pkg.instantiate()
