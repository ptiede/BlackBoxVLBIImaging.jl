using BlackBoxVLBIImaging
using Documenter

DocMeta.setdocmeta!(BlackBoxVLBIImaging, :DocTestSetup, :(using BlackBoxVLBIImaging); recursive = true)

makedocs(;
    modules = [BlackBoxVLBIImaging],
    authors = "Paul Tiede <ptiede91@gmail.com> and contributors",
    sitename = "BlackBoxVLBIImaging.jl",
    format = Documenter.HTML(;
        canonical = "https://ptiede.github.io/BlackBoxVLBIImaging.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/ptiede/BlackBoxVLBIImaging.jl",
    devbranch = "main",
)
