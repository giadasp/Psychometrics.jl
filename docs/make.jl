using Documenter
using Pkg
Pkg.activate(".") 
Pkg.instantiate()
using Psychometrics

makedocs(
    sitename="Psychometrics",
    format=Documenter.HTML(),
    modules=[ATA],
    doctest=true,
    pages = [
        "index.md",
        "utils.md",
        "build.md",
        "opt.md",
        "print.md",
        "examples.md"
    ]
)

deploydocs(
    repo = "github.com/giadasp/Psychometrics.jl.git",
    devurl = "docs"
)