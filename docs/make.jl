using Documenter
using Pkg
Pkg.activate(".") 
Pkg.instantiate()
using Psychometrics

makedocs(
    sitename="Psychometrics",
    format=Documenter.HTML(),
    modules=[Psychometrics],
    doctest=true,
    pages = [
        "index.md",
        "item.md",
        "parameter.md",
        "examiness.md",
        "latent.md",
        "response.md",
        "probability.md",
        "information.md",
        "likelihood.md"
         ]
)

deploydocs(
    repo = "github.com/giadasp/Psychometrics.jl.git",
    devurl = "docs"
)