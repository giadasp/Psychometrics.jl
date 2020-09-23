using Documenter
using Pkg
Pkg.activate(".") 
Pkg.instantiate()
using Psychometrics

makedocs(
    sitename="Psychometrics",
    format=Documenter.HTML(),
    modules=[Psychometrics],
    doctest=true
)

deploydocs(
    repo = "github.com/giadasp/Psychometrics.jl.git",
    devurl = "docs"
)