using Documenter
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using Psychometrics

makedocs(
    sitename = "Psychometrics",
    format = Documenter.HTML(),
    modules = [Psychometrics],
    doctest = true,
    
    pages = [
        "index.md",
        hide("lib/public.md", map(
            s -> "src/methods/exported/$(s)",
            sort(readdir(joinpath(@__DIR__, "src/methods/exported")))
        )),
        hide("lib/internals.md", map(
            s -> "src/methods/internals/$(s)",
            sort(readdir(joinpath(@__DIR__, "src/methods/internals")))
        ))
    ]
)

deploydocs(repo = ENV["REPO"], devurl = "docs", devbranch = ENV["DEVBRANCH"])
