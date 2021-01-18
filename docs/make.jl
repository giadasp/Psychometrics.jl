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
        hide("lib/public.md", [
            "methods/exported/bayesian.md",
            "methods/exported/distributions.md",
            "methods/exported/examinee.md",
            "methods/exported/information.md",
            "methods/exported/likelihood.md",
            "methods/exported/probability.md",
            "methods/exported/response.md",
            "methods/exported/item.md"
        ]),
        hide("lib/internals.md", [
            "methods/internals/bayesian.md",
            "methods/internals/distributions.md",
            "methods/internals/information.md",
            "methods/internals/latent.md",
            "methods/internals/likelihood.md",
            "methods/internals/probability.md",
            "methods/internals/response.md",
            "methods/internals/parameters.md"
        ])
    ]
    
)

deploydocs(repo = ENV["REPO"], devurl = "docs", devbranch = ENV["DEVBRANCH"])
