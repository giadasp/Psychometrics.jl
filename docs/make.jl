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
        hide("methods/exported/bayesian.md"),
        hide("methods/exported/distributions.md"),
        hide("methods/exported/examinee.md"),
        hide("methods/exported/information.md"),
        hide("methods/exported/likelihood.md"),
        hide("methods/exported/probability.md"),
        hide("methods/exported/response.md"),
        hide("methods/internals/item.md"),
        hide("methods/internals/bayesian.md"),
        hide("methods/internals/distributions.md"),
        hide("methods/internals/examinee.md"),
        hide("methods/internals/information.md"),
        hide("methods/internals/likelihood.md"),
        hide("methods/internals/probability.md"),
        hide("methods/internals/response.md"),
        hide("methods/internals/item.md"),

    ]
)

deploydocs(repo = ENV["REPO"], devurl = "docs", devbranch = ENV["DEVBRANCH"])
