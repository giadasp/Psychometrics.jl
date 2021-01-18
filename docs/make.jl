using Documenter
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using Psychometrics

makedocs(
    sitename = "Psychometrics",
    format = Documenter.HTML(),
    modules = [Psychometrics],
    doctest = false,
    
    pages = [
        "index.md",
        "lib/public.md",
        "lib/internals.md"   ]
    
)

#deploydocs(repo = ENV["REPO"], devurl = "docs", devbranch = ENV["DEVBRANCH"])
#deploydocs(repo = ENV["REPO"], devurl = "docs", devbranch = ENV["DEVBRANCH"])