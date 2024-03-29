```@meta
CurrentModule = Psychometrics
DocTestSetup = quote
    using Psychometrics
end
```

# Public Documentation

Documentation for `Psychometrics.jl`'s public interface.

See the [Internals Documentation](@ref) Section of the manual for internal package docs covering unexported functions.

## Index

```@contents
Pages = ["public.md"]
```

## Structs and Abstracts

```@autodocs
Modules = [Psychometrics]
Public = false
Order = [:type]
```

Each of the mentioned structs has a random default factory, callable by using the name of the struct followed by `()`.

_Example: `Parameters2PL()` generates a set of 2PL item parameters from the product distribution (independent bivariate)._

```julia
Distributions.Product([
   Distributions.LogNormal(0, 0.25),
   Distributions.Normal(0, 1)
])
```

The immutable structs `Item` and `Examinee` need the identification fields `id` and `idx` to be randomly generated. 

_Example:_
```@example 
Item(1, "item_1")
```
_Generates an operational item with an empty `content` description, and a default 1PL item parameter._

## Exported Methods

```@contents
Pages = [
    "../methods/exported/item.md",
    "../methods/exported/examinee.md",
    "../methods/exported/response.md",
    "../methods/exported/probability.md",
    "../methods/exported/information.md",
    "../methods/exported/likelihood.md",
    "../methods/exported/assessment.md",
    "../methods/exported/calibration.md",
    "../methods/exported/joint_estimation.md",
    "../methods/exported/online.md",
    "../methods/exported/distributions.md",
]
Depth = 1
```
