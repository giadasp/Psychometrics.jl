# Public Documentation

Documentation for `Psychometrics.jl`'s public interface.

See the [Internals Documentation](@ref) Section of the manual for internal package docs covering unexported functions.

## Contents

```@contents
Pages = ["public.md"]
```

## Index

```@index
Pages = ["public.md"]
```

## Structs

```@docs
AbstractItem
Item <: AbstractItem
AbstractExaminee
Examinee <: AbstractExaminee
AbstractParameters
AbstractParametersBinary <: AbstractParameters
Parameters1PL <: AbstractParametersBinary
Parameters2PL <: AbstractParametersBinary
Parameters3PL <: AbstractParametersBinary
AbstractLatent
Latent1D <: AbstractLatent
LatentND <: AbstractLatent
AbstractResponse
ResponseBinary <: AbstractResponse
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

_Example: `Item(1, "item_1")` generates an operational item with an empty `content`_description, and a default 1PL item parameter._

## Exported Methods

```@contents
Pages = [
    "methods/exported/item.md",
    "methods/exported/examinee.md",
    "methods/exported/probability.md",
    "methods/exported/information.md",
    "methods/exported/likelihood.md",
]
Depth = 1
```
