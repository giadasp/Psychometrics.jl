# Public Documentation

Documentation for `Psychometrics.jl`'s public interface.

See the [Internals Documentation](@ref) Section of the manual for internal package docs covering unexported functions.

## Index

```@contents
Pages = ["public.md"]
```

## Abstracts

```@docs
AbstractItem
AbstractExaminee
AbstractParameters
AbstractParametersBinary
AbstractLatent
AbstractResponse
```

## Structs and Abstracts

```@docs
Item
Examinee
Parameters1PL
Parameters2PL
Parameters3PL
Latent1D
LatentND
Response
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

_Example: `Item(1, "item_1")` generates an operational item with an empty `content` description, and a default 1PL item parameter._

## Exported Methods

```@contents
Pages = [
    "../methods/exported/item.md",
    "../methods/exported/examinee.md",
    "../methods/internals/response.md",
    "../methods/exported/probability.md",
    "../methods/exported/information.md",
    "../methods/exported/likelihood.md",
]
Depth = 1
```
