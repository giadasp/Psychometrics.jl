# Likelihood (Internals)

```@meta
    CurrentModule = Psychometrics
```

```@docs
__likelihood(
    response_val::Float64,
    latent_val::Float64,
    parameters::AbstractParametersBinary;
    weight::Float64 = 1.0
)
__log_likelihood(
    response_val::Float64,
    latent_val::Float64,
    parameters::AbstractParametersBinary,
)
_likelihood(
    response_val::Float64,
    latent_val::Float64,
    item::AbstractItem;
    weight::Float64 = 1.0
)
_likelihood(
    response::AbstractResponse,
    latent_val::Float64,
    item::AbstractItem;
    weight::Float64 = 1.0
)
_log_likelihood(
    response_val::Float64,
    latent::Latent1D,
    parameters::AbstractParametersBinary,
)
_log_likelihood(
    response_val::Float64,
    latent::Latent1D,
    parameters::AbstractParametersBinary,
    g_item::Vector{Float64},
    g_latent::Vector{Float64},
)
```
