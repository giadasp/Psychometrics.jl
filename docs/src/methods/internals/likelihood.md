# Likelihood (Internals)

```@meta
    CurrentModule = Psychometrics
```

## Using Structs

```@docs
    _likelihood(response_val::Float64, latent_val::Float64, parameters::AbstractParametersBinary)
    _log_likelihood(response_val::Float64, latent::Latent1D, parameters::AbstractParametersBinary)
    _log_likelihood(response_val::Float64, latent::Latent1D, parameters::AbstractParametersBinary, g_item::Vector{Float64}, g_latent::Vector{Float64})
    _likelihood(response_val::Float64, latent::Latent1D, parameters::AbstractParametersBinary)
```
