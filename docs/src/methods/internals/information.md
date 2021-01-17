# Information (Item Information Function) (Internals)

```@meta
    CurrentModule = Psychometrics
```

## Information wrt $\theta$

```@docs
    _information_latent(latent::Latent1D, parameters::Parameters1PL)
    _information_latent(latent::Latent1D, parameters::Parameters2PL)
    _information_latent(latent::Latent1D, parameters::Parameters3PL)
```

## Information wrt Item Parameters

```@docs
_expected_information_item(parameters::Parameters1PL, latent::Latent1D)
_expected_information_item(parameters::Parameters2PL, latent::Latent1D)
_expected_information_item(parameters::Parameters3PL, latent::Latent1D)
_observed_information_item(
    parameters::Parameters3PL,
    latent::Latent1D,
    response_val::Float64,
)
```
