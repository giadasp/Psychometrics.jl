# Information (Item Information Function) (Internals)

```@meta
    CurrentModule = Psychometrics
```

## Information wrt $\theta$

```@docs
    _latent_information(latent::Latent1D, parameters::Parameters1PL)
    _latent_information(latent::Latent1D, parameters::Parameters2PL)
    _latent_information(latent::Latent1D, parameters::Parameters3PL)
```

## Information wrt Item Parameters

```@docs
_item_expected_information(parameters::Parameters1PL, latent::Latent1D)
_item_expected_information(parameters::Parameters2PL, latent::Latent1D)
_item_expected_information(parameters::Parameters3PL, latent::Latent1D)
_item_observed_information(
    parameters::Parameters3PL,
    latent::Latent1D,
    response_val::Float64,
)
```
