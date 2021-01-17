# Latent Variable (Internals)

```@meta
    CurrentModule = Psychometrics
```
```@docs
    _empty_chain!(latent::Latent1D)
    _set_val!(latent::Latent1D, val::Float64)
    _set_val_from_chain!(latent::Latent1D)
    _update_estimate!(latent::Latent1D; sampling = true)
    _chain_append!(latent::Latent1D; sampling = false)
```
