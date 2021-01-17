# Probability (Item Characteristic Function) (Internals)

```@meta
    Modules = Psychometrics
```

## 1PL

```@docs
    __probability(latent_val::Float64, parameters::Parameters1PL)
    _probability(latent::Latent1D, parameters::Parameters1PL)
    _probability(latent::Latent1D, parameters::Parameters1PL, g_item::Vector{Float64}, g_latent::Vector{Float64})
```

## 2PL

```@docs
    __probability(latent_val::Float64, parameters::Parameters2PL)
    _probability(latent::Latent1D, parameters::Parameters2PL)
    _probability(latent::Latent1D, parameters::Parameters2PL, g_item::Vector{Float64}, g_latent::Vector{Float64})
```

## 3PL
    __probability(latent_val::Float64, parameters::Parameters3PL)
    _probability(latent::Latent1D, parameters::Parameters3PL)
    _probability(latent::Latent1D, parameters::Parameters3PL,  g_item::Vector{Float64}, g_latent::Vector{Float64})
    __probability(latent_vals::Vector{Float64}, parameters::Parameters3PL)
    _probability(latent::Latent1D, parameters::Parameters3PL,  g_item::Vector{Float64}, g_latent::Vector{Float64})
```

