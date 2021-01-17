# Item Parameters (Internals)

```@meta
    CurrentModule = Psychometrics
```
```@docs
    _add_prior!(parameters::AbstractParameters, prior::Distributions.Distribution) 
    _add_prior!(parameters::AbstractParameters, priors::Vector{Distributions.UnivariateDistribution})
    _add_posterior!(parameters::AbstractParameters, posterior::Distributions.Distribution) 
    _add_posterior!(parameters::AbstractParameters, priors::Vector{Distributions.UnivariateDistribution}) 
    _empty_chain!(parameters::Parameters1PL)
    _empty_chain!(parameters::Parameters2PL)
    _empty_chain!(parameters::Parameters3PL)
    _chain_append!(parameters::Union{Parameters2PL,Parameters3PL}; sampling = false)
    _set_val!(parameters::Parameters2PL, vals::Vector{Float64})
    _set_val_from_chain!(parameters::Parameters2PL)
    _update_estimate!(parameters::Parameters2PL; sampling = true)
```
