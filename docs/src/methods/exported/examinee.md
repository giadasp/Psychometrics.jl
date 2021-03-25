# Examinees

```@meta
CurrentModule = Psychometrics
```

## Struct

```@docs
Examinee
```

## Methods for a Single Examinee

```@docs
get_examinee_by_id(examinee_id::String, examinees::Vector{<:AbstractExaminee})
empty_chain!(examinee::AbstractExaminee)
set_val!(examinee::AbstractExaminee, val::Float64)
set_val_from_chain!(examinee::AbstractExaminee)
update_estimate!(examinee::AbstractExaminee; sampling = true)
chain_append!(examinee::AbstractExaminee; sampling = false)
set_prior!(
    examinee::AbstractExaminee,
    prior::Union{Distributions.DiscreteUnivariateDistribution, Distributions.ContinuousUnivariateDistribution}
)
get_latents(examinee::AbstractExaminee)
get_latents_vals(examinee::AbstractExaminee)
```


## Methods for a Vector of Examinees

```@docs
get_latents(examinees::Vector{<:AbstractExaminee})
get_latents_vals(examinees::Vector{<:AbstractExaminee})
set_prior!(
    examinees::Vector{<:AbstractExaminee},
    prior::Union{Distributions.DiscreteUnivariateDistribution, Distributions.ContinuousUnivariateDistribution}
)
```

