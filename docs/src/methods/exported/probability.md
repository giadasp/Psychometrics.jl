# Probability (Item Characteristic Function)

```@meta
    CurrentModule = Psychometrics
```

## Input Values in Matrix form

```@docs
probability(parameters_matrix::Matrix{Float64}, latents_matrix::Matrix{Float64})
probability_3PL(
    parameters_matrix::Matrix{Float64},
    latents_matrix::Matrix{Float64},
)
```

## Using Structs

```@docs
probability(examinee::AbstractExaminee, item::AbstractItem)
probability(items::Vector{<:AbstractItem}, examinees::Vector{<:AbstractExaminee})
probability(examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})
```
