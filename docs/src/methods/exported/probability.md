# Probability (Item Characteristic Function)

```@meta
    Modules = Psychometrics
```

## Values in Matrix form

```@docs
    probability(parameters_matrix::Matrix{Float64}, latent_matrix::Matrix{Float64})
    probability_3PL(parameters_matrix::Matrix{Float64}, latent_matrix::Matrix{Float64})
```

## Using Structs

```@docs
    probability(examinee::AbstractExaminee, item::AbstractItem)
    probability(examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})
    probability(items::Vector{<:AbstractItem}, examinees::Vector{<:AbstractExaminee})
```
