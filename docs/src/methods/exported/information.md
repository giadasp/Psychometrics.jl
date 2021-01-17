# Information (Item Information Function)

```@meta
    CurrentModule = Psychometrics
```

## Information wrt $\theta$

### Values in Matrix form

```@docs
    information_latent(latents_matrix::Matrix{Float64}, parameters_matrix::Matrix{Float64})
    information_latent_3PL(latents_matrix::Matrix{Float64}, parameters_matrix::Matrix{Float64})
```

### Using Structs

```@docs
    information_latent(examinee::AbstractExaminee, item::AbstractItem)
    information_latent(examinee::AbstractExaminee, items::Vector{<:AbstractItem})
    information_latent(examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})
```

## Information wrt Item Parameters

```@docs
    expected_information_item(items::Vector{<:AbstractItem}, examinees::Vector{<:AbstractExaminee})
    observed_information_item(items::Vector{<:AbstractItem}, examinees::Vector{<:AbstractExaminee}, responses::Vector{<:AbstractResponse})
```
