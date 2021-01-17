# Likelihood

```@meta
    CurrentModule = Psychometrics
```

## Values in Matrix form

```@docs
log_likelihood(responses::Matrix{Float64}, latents_matrix::Matrix{Float64}, parameters_matrix::Matrix{Float64}, design::Matrix{Float64})
```

## Using Structs

```@docs
log_likelihood(
    response_val::Float64,
    latent_val::Float64,
    parameters::AbstractParametersBinary,
)
log_likelihood(
    response::AbstractResponse,
    examinee::AbstractExaminee,
    item::AbstractItem,
)
log_likelihood(
    response::AbstractResponse,
    examinee::AbstractExaminee,
    item::AbstractItem,
    g_item::Vector{Float64},
    g_latent::Vector{Float64},
)
log_likelihood(
    responses::Vector{<:AbstractResponse},
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
)
log_likelihood(
    responses::Vector{<:AbstractResponse},
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
    g_item::Vector{Float64},
    g_latent::Vector{Float64},
)
likelihood(
    response::AbstractResponse,
    examinee::AbstractExaminee,
    item::AbstractItem,
)
likelihood(
    responses::Vector{<:AbstractResponse},
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
)    
```
