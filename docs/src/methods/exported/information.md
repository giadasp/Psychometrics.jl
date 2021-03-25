# Information (Item Information Function)

```@meta
    CurrentModule = Psychometrics
```

## Information wrt $\theta$

```@docs
latent_information(examinee::AbstractExaminee, item::AbstractItem)
latent_information(examinee::AbstractExaminee, items::Vector{<:AbstractItem})
latent_information(
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
)
latent_information(
    latents_matrix::Matrix{Float64},
    parameters_matrix::Matrix{Float64},
)
latent_information_3PL(
    latents_matrix::Matrix{Float64},
    parameters_matrix::Matrix{Float64},
)
```

## Information wrt Item Parameters

```@docs
item_expected_information(
    item::AbstractItem,
    examinee::AbstractExaminee,
)
item_expected_information(
    items::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
)
item_observed_information(
    item::AbstractItem,
    examinee::AbstractExaminee,
    response::AbstractResponse,
)
item_observed_information(
    items::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{<:AbstractResponse},
)
```
