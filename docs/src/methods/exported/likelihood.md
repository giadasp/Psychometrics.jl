# Likelihood

```@meta
    CurrentModule = Psychometrics
```

```@docs
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
