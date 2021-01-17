# Responses

```@meta
    CurrentModule = Psychometrics
```

```@docs
get_examinees_by_item_id(item_id::String, responses::Vector{<:AbstractResponse},  examinees::Vector{<:AbstractExaminee})
get_items_by_examinee_id(examinee_id::String, responses::Vector{<:AbstractItem})
add_response!(response::AbstractResponse, responses::Vector{<:AbstractResponse})
get_responses_by_examinee_id(examinee_id::String, responses::Vector{<:AbstractResponse})
get_responses_by_item_id(item_id::String, responses::Vector{<:AbstractResponse})
get_responses_by_item_idx(item_idx::Int64, responses::Vector{<:AbstractResponse}; sorted = true)
answer(examinee::AbstractExaminee, item::AbstractItem)
answer(examinee::AbstractExaminee, items::Vector{<:AbstractItem})
answer(examinee_id::String, item_id::String, examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})
answer(examinees:Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})
get_design_matrix(responses::Vector{Response}, I::Int64, N::Int64)
get_response_matrix(responses::Vector{Response}, I::Int64, N::Int64)
get_responses(response_matrix::Matrix{Float64}, design_matrix::Matrix{Float64}, items::Vector{<:AbstractItem}, examinees::Vector{<:AbstractExaminee})
get_items_idx_answered_by_examinee(
    examinee::AbstractExaminee,
    responses::Vector{<:AbstractResponse},
)
get_examinees_idx_who_answered_item(
    item::AbstractItem,
    responses::Vector{<:AbstractResponse},
)
```