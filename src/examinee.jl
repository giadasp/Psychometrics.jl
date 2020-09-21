abstract type AbstractExaminee end

mutable struct Examinee1D <: AbstractExaminee
    idx::Int64
    id::String
    latent::Latent1D
    Examinee1D(idx,id,latent) = new(idx,id,latent)
end


"""
    answer(examinee::AbstractExaminee, item::AbstractParameters)

Randomly generate a response by `examinee` to `item`.
"""
function answer(examinee::AbstractExaminee, item::AbstractParameters)
    _generate_response(examinee.latent, item.parameters)
end

"""
    answer(examinee_idx::Int64, item_idx::Int64, examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})

Randomly generate a response by `examinee` with index `examinee_idx` to `item` with index `item_idx`.
"""
function answer(examinee_idx::Int64, item_idx::Int64, examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})
    answer(get_examinee_by_idx!(examinee_idx, examinees), get_item_by_idx!(item_idx, items))
end

"""
    answer(examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})

Randomly generate responses by all the examinees in `examinees` to items in `items`.
"""
function answer(examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})
    map((e, i) -> answer(e, i), examinees, items)
end

"""
    get_examinee_by_idx!(examinee_idx::Int64, examinees::Vector{<:AbstractExaminee})

It returns the examinee with index `examinee_idx` from a vector of ::AbstractExaminee.
"""
function get_examinee_by_idx(examinee_idx::Int64, examinees::Vector{<:AbstractItem})
   filter(e -> e.idx == examinee_idx, examinees)
end