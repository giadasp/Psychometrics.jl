abstract type AbstractResponse end

mutable struct Response <: AbstractResponse
    item_idx::Int64
    examinee_idx::Int64
    val::Union{Missing,Float64}
    start_time::Dates.DateTime
    end_time::Dates.DateTime
    Response(item_idx, examinee_idx, val, start_time, end_time) = new(item_idx, examinee_idx, val, start_time, end_time)
    Response(item_idx, examinee_idx, val, time::Dates.DateTime) = new(item_idx, examinee_idx, val, time, time)
end

# Outer Constructor Methods

"""
    get_examinees(item_idx::Int64, responses::Vector{<:AbstractResponse}, examinees::Dict{Int64,<:AbstractExaminee})

It returns the examinees who answered to the item with index `item_idx`.
"""
function get_examinees(item_idx::Int64, responses::Vector{<:AbstractResponse}, examinees::Dict{Int64,<:AbstractExaminee})
    examinees[filter(r -> r.item_idx == item_idx, responses).examinee_idx]
end

"""
    get_items(examinee_idx::Int64, responses::Dict{Int64,<:AbstractItem})

It returns the items answered by the examinee with index `examinee_idx`.
"""
function get_items(examinee_idx::Int64, responses::Vector{<:AbstractResponse}, items::Dict{Int64,<:AbstractItem})
        items[filter(r -> r.examinee_idx == examinee_idx, responses).item_idx]
end

"""
    add_response!(response::AbstractResponse, responses::Vector{Response})

Push the response in the response vector `responses`.
"""
function add_response!(response::AbstractResponse, responses::Vector{Response})
    push!(response, responses)
end

"""
    get_examinee_responses(idx::Int64, responses::Vector{Response})

It returns the vector of responses given by examinee with index `idx`.
"""
function get_examinee_responses(idx::Int64, responses::Vector{Response})
    filter(r -> r.examinee_idx == idx, responses)
end

"""
    get_item_responses(idx::Int64, responses::Vector{Response})

It returns the vector of responses given to item with index `idx`.
"""
function get_item_responses(idx::Int64, responses::Vector{Response})
    filter(r -> r.item_idx == idx, responses)
end

"""
    generate_response(latent::Latent1D, parameters::AbstractParameters)

Randomly generate a response for a 1-dimensional latent variable and custom item parameters.
"""
function generate_response(latent::Latent1D, parameters::AbstractParameters)
    Float64(rand(Distributions.Bernoulli(probability(latent, parameters, Array{Float64,1}(undef,0), Array{Float64,1}(undef,0)))))::Float64
end


"""
    generate_response(examinee::Dict{Int64,<:AbstractExaminee}, item::Dict{Int64,<:AbstractItem})

Randomly generate a response by `examinee` to `item`.
"""
function generate_response(examinees::Dict{Int64,<:AbstractExaminee}, items::Dict{Int64,<:AbstractItem})
    vcat([[Response(i_key, e_key, generate_response(e.latent, i.parameters), Dates.now()) for (e_key, e) in examinees] for (i_key, i) in items]...)
end


"""
    generate_response(examinees::Dict{Int64,<:AbstractExaminee}, items::Dict{Int64,<:AbstractItem}, design::Dict{Int64,Vector{Int64}})

Randomly generate a response by `examinee` to `item` under a `design` dictionary of the same size of examinees with indices of items which must be answered by each examinee.
"""
function generate_response(examinees::Dict{Int64,<:AbstractExaminee}, items::Dict{Int64,<:AbstractItem}, design::Dict{Int64,Vector{Int64}})
    vcat([[Response(i_key, e_key, generate_response(e.latent, i.parameters), Dates.now()) for (e_key, e) in examinees] for (i_key, i) in items]...)
end
