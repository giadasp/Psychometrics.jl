abstract type AbstractResponse end

mutable struct Response <: AbstractResponse
    item::AbstractItem
    examinee::AbstractExaminee
    val::Union{Missing,Float64}
    start_time::Dates.DateTime
    end_time::Dates.DateTime
    Response(item, examinee, val, start_time, end_time) = new(item, examinee, val, start_time, end_time)
    Response(item, examinee, val, time::Dates.DateTime) = new(item, examinee, val, time, time)
end

# Outer Constructor Methods
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
    filter(r -> r.examinee.idx == idx, responses)
end

"""
    get_item_responses(idx::Int64, responses::Vector{Response})

It returns the vector of responses given to item with index `idx`.
"""
function get_examinee_responses(idx::Int64, responses::Vector{Response})
    filter(r -> r.item.idx == idx, responses)
end

"""
    generate_response(latent::Latent1D, parameters::AbstractParameters)

Randomly generate a response for a 1-dimensional latent variable and custom item parameters.
"""
function generate_response(latent::Latent1D, parameters::AbstractParameters)
    Float64(rand(Distributions.Bernoulli(probability(latent, parameters, Array{Float64,1}(undef,0), Array{Float64,1}(undef,0)))))::Float64
end


"""
    generate_response(examinee::AbstractExaminee, item::AbstractItem)

Randomly generate a response by `examinee` to `item`.
"""
function generate_response(examinee::AbstractExaminee, item::AbstractItem)
    Response(item, examinee,generate_response(examinee.latent, item.parameters), Dates.now())::Response
end