
abstract type AbstractExaminee end
include("latent/latent.jl")

"""
# Description
An immutable containing the information about the examinee (the test taker).

# Fields
  - **`idx::Int64`**: An integer that identifies the Examinee in this session.
  - **`id::String`**: A string that identifies the Examinee.
  - **`latent::Latent`**: A mutable latent struct associated with the Examinee.

# Factories
    Examinee(idx, id, latent) = new(idx, id, latent)

Creates a new Examinee with custom index, id and a generic latent variable.

# Random initializers
    Examinee(idx, id) = new(idx, id, Latent1D())

Randomly generates an Examinee with custom index and id and with a default 1-dimensional latent variable 
(Look at (`Latent1D`)[#Psychometrics.Latent1D] for the defaults).
"""
struct Examinee <: AbstractExaminee
    idx::Int64
    id::String
    latent::AbstractLatent

    # Factories
    Examinee(idx, id, latent) = new(idx, id, latent)

    # Random initializers
    Examinee(id) = new(idx, id, Latent1D())
end

"""
    get_examinee_by_id(examinee_id::String, examinees::Vector{<:AbstractExaminee})

It returns the Examinee with index `examinee_id` from a Vector of AbstractExaminee.
"""
function get_examinee_by_id(examinee_id::String, examinees::Vector{<:AbstractExaminee})
    filter(e -> e.id == examinee_id, examinees)[1]
end


"""
    get_latents(examinees::Vector{<:AbstractExaminee})

Returns a matrix with latent values displayed by row.
"""
function get_latents(examinees::Vector{<:AbstractExaminee})
    ret = Vector{Vector{Float64}}(undef, size(examinees, 1))
    max_length = 1
    e_2 = 0
    for e in examinees
        e_2 += 1
        local latents = get_latents(e)
        ret[e_2] = copy(latents)
        max_length = max_length < size(latents, 1) ? size(latents, 1) : max_length
    end

    for e_3 = 1:e_2
        local length_i = size(ret[e_3], 1)
        if length_i < max_length
            ret[e_3] = vcat(ret[e_3], zeros(Float64, max_length - length_i))
        end
    end
    return reduce(hcat, ret)
end


"""
    empty_chain!(examinee::AbstractExaminee)
"""
function empty_chain!(examinee::AbstractExaminee)
    _empty_chain!(examinee.latent)
end