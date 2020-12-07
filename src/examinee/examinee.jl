
abstract type AbstractExaminee end

include("latent/1D.jl")
include("latent/latent.jl")

"""
    Examinee <: AbstractExaminee

# Description
Examinee struct with a generic latent variable.

# Fields
  - **`idx::Int64`**: An integer that identifies the Examinee in this session.
  - **`id::String`**: A string that identifies the Examinee.
  - **`latent::Latent`**: A generic latent variable associated with the Examinee.

# Factories
    Examinee1D(idx, id, latent) = new(idx, id, latent)

Creates a new Examinee with custom index, id and a generic latent variable.

# Random initializers
    Examinee(idx, id) = Examinee1D(idx, id)

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
    Examinee(id) = Examinee1D(id)
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

