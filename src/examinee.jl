abstract type AbstractExaminee end

"""
    Examinee1D <: AbstractExaminee

# Description
Examinee struct with a 1-dimensional latent variable.

# Fields
  - **`id::String`**: A string that identifies the Examinee.
  - **`latent::Latent1D`**: A 1-dimensional latent variable associated with the Examinee.

# Factories
    Examinee1D(id, latent) = new(id, latent)

Creates a new Examinee with custom index, id and 1-dimensional latent variable.

# Random initializers
    Examinee1D(id) = new(id, Latent1D())

Randomly generates an Examinee with custom index and id and with a default 1-dimensional latent variable 
(Look at (`Latent1D`)[#Psychometrics.Latent1D] for the defaults).
"""
mutable struct Examinee1D <: AbstractExaminee
    id::String
    latent::Latent1D

    # Factories
    Examinee1D(id, latent) = new(id, latent)

    # Random initializers
    Examinee1D(id) = new(id, Latent1D())
end

"""
    Examinee <: AbstractExaminee

# Description
Examinee struct with a generic latent variable.

# Fields
  - **`id::String`**: A string that identifies the Examinee.
  - **`latent::Latent`**: A generic latent variable associated with the Examinee.

# Factories
    Examinee1D(id, latent) = new(id, latent)

Creates a new Examinee with custom index, id and a generic latent variable.

# Random initializers
    Examinee(id) = Examinee1D(id)

Randomly generates an Examinee with custom index and id and with a default 1-dimensional latent variable 
(Look at (`Latent1D`)[#Psychometrics.Latent1D] for the defaults).
"""
mutable struct Examinee <: AbstractExaminee
    id::String
    latent::AbstractLatent

    # Factories
    Examinee(id, latent) = new(id, latent)

    # Random initializers
    Examinee(id) = Examinee1D(id)
end


"""
    answer(Examinee::AbstractExaminee, item::AbstractParameters)

Randomly generate a response by `Examinee` to `item`.
"""
function answer(Examinee::AbstractExaminee, item::AbstractParameters)
    _generate_response(Examinee.latent, item.parameters)
end

"""
    answer(examinee_idx::Int64, item_idx::Int64, examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})

Randomly generate a response by `Examinee` with index `examinee_idx` to `item` with index `item_idx`.
"""
function answer(examinee_idx::Int64, item_idx::Int64, examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})
    answer(get_examinee_by_idx!(examinee_idx, examinees), get_item_by_idx!(item_idx, items))
end

"""
    answer(examinees:AbstractExaminee}, items::Vector{<:AbstractItem})

Randomly generate responses by all the examinees in `examinees` to items in `items`.
"""
function answer(examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})
    map((e, i) -> answer(e, i), examinees, items)
end

"""
    get_examinee_by_idx!(examinee_idx::Int64, examinees::Dict{Int64,<:AbstractExaminee})

It returns the Examinee with index `examinee_idx` from a Dict of AbstractExaminee.
"""
function get_examinee_by_idx(examinee_idx::Int64, examinees::Dict{Int64,<:AbstractExaminee})
    examinees[examinee_idx]
end