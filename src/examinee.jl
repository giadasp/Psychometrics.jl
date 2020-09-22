abstract type AbstractExaminee end

"""
    Examinee1D <: AbstractExaminee

# Description
Examinee struct with a 1-dimensional latent variable.

# Fields
  - **`idx::Int64`**: An integer that identifies the examinee.
  - **`id::String`**: A string that identifies the examinee.
  - **`latent::Latent1D`**: A 1-dimensional latent variable associated with the examinee.

# Factories
    Examinee1D(idx, id, latent) = new(idx, id, latent)

Creates a new examinee with custom index, id and 1-dimensional latent variable.

# Random initializers
    Examinee1D(idx, id) = new(idx, id, Latent1D())

Randomly generates an examinee with custom index and id and with a default 1-dimensional latent variable 
(Look at (`Latent1D`)[#Psychometrics.Latent1D] for the defaults).
"""
mutable struct Examinee1D <: AbstractExaminee
    idx::Int64
    id::String
    latent::Latent1D

    # Factories
    Examinee1D(idx, id, latent) = new(idx, id, latent)

    # Random initializers
    Examinee1D(idx, id) = new(idx, id, Latent1D())
end

"""
    Examinee <: AbstractExaminee

# Description
Examinee struct with a generic latent variable.

# Fields
  - **`idx::Int64`**: An integer that identifies the examinee.
  - **`id::String`**: A string that identifies the examinee.
  - **`latent::Latent`**: A generic latent variable associated with the examinee.

# Factories
    Examinee1D(idx, id, latent) = new(idx, id, latent)

Creates a new examinee with custom index, id and a generic latent variable.

# Random initializers
    Examinee(idx, id) = Examinee1D(idx, id)

Randomly generates an examinee with custom index and id and with a default 1-dimensional latent variable 
(Look at (`Latent1D`)[#Psychometrics.Latent1D] for the defaults).
"""
mutable struct Examinee <: AbstractExaminee
    idx::Int64
    id::String
    latent::AbstractLatent

    # Factories
    Examinee(idx, id, latent) = new(idx, id, latent)

    # Random initializers
    Examinee(idx, id) = Examinee1D(idx, id)
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