
abstract type AbstractExaminee end
include("latent/latent.jl")

"""
    Examinee <: AbstractExaminee

# Description

An immutable containing the information about the examinee (the test taker).

# Fields

  - **`idx::Int64`**: An integer that identifies the Examinee in this session.
  - **`id::String`**: A string that identifies the Examinee.
  - **`latent::Latent`**: A mutable latent struct associated with the Examinee.
  - **`assessed::Bool`**: If the examinee has already been assessed.

# Factories
    Examinee(idx::Int64, id::String, latent::AbstractLatent, assessed::Bool) = new(idx, id, latent, assessed)
    Examinee(idx::Int64, id::String, latent::AbstractLatent) = new(idx, id, latent, false)
    Examinee(idx::Int64, latent::AbstractLatent) = new(idx, string(idx), latent, false)

Creates a new Examinee with custom index, (id) and a latent variable.

# Random default initializers (1-D latent)
    Examinee(idx::Int64) = new(idx, string(idx), Latent1D(), false)
    Examinee(idx::Int64, id::String) = new(idx, id, Latent1D(), false)

# Description

Randomly generates an Examinee with custom index and id and with a default 1-dimensional latent variable 
(Look at (`Latent1D`)[#Psychometrics.Latent1D] for the defaults).
"""
struct Examinee <: AbstractExaminee
    idx::Int64
    id::String
    latent::AbstractLatent
    assessed::Bool

    # Factories
    Examinee(idx::Int64, id::String, latent::AbstractLatent, assessed::Bool) = new(idx, id, latent, assessed)
    Examinee(idx::Int64, id::String, latent::AbstractLatent) = new(idx, id, latent, false)
    Examinee(idx::Int64, latent::AbstractLatent) = new(idx, string(idx), latent, false)

    # Random default initializers
    Examinee(idx::Int64) = new(idx, string(idx), Latent1D(), false)
    Examinee(idx::Int64, id::String) = new(idx, id, Latent1D(), false)
end

"""
    get_examinee_by_id(examinee_id::String, examinees::Vector{<:AbstractExaminee})

It returns the Examinee with index `examinee_id` from a Vector of AbstractExaminee.
"""
function get_examinee_by_id(examinee_id::String, examinees::Vector{<:AbstractExaminee})
    filter(e -> e.id == examinee_id, examinees)[1]
end

"""
    empty_chain!(examinee::AbstractExaminee)
"""
function empty_chain!(examinee::AbstractExaminee)
    _empty_chain!(examinee.latent)
end

"""
    set_val!(examinee::AbstractExaminee, val::Float64)
"""
function set_val!(examinee::AbstractExaminee, val::Float64)
    _set_val!(examinee.latent, val)
end

"""
    set_val_from_chain!(examinee::AbstractExaminee)
"""
function set_val_from_chain!(examinee::AbstractExaminee)
    _set_val_from_chain!(examinee.latent)
end


"""
    update_estimate!(examinee::AbstractExaminee; sampling = true)
"""
function update_estimate!(examinee::AbstractExaminee; sampling = true)
    _update_estimate!(examinee.latent; sampling = sampling)
end


"""
    chain_append!(examinee::AbstractExaminee; sampling = false)
"""
function chain_append!(examinee::AbstractExaminee; sampling = false)
    _chain_append!(examinee.latent; sampling = sampling)
end


"""
    set_prior!(
        examinee::AbstractExaminee,
        prior::Union{Distributions.DiscreteUnivariateDistribution, Distributions.ContinuousUnivariateDistribution}
    )
"""
function  set_prior!(
    examinee::AbstractExaminee,
    prior::Union{Distributions.DiscreteUnivariateDistribution, Distributions.ContinuousUnivariateDistribution}
)
    _set_prior!(examinee.latent, prior)
    return nothing
end