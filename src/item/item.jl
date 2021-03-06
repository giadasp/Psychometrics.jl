
include("parameters/parameters.jl")
"""
    AbstractItem
"""
abstract type AbstractItem end

"""
    Item <: AbstractItem

# Description

An immutable containing information about an item (a question in a test), e.g. `id::String` the item identifier, `calibrated::Bool` says if it is a field item (`false`) or an operational item (`true`), and the field `parameters::AbstractParameter` which accepts a mutable item parameter object

# Fields

  - **`idx::Int64`**: An integer that identifies the item in this session.
  - **`id::String`**: A string that identifies the examinee.
  - **`content::Vector{String}`**: A string vector containing the content features of an item.
  - **`parameters::AbstractParameters`**: A generic item parameters object.
  - **`calibrated::Bool`**: Tells if the item has been already calibrated.

# Factories
    Item(idx, id, content, parameters) = new(idx, id, content, parameters)
    Item(idx, id, parameters) = new(idx, id, "", parameters)

Creates a new generic item with custom index, id, content features and item parameters.

# Random initilizers
    Item(idx, id) = new(idx, id, "", Parameters1PL(), true)
    Item(idx, id, content) = new(idx, id, content, Parameters1PL(), true)


Randomly generates a new generic calibrated item with custom index, id, content features and default 1PL item parameters 
(Look at (`Parameters1PL`)[#Psychometrics.Parameters1PL] for the defaults).
"""
struct Item <: AbstractItem
    idx::Int64
    id::String
    content::Vector{String}
    parameters::AbstractParameters

    # Factories
    Item(idx, id, content, parameters) = new(idx, id, content, parameters)
    Item(idx, id, parameters) = new(idx, id, "", parameters)

    # Random default initilizers
    Item(idx, id) = new(idx, id, "", Parameters1PL())
    Item(idx, id, content) = new(idx, id, content, Parameters1PL())
end

"""
    get_parameters(item::AbstractItem)
"""
function get_parameters(item::AbstractItem)
    item.parameters
end

"""
    get_item_by_id(item_id::String, items::Vector{<:AbstractItem})

# Description

It returns the item with index `item_id` from a vector of <:AbstractItem.
"""
function get_item_by_id(item_id::String, items::Vector{<:AbstractItem})
    filter(i -> i.id == item_id, items)[1]
end

"""
    get_parameters_vals(item::AbstractItem)
"""
function get_parameters_vals(item::AbstractItem)
    _get_parameters_vals(item.parameters)
end

"""
    empty_chain!(item::AbstractItem)
"""
function empty_chain!(item::AbstractItem)
    _empty_chain!(item.parameters)
end

"""
    chain_append!(item::AbstractItem; sampling = false)
"""
function chain_append!(item::AbstractItem)
    _chain_append!(item.parameters)
end

