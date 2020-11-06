abstract type AbstractItem end

include("parameters/parameters.jl")

include("1pl.jl")
include("2pl.jl")
include("3pl.jl")

"""
    Item <: AbstractItem

# Description
A generic item struct.

# Fields
  - **`idx::Int64`**: An integer that identifies the item in this session.
  - **`id::String`**: A string that identifies the examinee.
  - **`content::Vector{String}`**: A string vector containing the content features of an item.
  - **`parameters::AbstractParameters`**: A generic item parameters object.
  - **`calibrated::Bool`**: Tells if the item has been already calibrated.

# Factories
    Item(idx, id, content, parameters, calibrated) = new(idx, id, content, parameters, calibrated)

Creates a new generic item with custom index, id, content features and item parameters.

# Random initilizers
    Item(idx, id, content) = new(idx, id, content, Parameters1PL(), true)

Randomly generates a new generic calibrated item with custom index, id, content features and default 1PL item parameters 
(Look at (`Parameters1PL`)[#Psychometrics.Parameters1PL] for the defaults).
"""
mutable struct Item <: AbstractItem
    idx::Int64
    id::String
    content::Vector{String}
    parameters::AbstractParameters
    calibrated::Bool

    # Factories
    Item(idx, id, content, parameters, calibrated) = new(idx, id, content, parameters, calibrated)

    # Random initilizers
    Item(idx, id, content) = new(idx, id, content, Parameters1PL(), true)
end

"""
    get_item_by_id(item_id::String, items::Vector{<:AbstractItem})

It returns the item with index `item_id` from a vector of <:AbstractItem.
"""
function get_item_by_id(item_id::String, items::Vector{<:AbstractItem})
    filter(i -> i.id == item_id, items)[1]
end


"""
    get_parameters(items::Vector{<:AbstractItem})

Returns a matrix with item parameters displayed by row.
"""
function get_parameters(items::Vector{<:AbstractItem})
    ret = Vector{Vector{Float64}}(undef, size(items, 1))
    max_length = 1
    i_2 = 0
    for i in items
        i_2 += 1
        local pars = get_parameters(i)
        ret[i_2] = copy(pars)
        max_length = max_length < size(pars, 1) ? size(pars, 1) : max_length
    end

    for i_3 = 1:i_2
        local length_i = size(ret[i_3], 1)
        if length_i < max_length
            ret[i_3] = vcat(ret[i_3], zeros(Float64, max_length - length_i))
        end
    end
    return permutedims(reduce(hcat, ret))
end
