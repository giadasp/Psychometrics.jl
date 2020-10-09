abstract type AbstractItem end

"""
    Item1PL <: AbstractItem

# Description
Item struct under the 1-parameter logistic model.

# Fields
  - **`idx::Int64`**: An integer that identifies the item in this session.
  - **`id::String`**: A string that identifies the item.
  - **`content::Vector{String}`**: A string vector containing the content features of an item.
  - **`parameters::Parameters1PL`**: A `Parameters1PL` object.

# Factories
    Item1PL(idx, id, content, parameters) = new(idx, id, content, parameters)

Creates a new 1PL item with custom index, id, content features and item parameters.

# Random initilizers
    Item1PL(idx, id, content) = new(idx, id, content, Parameters1PL())

Randomly generates a new 1PL item with custom index, id, content features and default 1PL item parameters 
(Look at (`Parameters1PL`)[#Psychometrics.Parameters1PL] for the defaults).
"""
mutable struct Item1PL <: AbstractItem
    idx::Int64
    id::String
    content::Vector{String}
    parameters::Parameters1PL

    # Factories
    Item1PL(idx, id, content, parameters) = new(idx, id, content, parameters)

    # Random initilizers
    Item1PL(idx, id, content) = new(idx, id, content, Parameters1PL())
end


"""
    Item2PL <: AbstractItem

# Description
Item struct under the 2-parameter logistic model.

# Fields
  - **`idx::Int64`**: An integer that identifies the item in this session.
  - **`id::String`**: A string that identifies the item.
  - **`content::Vector{String}`**: A string vector containing the content features of an item.
  - **`parameters::Parameters2PL`**: A `Parameters2PL` object.

# Factories
    Item2PL(idx, id, content, parameters) = new(idx, id, content, parameters)

Creates a new 2PL item with custom index, id, content features and item parameters.

# Random initilizers
    Item2PL(idx, id, content) = new(idx, id, content, Parameters2PL())

Randomly generates a new 2PL item with custom index, id, content features and default 2PL item parameters 
(Look at (`Parameters2PL`)[#Psychometrics.Parameters2PL] for the defaults).
"""
mutable struct Item2PL <: AbstractItem
    idx::Int64
    id::String
    content::Vector{String}
    parameters::Parameters2PL

    #Factories
    Item2PL(idx, id, content, parameters) = new(idx, id, content, parameters)

    # Random initilizers
    Item2PL(idx, id, content) = new(idx, id, content, Parameters2PL())
end


"""
    Item3PL <: AbstractItem

# Description
Item struct under the 3-parameter logistic model.

# Fields
  - **`idx::Int64`**: An integer that identifies the item in this session.
  - **`id::String`**: A string that identifies the item.
  - **`content::Vector{String}`**: A string vector containing the content features of an item.
  - **`parameters::Parameters2PL`**: A `Parameters3PL` object.

# Factories
    Item3PL(idx, id, content, parameters) = new(idx, id, content, parameters)

Creates a new 3PL item with custom index, id, content features and item parameters.

# Random initilizers
    Item3PL(idx, id, content) = new(idx, id, content, Parameters3PL())

Randomly generates a new 3PL item with custom index, id, content features and default 3PL item parameters 
(Look at (`Parameters3PL`)[#Psychometrics.Parameters3PL] for the defaults).
"""
mutable struct Item3PL <: AbstractItem
    idx::Int64
    id::String
    content::Vector{String}
    parameters::Parameters3PL

    # Factories
    Item3PL(idx, id, content, parameters) = new(idx, id, content, parameters)

    # Random initilizers
    Item3PL(idx, id, content) = new(idx, id, content, Parameters3PL())
end

"""
    Item <: AbstractItem

# Description
A generic item struct.

# Fields
  - **`idx::Int64`**: An integer that identifies the item in this session.
  - **`id::String`**: A string that identifies the examinee.
  - **`content::Vector{String}`**: A string vector containing the content features of an item.
  - **`parameters::AbstractParameters`**: A generic item parameters object.

# Factories
    Item(idx, id, content, parameters) = new(idx, id, content, parameters)

Creates a new 3PL item with custom index, id, content features and item parameters.

# Random initilizers
    Item(idx, id, content) = new(idx, id, content, Parameters1PL())

Randomly generates a new generic item with custom index, id, content features and default 1PL item parameters 
(Look at (`Parameters1PL`)[#Psychometrics.Parameters1PL] for the defaults).
"""
mutable struct Item <: AbstractItem
    idx::Int64
    id::String
    content::Vector{String}
    parameters::AbstractParameters

    #Factories
    Item(idx, id, content, parameters) = new(idx, id, content, parameters)

    # Random initilizers
    Item(idx,  id, content) = new(idx, id, content, Parameters1PL())
end

"""
    get_item_by_id(item_id::String, items::Vector{<:AbstractItem})

It returns the item with index `item_id` from a vector of <:AbstractItem.
"""
function get_item_by_id(item_id::String, items::Vector{<:AbstractItem})
    filter(i -> i.id == item_id, items)[1]
end

"""
    get_parameters(item::Item1PL)

"""
function get_parameters(item::Item1PL)
    [item.parameters.b]
end

"""
    get_parameters(item::Item2PL)

"""
function get_parameters(item::Item2PL)
    [item.parameters.b, item.parameters.a]
end

"""
    get_parameters(item::Item3PL)

Returns a vector with three values = [b, a, c]
"""
function get_parameters(item::Item3PL)
    [item.parameters.b, item.parameters.a, item.parameters.c]
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
        max_length = max_length < size(pars,1) ? size(pars,1) : max_length
    end
    
    for i_3 in 1:i_2
        local length_i = size(ret[i_3], 1)
        if length_i < max_length
            ret[i_3] = vcat(ret[i_3], zeros(Float64, max_length - length_i))
        end
    end
    return permutedims(reduce(hcat, ret))
end