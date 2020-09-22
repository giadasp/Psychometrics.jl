
abstract type AbstractItem end

"""
    Item1PL <: AbstractItem

# Description
Item struct under the 1-parameter logistic model.

# Fields
  - **`idx::Int64`**: An integer that identifies the examinee.
  - **`id::String`**: A string that identifies the examinee.
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
  - **`idx::Int64`**: An integer that identifies the examinee.
  - **`id::String`**: A string that identifies the examinee.
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
  - **`idx::Int64`**: An integer that identifies the examinee.
  - **`id::String`**: A string that identifies the examinee.
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
  - **`idx::Int64`**: An integer that identifies the examinee.
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
    Item(idx, id, content) = new(idx, id, content, Parameters1PL())
end

"""
    get_item_by_idx(item_idx::Int64, items::Vector{<:AbstractItem})

It returns the item with index `item_idx` from a vector of <:AbstractItem.
"""
function get_item_by_idx(item_idx::Int64, items::Vector{<:AbstractItem})
   filter(i -> i.idx == item_idx, items)
end