
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
struct Item2PL <: AbstractItem
    idx::Int64
    id::String
    content::Vector{String}
    parameters::Parameters2PL

    # Factories
    Item2PL(idx, id, content, parameters) = new(idx, id, content, parameters)

    # Random initilizers
    Item2PL(idx, id, content) = new(idx, id, content, Parameters2PL())
end



"""
    get_parameters(item::Item2PL)

"""
function get_parameters(item::Item2PL)
    [item.parameters.b, item.parameters.a]
end


"""
    empty_chain!(item::Item2PL)

"""
function empty_chain!(item::Item2PL)
    item.parameters.chain = Vector{Vector{Float64}}(undef,0)
end
