
"""
Item3PL <: AbstractItemBinary

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
Item3PL(idx, id, content) = new(idx, id, content, Parameters3PL(), true)

Randomly generates a new 3PL item with custom index, id, content features and default 3PL item parameters 
(Look at (`Parameters3PL`)[#Psychometrics.Parameters3PL] for the defaults).
"""
struct Item3PL <: AbstractItemBinary
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
    get_parameters(item::Item3PL)

Returns a vector with three values = [b, a, c]
"""
function get_parameters(item::Item3PL)
    [item.parameters.b, item.parameters.a, item.parameters.c]
end
