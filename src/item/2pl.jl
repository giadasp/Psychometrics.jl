
"""
Item2PL <: AbstractItem

# Description
Item struct under the 2-parameter logistic model.

# Fields
- **`idx::Int64`**: An integer that identifies the item in this session.
- **`id::String`**: A string that identifies the item.
- **`content::Vector{String}`**: A string vector containing the content features of an item.
- **`parameters::Parameters2PL`**: A `Parameters2PL` object.
- **`calibrated::Bool`**: Tells if the item has been already calibrated.

# Factories
Item2PL(idx, id, content, parameters, calibrated) = new(idx, id, content, parameters, calibrated)

Creates a new 2PL item with custom index, id, content features and item parameters.

# Random initilizers
Item2PL(idx, id, content) = new(idx, id, content, Parameters2PL(), true)

Randomly generates a new 2PL item with custom index, id, content features and default 2PL item parameters 
(Look at (`Parameters2PL`)[#Psychometrics.Parameters2PL] for the defaults).
"""
struct Item2PL <: AbstractItem
idx::Int64
id::String
content::Vector{String}
parameters::Parameters2PL
calibrated::Bool

# Factories
Item2PL(idx, id, content, parameters, calibrated) = new(idx, id, content, parameters, calibrated)

# Random initilizers
Item2PL(idx, id, content) = new(idx, id, content, Parameters2PL(), true)
end



"""
    get_parameters(item::Item2PL)

"""
function get_parameters(item::Item2PL)
    [item.parameters.b, item.parameters.a]
end
