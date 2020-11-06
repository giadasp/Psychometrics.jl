
"""
Item1PL <: AbstractItem

# Description
Item struct under the 1-parameter logistic model.

# Fields
- **`idx::Int64`**: An integer that identifies the item in this session.
- **`id::String`**: A string that identifies the item.
- **`content::Vector{String}`**: A string vector containing the content features of an item.
- **`parameters::Parameters1PL`**: A `Parameters1PL` object.
- **`calibrated::Bool`**: Tells if the item has been already calibrated.

# Factories
Item1PL(idx, id, content, parameters, calibrated) = new(idx, id, content, parameters, calibrated)

Creates a new 1PL item with custom index, id, content features and item parameters.

# Random initilizers
Item1PL(idx, id, content) = new(idx, id, content, Parameters1PL(), true)

Randomly generates a new 1PL item with custom index, id, content features and default 1PL item parameters 
(Look at (`Parameters1PL`)[#Psychometrics.Parameters1PL] for the defaults).
"""
struct Item1PL <: AbstractItem
idx::Int64
id::String
content::Vector{String}
parameters::Parameters1PL
calibrated::Bool

# Factories
Item1PL(idx, id, content, parameters, calibrated) = new(idx, id, content, parameters, calibrated)

# Random initilizers
Item1PL(idx, id, content) = new(idx, id, content, Parameters1PL(), true)
end


"""
    get_parameters(item::Item1PL)

"""
function get_parameters(item::Item1PL)
    [item.parameters.b]
end