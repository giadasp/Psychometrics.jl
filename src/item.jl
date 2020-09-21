
abstract type AbstractItem end

mutable struct Item <: AbstractItem
end

mutable struct Item1PL <: AbstractItem
    idx::Int64
    id::String
    content::Vector{String}
    parameters::Parameters1PL
    Item1PL(idx, id, content, parameters) = new(idx, id, content, parameters)
    # Random initilizer
    Item1PL(idx, id, content) = new(idx, id, content, Parameters1PL())
end 

mutable struct Item2PL <: AbstractItem
    idx::Int64
    id::String
    content::Vector{String}
    parameters::Parameters2PL
    Item2PL(idx, id, content, parameters) = new(idx, id, content, parameters)
    # Random initilizer
    Item2PL(idx, id, content) = new(idx, id, content, Parameters2PL())
end
    
mutable struct Item3PL <: AbstractItem
    idx::Int64
    id::String
    content::Vector{String}
    parameters::Parameters3PL
    Item3PL(idx, id, content, parameters) = new(idx, id, content, parameters)
    # Random initilizer
    Item3PL(idx, id, content) = new(idx, id, content, Parameters3PL())
end

"""
    get_examinees!(item_idx::Int64, responses::Vector{<:AbstractResponse})

It returns the esaminees who answered to the item with index `item_idx`.
"""
function get_examinees!(item_idx::Int64, responses::Vector{<:AbstractResponse})
   filter(r -> r.item.idx == item_idx,responses)
end

"""
    get_item_by_idx!(item_idx::Int64, items::Vector{<:AbstractItem})

It returns the item with index `item_idx` from a vector of <:AbstractItem.
"""
function get_item_by_idx!(item_idx::Int64, items::Vector{<:AbstractItem})
   filter(i -> i.idx == item_idx, items)
end