# Items

```@meta
    CurrentModule = Psychometrics
```
```@docs
get_parameters(item::AbstractItem)
get_parameters_vals(item::AbstractItem)
empty_chain!(item::AbstractItem)
chain_append!(item::AbstractItem; sampling = false)

get_parameters(items::Vector{<:AbstractItem})
get_parameters_vals(items::Vector{<:AbstractItem})
get_item_by_id(item_id::String, items::Vector{<:AbstractItem})
```
