# Items

```@meta
CurrentModule = Psychometrics
```

## Struct

```@docs
Item
```

## Methods for a Single Item

```@docs
get_parameters(item::AbstractItem)
get_item_by_id(item_id::String, items::Vector{<:AbstractItem})
get_item_by_id(item_id::String, items::Vector{<:AbstractItem})
get_parameters_vals(item::AbstractItem)
empty_chain!(item::AbstractItem)
chain_append!(item::AbstractItem; sampling = false)
chain_append!(item::AbstractItem)
```

## Methods for a Vector of Items

```@docs
get_parameters_vals(items::Vector{<:AbstractItem})
get_parameters(items::Vector{<:AbstractItem})
```
