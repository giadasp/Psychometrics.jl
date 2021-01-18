"""
    find_best_item(examinee, items)

# Description
Finds the best item for the `examinee` among the vector `items` using maximum information criterion.

# Arguments
  - **`examinee::AbstractExaminee`**: The examinee.
  - **`items:::Vector{<:AbstractItem}`**: Set of items in which searching for the best.
  - **`method`**: Optional. "max-info" | "D-gain". 
  max-info takes the item with the maximum latent information,
  D-gain takes the item with the maximum gain in the determinant of the expected information matrix if examinee would answer to the item (D-VC in __Ren, van der Linden, Diao, 2017__). 

# Output
It returns an item of generic type.

# References
[^RenDiao2017]

[^RenDiao2017]:Ren H, van der Linden WJ, Diao Q. Continuous online item calibration: Parameter recovery and item calibration. Psychometrika. 2017;82:498–522. doi: 10.1007/s11336-017-9553-1.
"""
function find_best_item(
    examinee::AbstractExaminee,
    items::Vector{<:AbstractItem};
    method = "max-info",
)
    if method in ["max-info", "D-gain"]
        if method == "max-info"
            infos = information_latent(examinee, items)
        elseif method == "D-gain"
            infos = map(i -> D_gain_method(i, examinee), items)
        end
        return items[findmax(infos)[2]].idx::Int64
    else
        error("only max-info and D-gain optimality are available.")
    end
end

"""
    find_best_examinee(item, examinees; method = "D")

# Description
Finds the best examinee among the `examinees` vector, for the `item` using maximum expected information criterion.

# Arguments
- **`item::AbstractItem`**: Required. Set of items in which searching for the best.
- **`examinees::Vector{<:AbstractExaminee}`**: Required. The examinee.
- **`method`**: Optional. "D" | "A". 
  D stands for D-optimality (determinant of the expected information matrix),
  A stands for A-optimality (trace of the expected information matrix). 

# Output
It returns the idx of the best examinee.

"""
function find_best_examinee(
    item::AbstractItem,
    examinees::Vector{<:AbstractExaminee};
    method = "D",
)
    if method in ["D", "A"]
        if method == "D"
            infos = map(e -> D_method(item, e), examinees)
        elseif method == "A"
            infos = map(e -> A_method(item, e), examinees)
        end
        return examinees[findmax(infos)[2]].idx::Int64
    else
        error("only D-optimalit and A-optimality are available.")
    end
end
