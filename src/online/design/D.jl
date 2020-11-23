
"""
D_method(item::Item1PL, examinee::AbstractExaminee)

# Description
It calls the function `expected_information_item` and returns its result.

# Arguments
- **`item::Item1PL`**: The 1PL item.
- **`examinee::AbstractExaminee`**: The examinee at which computing the information.

# Output
It returns a `Float64` scalar.
"""
function D_method(item::Item1PL, examinee::AbstractExaminee)
return expected_information_item(item.parameters, examinee.latent)
end

"""
D_method(item::Union{Item2PL, Item3PL}, examinee::AbstractExaminee)

# Description
Computes the determinant of the expected information matrix for a 2PL or 3PL item and a generic type examinee.

# Arguments
- **`item::Union{Item2PL, Item3PL},`**: The 2PL or 3PL item.
- **`examinee::AbstractExaminee`**: The examinee at which computing the information.

# Output
It returns a `Float64` scalar.
"""
function D_method(item::Union{Item2PL, Item3PL}, examinee::AbstractExaminee)
return LinearAlgebra.det(expected_information_item(item.parameters, examinee.latent))
end

