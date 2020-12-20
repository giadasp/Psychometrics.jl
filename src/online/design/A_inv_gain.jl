"""
A_inv_gain_method(item::Item1PL, examinee::AbstractExaminee)

# Description
Computes the trace of the inverse of the expected information matrix for a 1PL item.

# Arguments
- **`item::Item1PL`**: The 1PL item.
- **`examinee::AbstractExaminee`**: The examinee at which computing the information.

# Output
It returns a `Float64` scalar.
"""
function A_inv_gain_method(item::Item1PL, examinee::AbstractExaminee)
    return expected_information_item(item.parameters, examinee.latent)
end

"""
A_inv_gain_method(item::Union{Item2PL, Item3PL}, examinee::AbstractExaminee)

# Description
Computes the trace of the inverse of the expected information matrix for a 2PL or 3PL item.

# Arguments
- **`item::Union{Item2PL, Item3PL}`**: The 2PL or 3PL item.
- **`examinee::AbstractExaminee`**: The examinee at which computing the information.

# Output
It returns a `Float64` scalar.
"""
function A_inv_gain_method(item::Union{Item2PL,Item3PL}, examinee::AbstractExaminee)
    old_exp_info = copy(item.parameters.expected_information)
    return LinearAlgebra.tr(
        LinearAlgebra.inv(
            old_exp_info + expected_information_item(item.parameters, examinee.latent),
        ) - LinearAlgebra.inv(old_exp_info),
    )
end
