
"""
D_gain_method(item::Item1PL, examinee::AbstractExaminee)

# Description
It calls the function `expected_information_item` and returns its result.

# Arguments
- **`item::Item1PL`**: The 1PL item.
- **`examinee::AbstractExaminee`**: The examinee at which computing the information.

# Output
It returns a `Float64` scalar.

# References
__Ren H, van der Linden WJ, Diao Q. Continuous online item calibration: Parameter recovery and item calibration. Psychometrika. 2017;82:498–522. doi: 10.1007/s11336-017-9553-1.__
"""
function D_gain_method(item::Item1PL, examinee::AbstractExaminee)
    return expected_information_item(item.parameters, examinee.latent)
end

"""
D_gain_method(item::Union{Item2PL, Item3PL}, examinee::AbstractExaminee)

# Description
Computes the gain in the determinant of the expected information for a 2PL or 3PL item.

# Arguments
- **`item::Item2PL`**: The 2PL or 3PL item.
- **`examinee::AbstractExaminee`**: The examinee at which computing the information.

# Output
It returns a `Float64` scalar.

# References
__Ren H, van der Linden WJ, Diao Q. Continuous online item calibration: Parameter recovery and item calibration. Psychometrika. 2017;82:498–522. doi: 10.1007/s11336-017-9553-1.__
"""
function D_gain_method(item::Union{Item2PL, Item3PL}, examinee::AbstractExaminee)
    old_exp_info = copy(item.parameters.expected_information)
    return LinearAlgebra.det(old_exp_info + expected_information_item(item.parameters, examinee.latent)) - LinearAlgebra.det(old_exp_info)
end
