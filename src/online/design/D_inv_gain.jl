"""
    D_inv_gain_method(item::Item1PL, examinee::AbstractExaminee)

# Description
It computes the inverse of the expected information of the item (1/expected_information).

# Arguments
  - **`item::Item1PL`**: The 1PL item.
  - **`examinee::AbstractExaminee`**: The examinee at which computing the information.

# Output
It returns a `Float64` scalar.

# References
__Yinhong He & Ping Chen, 2020. "Optimal Online Calibration Designs for Item Replenishment in Adaptive Testing," Psychometrika, Springer;The Psychometric Society, vol. 85(1), pages 35-55, March.__
"""
function D_inv_gain_method(item::Item1PL, examinee::AbstractExaminee)
    return 1 / expected_information_item(item.parameters, examinee.latent)
end

"""
    D_inv_gain_method(item::Union{Item2PL, Item3PL}, examinee::AbstractExaminee)

# Description
Computes the gain in the determinant of the expected information for a 2PL or 3PL item.

# Arguments
  - **`item::Item2PL`**: The 2PL or 3PL item.
  - **`examinee::AbstractExaminee`**: The examinee at which computing the information.

# Output
It returns a `Float64` scalar.

# References
__Yinhong He & Ping Chen, 2020. "Optimal Online Calibration Designs for Item Replenishment in Adaptive Testing," Psychometrika, Springer;The Psychometric Society, vol. 85(1), pages 35-55, March.__
"""
function D_inv_gain_method(item::Union{Item2PL,Item3PL}, examinee::AbstractExaminee)
    old_exp_info = copy(item.parameters.expected_information)
    return LinearAlgebra.det(LinearAlgebra.inv(old_exp_info)) -
           LinearAlgebra.det(LinearAlgebra.inv(
        old_exp_info + expected_information_item(item.parameters, examinee.latent),
    ))
end
