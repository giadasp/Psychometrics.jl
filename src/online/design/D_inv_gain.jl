"""
    _D_inv_gain_method(parameters::Parameters1PL, latent::Latent1D)
"""
function _D_inv_gain_method(parameters::Parameters1PL, latent::Latent1D)
    return 1 / _expected_information_item(parameters, latent)
end

"""
    _D_inv_gain_method(parameters::Parameters2PL, latent::Latent1D)
"""
function _D_inv_gain_method(parameters::Parameters2PL, latent::Latent1D)
    old_exp_info = copy(parameters.expected_information)
    return LinearAlgebra.det(LinearAlgebra.inv(old_exp_info)) -
           LinearAlgebra.det(LinearAlgebra.inv(
        old_exp_info + _expected_information_item(parameters, latent),
    ))
end

"""
    D_inv_gain_method(item::AbstractItem, examinee::AbstractExaminee)

# Description
It computes the gain in the inverse of the expected information of the item.

# Arguments
  - **`item::AbstractItem`**: The item.
  - **`examinee::AbstractExaminee`**: The examinee at which computing the information.

# Output
It returns a `Float64` scalar.

# References
__Yinhong He & Ping Chen, 2020. "Optimal Online Calibration Designs for Item Replenishment in Adaptive Testing," Psychometrika, Springer;The Psychometric Society, vol. 85(1), pages 35-55, March.__
"""
function D_inv_gain_method(item::AbstractItem, examinee::AbstractExaminee)
    return _D_inv_gain_method(item.parameters, examinee.latent)
end
