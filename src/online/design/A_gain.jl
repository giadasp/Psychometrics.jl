"""
    _A_gain_method(parameters::Parameters1PL, latent::Latent1D)
"""
function _A_gain_method(parameters::Parameters1PL, latent::Latent1D)
    return _expected_information_item(parameters, latent)
end

"""
    _A_gain_method(parameters::Parameters2PL, latent::Latent1D)
"""
function _A_gain_method(parameters::Parameters2PL, latent::Latent1D)
    old_exp_info = copy(parameters.expected_information)
    return LinearAlgebra.tr(old_exp_info + _expected_information_item(parameters, latent)) -
           LinearAlgebra.tr(old_exp_info)
end

"""
    A_gain_method(item::AbstractItem, examinee::AbstractExaminee)

# Description
Computes the gain in the trace of the expected information matrix for an item.

# Arguments
- **`item::AbstractItem`**: The  item.
- **`examinee::AbstractExaminee`**: The examinee at which computing the information.

# Output
It returns a `Float64` scalar.
"""
function A_gain_method(item::AbstractItem, examinee::AbstractExaminee)
    return _A_gain_method(item.parameters, examinee.latent)
end
