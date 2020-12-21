"""
    _D_gain_method(parameters::Parameters1PL, latent::Latent1D)
"""
function _D_gain_method(parameters::Parameters1PL, latent::Latent1D)
    return _expected_information_item(parameters, latent)
end

"""
    _D_gain_method(parameters::Parameters2PL, latent::Latent1D)
"""
function _D_gain_method(parameters::Parameters2PL, latent::Latent1D)
    old_exp_info = copy(parameters.expected_information)
    return LinearAlgebra.det(old_exp_info + _expected_information_item(parameters, latent)) -
           LinearAlgebra.det(old_exp_info)
end

"""
    D_gain_method(item::AbstractItem, examinee::AbstractExaminee)

    # Description
    Computes the gain in the determinant of the expected information matrix for an item.
    
    # Arguments
    - **`item::AbstractItem`**: The item.
    - **`examinee::AbstractExaminee`**: The examinee at which computing the information.
    
    # Output
    It returns a `Float64` scalar.
    
    # References
    __Ren H, van der Linden WJ, Diao Q. Continuous online item calibration: Parameter recovery and item calibration. Psychometrika. 2017;82:498–522. doi: 10.1007/s11336-017-9553-1.__
"""
function D_gain_method(item::AbstractItem, examinee::AbstractExaminee)
    return _D_gain_method(item.parameters, examinee.latent)
end
