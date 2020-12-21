
"""
    _D_method(parameters::Parameters1PL, latent::Latent1D)
"""
function _D_method(parameters::Parameters1PL, latent::Latent1D)
    return _expected_information_item(parameters, latent)
end

"""
    _D_method(parameters::Parameters2PL, latent::Latent1D)
"""
function _D_method(parameters::Parameters2PL, latent::Latent1D)
    return LinearAlgebra.det(_expected_information_item(parameters, latent))
end

"""
    D_method(item::AbstractItem, examinee::AbstractExaminee)

# Description
Computes the determinant of the expected information matrix for an item and an examinee.

# Arguments
- **`item::AbstractItem`**: The item.
- **`examinee::AbstractExaminee`**: The examinee at which computing the information.

# Output
It returns a `Float64` scalar.
"""
function D_method(item::AbstractItem, examinee::AbstractExaminee)
    return _D_method(item.parameters, examinee.latent)
end
