"""
    _A_method(parameters::Parameters1PL, latent::Latent1D)
"""
function _A_method(parameters::Parameters1PL, latent::Latent1D)
    return _item_expected_information(parameters, latent)
end

"""
    _A_method(parameters::Parameters2PL, latent::Latent1D)
"""
function _A_method(parameters::Parameters2PL, latent::Latent1D)
    return LinearAlgebra.tr(_item_expected_information(parameters, latent))
end

"""
    A_method(item::AbstractItem, examinee::AbstractExaminee)

# Description
Computes the trace of the expected information matrix for an item and an examinee.
    
# Arguments
- **`item::AbstractItem`**: The item.
- **`examinee::AbstractExaminee`**: The examinee at which computing the information.
    
# Output
It returns a `Float64` scalar.
"""
function A_method(item::AbstractItem, examinee::AbstractExaminee)
    return _A_method(item.parameters, examinee.latent)
end
