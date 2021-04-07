
########################################################################
# Expected item info -  for 1PL pars and 1-D latent                    #
########################################################################
"""
```julia
_item_expected_information(parameters::Parameters1PL, latent::Latent1D)
```

# Description

It computes the expected information (-second derivative of the likelihood) with respect to the difficulty parameter of the 1PL model.
It follows the parametrization ``a(θ - b)``.

# Arguments

- **`parameters::Parameters1PL`** : Required. A 1-parameter logistic parameters object. 
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 

# Output

A `Float64` scalar. 
"""
function _item_expected_information(parameters::Parameters1PL, latent::Latent1D)
    p = _probability(latent, parameters)
    return p * (1 - p)::Float64
end

########################################################################
# Expected info -  2PL pars and 1-D latent                             #
########################################################################
"""
```julia
_item_expected_information(parameters::Parameters2PL, latent::Latent1D)
```

# Description

It computes the expected information (-second derivative of the likelihood) with respect to the 2 parameters of the 2PL model.
It follows the parametrization ``a(θ - b)``.

# Arguments

- **`parameters::Parameters1PL`** : Required. A 2-parameter logistic parameters object. 
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 

# Output

A ``2 \times 2`` matrix of the expected informations. 
"""
function _item_expected_information(parameters::Parameters2PL, latent::Latent1D)
    p = _probability(latent, parameters)
    p_1_p = (1 - p) * p
    i_aa = p_1_p * (latent.val - parameters.b)^2
    i_ab = -parameters.a * p_1_p * (latent.val - parameters.b)
    i_bb = parameters.a^2 * p_1_p
    return [i_aa i_ab; i_ab i_bb]::Matrix{Float64}
end

########################################################################
# Expected info -  for 3PL pars and 1-D latent                         #
########################################################################
"""
```julia
_item_expected_information(parameters::Parameters3PL, latent::Latent1D)
```

# Description

It computes the expected information (-second derivative of the likelihood) with respect to the 3 parameters of the 3PL model. 
It follows the parametrization ``a(θ - b)``.
It should be always positive.

# Arguments
- **`parameters::Parameters1PL`** : Required. A 3-parameter logistic parameters object. 
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 

# Output
A ``3 \times 3`` matrix of the expected informations. 
"""
function _item_expected_information(parameters::Parameters3PL, latent::Latent1D)
    p = _probability(latent, parameters)
    den = p * (1 - parameters.c)^2
    theta_b = latent.val - parameters.b
    p_c = p - parameters.c
    i_cc = (1 - p) / den # vv p. 51 Kim, Baker, 2008 : - E(L_33)
    i_aa = i_cc * p_c^2 * theta_b^2  # v p. 51 Kim, Baker, 2008
    i_ab = - i_aa / theta_b * parameters.a
    i_ac = i_cc * p_c * theta_b
    i_bc = i_cc * p_c * parameters.a
    i_bb = (i_bc ^ 2) / i_cc #v p. 51 Kim, Baker, 2008
    return [i_aa i_ab i_ac; i_ab i_bb i_bc; i_ac i_bc i_cc]::Matrix{Float64}
end