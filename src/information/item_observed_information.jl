########################################################################
# Observed info -  for 2PL pars and 1-D latent                         #
########################################################################
"""
```julia
_item_observed_information(
    parameters::Parameters1PL,
    latent::Latent1D,
    response_val::Float64,
)
```

# Description

It is equal to the item expected information.

# Arguments

- **`parameters::Parameters1PL`** : Required. A 1-parameter logistic parameters object. 
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`response_val::Float64`** : Required. A scalar response. 

# Output
A `Float64` scalar observed (expected) informations. 
"""
function _item_observed_information(
    parameters::Parameters1PL,
    latent::Latent1D,
    response_val::Float64
)
    return _item_expected_information(parameters, latent)::Matrix{Float64}
end

########################################################################
# Observed info -  for 2PL pars and 1-D latent                         #
########################################################################
"""
```julia
_item_observed_information(
    parameters::Parameters2PL,
    latent::Latent1D,
    response_val::Float64,
)
```

# Description

It is equal to the item expected information.

# Arguments

- **`parameters::Parameters2PL`** : Required. A 2-parameter logistic parameters object. 
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`response_val::Float64`** : Required. A scalar response. 

# Output
A ``2 \\times 2`` matrix of the observed (expected) informations. 
"""
function _item_observed_information(
    parameters::Parameters2PL,
    latent::Latent1D,
    response_val::Float64
)
    return _item_expected_information(parameters, latent)::Matrix{Float64}
end

########################################################################
# Observed info -  for 3PL pars and 1-D latent                         #
########################################################################
"""
```julia
_item_observed_information(
    parameters::Parameters3PL,
    latent::Latent1D,
    response_val::Float64,
)
```

# Description

It computes the observed information (-second derivative of the likelihood) with respect to the 3 parameters of the 3PL model. 
It follows the parametrization \$a(θ - b)\$.

# Arguments

- **`parameters::Parameters3PL`** : Required. A 3-parameter logistic parameters object. 
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`response_val::Float64`** : Required. A scalar response. 

# Output
A ``3 \\times 3`` matrix of the observed informations. 
"""
function _item_observed_information(
    parameters::Parameters3PL,
    latent::Latent1D,
    response_val::Float64,
)
    p = _probability(latent, parameters)
    i = (1 - p) * (p - parameters.c)
    h = (response_val * parameters.c - p^2) * i
    j = response_val * i
    den = ((1 - parameters.c) * p)^2
    i_aa = - h * (latent.val - parameters.b)^2 / den # v p.49 Kim, Baker: - L_11
    i_ab =
        (
            (p - parameters.c) * ((parameters.a * (latent.val - parameters.b) * h) +
            (p * (response_val - p) * (1 - parameters.c)))
        ) / den # X p.50 Kim, Baker: - L_12 #TODO
    i_ac = j * (latent.val - parameters.b) / den # v p.51 Kim, Baker: - L_13
    i_bc = - parameters.a * j / den # v p.51 Kim, Baker: - L_23
    i_bb = - parameters.a^2 * h / den # v p.49 Kim, Baker: - L_22
    i_cc = (response_val - 2 * response_val * p + p^2) / den # X p.50 Kim, Baker: - L_33
    return [i_aa i_ab i_ac; i_ab i_bb i_bc; i_ac i_bc i_cc]::Matrix{Float64}
end