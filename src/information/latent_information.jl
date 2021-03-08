########################################################################
# Expected latent info -  for 1-D latent and 1PL pars                  #
########################################################################
"""
```julia
_latent_information(latent::Latent1D, parameters::Parameters1PL)
```

# Description

It computes the information (second derivative of the likelihood) with respect to the 1-dimensional latent variable under the 1PL model.
It follows the parametrization \$a(θ - b)\$.

# Arguments

- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters1PL`** : Required. A 1-parameter logistic parameters object. 

# Output

A `Float64` scalar. 
"""
function _latent_information(latent::Latent1D, parameters::Parameters1PL)
    p = _probability(latent, parameters)
    return p * (1 - p)::Float64
end

########################################################################
# Expected latent info -  for 1-D latent and 2PL pars                  #
########################################################################
"""
```julia
_latent_information(latent::Latent1D, parameters::Parameters2PL)
```

# Description

It computes the information (-second derivative of the likelihood) with respect to the 1-dimensional latent variable under the 2PL model.
It follows the parametrization \$a(θ - b)\$.

# Arguments

- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters2PL`** : Required. A 2-parameter logistic parameters object. 

# Output

A `Float64` scalar. 
"""
function _latent_information(latent::Latent1D, parameters::Parameters2PL)
    p = _probability(latent, parameters)
    return p * (1 - p) * parameters.a^2::Float64
end

########################################################################
# Expected latent info -  for 1-D latent and 3PL pars                  #
########################################################################
"""
```julia
_latent_information(latent::Latent1D, parameters::Parameters3PL)
```

# Description

It computes the information (-second derivative of the likelihood) with respect to the 1-dimensional latent variable under the 3PL model.
It follows the parametrization \$a(θ - b)\$.

# Arguments

- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters3PL`** : Required. A 3-parameter logistic parameters object. 

# Output

A `Float64` scalar. 
"""
function _latent_information(latent::Latent1D, parameters::Parameters3PL)
    p = _probability(latent, parameters)
    return (p - parameters.c)^2 * (1 - p) * parameters.a^2 / (1 - parameters.c)^2 / p::Float64
end