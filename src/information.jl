## 1D Latent Informations

"""
    information_latent(latent::Latent1D, parameters::Parameters1PL)

# Description
It computes the information (-second derivative of the likelihood) with respect to the 1-dimensional latent variable under the 1PL model.

# Arguments
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters1PL`** : Required. A 1-parameter logistic parameters object. 

# Output
A `Float64` scalar. 
"""
function information_latent(latent::Latent1D, parameters::Parameters1PL)
    p = probability(latent, parameters)
    return p * (1 - p)
end

"""
    information_latent(latent::Latent1D, parameters::Parameters2PL)

# Description
It computes the information (-second derivative of the likelihood) with respect to the 1-dimensional latent variable under the 2PL model.

# Arguments
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters2PL`** : Required. A 2-parameter logistic parameters object. 

# Output
A `Float64` scalar. 
"""
function information_latent(latent::Latent1D, parameters::Parameters2PL)
    p = probability(latent, parameters)
    return p * (1 - p) * parameters.a^2
end

"""
    information_latent(latent::Latent1D, parameters::Parameters3PL)

# Description
It computes the information (-second derivative of the likelihood) with respect to the 1-dimensional latent variable under the 3PL model.

# Arguments
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters3PL`** : Required. A 3-parameter logistic parameters object. 

# Output
A `Float64` scalar. 
"""
function information_latent(latent::Latent1D, parameters::Parameters3PL)
    p = probability(latent, parameters)
    return (p - parameters.c)^2 * (1 - p) * parameters.a^2 / (1 - parameters.c)^2 / p
end

## Item Expected Informations

"""
    expected_information_item(latent::Latent1D, parameters::Parameters1PL)

# Description
It computes the expected information (-second derivative of the likelihood) with respect to the difficulty parameter of the 1PL model.

# Arguments
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters1PL`** : Required. A 1-parameter logistic parameters object. 

# Output
A `Float64` scalar. 
"""
function  expected_information_item(latent::Latent1D, parameters::Parameters1PL)
    p = probability(latent, parameters)
    return (latent.val - parameters.b)^2 * p * (1 - p)
end

"""
    expected_information_item(latent::Latent1D, parameters::Parameters2PL)

# Description
It computes the expected information (-second derivative of the likelihood) with respect to the 2 parameters of the 2PL model.

# Arguments
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters1PL`** : Required. A 2-parameter logistic parameters object. 

# Output
A ``2 \\times 2`` matrix of the expected informations. 
"""
function  expected_information_item(latent::Latent1D, parameters::Parameters2PL)
    p = probability(latent, parameters)
    i_aa =                  (1 - p) * p *                         (latent.val - parameters.b)^2 
    i_ab = - parameters.a * (1 - p) * p *                         (latent.val - parameters.b)  
    i_bb = parameters.a^2 * (1 - p) * p *
    return [i_aa  i_ab; i_ab  i_bb]
end

"""
    expected_information_item(latent::Latent1D, parameters::Parameters3PL)

# Description
It computes the expected information (-second derivative of the likelihood) with respect to the 3 parameters of the 3PL model. 


# Arguments
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters1PL`** : Required. A 3-parameter logistic parameters object. 

# Output
A ``3 \\times 3`` matrix of the expected informations. 
"""
function expected_information_item(latent::Latent1D, parameters::Parameters3PL)
    p = probability(latent, parameters)
    den = p * (1 - parameters.c)^2
    i_aa =                  (1 - p) * (p - parameters.c)   * (latent.latent.val - parameters.b)^2 / den
    i_ab = - parameters.a * (1 - p) * (p - parameters.c)^2 * (latent.val - parameters.b)   / den
    i_ac = i_aa                     * (p - parameters.c)   * (latent.val - parameters.b)   / den
    i_bc = - parameters.a * (1 - p) * (p - parameters.c)                                       / den
    i_bb = - parameters.a * i_bc    * (p - parameters.c)
    i_cc =                  (1 - p)                                                            / den
    return [i_aa  i_ab  i_ac; i_ab  i_bb  i_bc; i_ac  i_bc  i_cc]
end

## Item Observed Informations

"""
    observed_information_item(response_val::Float64, latent::Latent1D, parameters::Parameters3PL)

# Description
It computes the observed information (-second derivative of the likelihood) with respect to the 3 parameters of the 3PL model. 

# Arguments
- **`response_val::Float64`** : Required. A scalar response. 
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters3PL`** : Required. A 3-parameter logistic parameters object. 

# Output
A ``3 \\times 3`` matrix of the observed informations. 
"""
function observed_information_item(response_val::Float64, latent::Latent1D, parameters::Parameters3PL)
    p = probability(latent, parameters)
    i = (1 - p) * (p - c)
    h = (response_val * parameters.c - p^2) * i
    j = response_val * i
    den = ((1 - parameters.c)*p)^2
    i_aa = - h * (latent.val - parameters.b)^2   / den
    i_ab = ((parameters.a * (latent.val - parameters.b) * h) + (p * (response_val - p) * (p - parameters.c))) / den
    i_ac = j * (latent.val - parameters.b) / den
    i_bc = parameters.a * j / den
    i_bb = - parameters.a^2 * h / den
    i_cc = ( response_val - 2*response_val*p + p^2) / den
    return [i_aa  i_ab  i_ac; i_ab  i_bb  i_bc; i_ac  i_bc  i_cc]
end

"""
    observed_information_item(response::AbstractResponse)

# Description
It computes the observed information (-second derivative of the likelihood) with respect to the 3 parameters of the 3PL model. 

# Arguments
- **`response::AbstractResponse`** : Required. An instance of the `AbstractResponse` struct. 

# Output
A ``3 \\times 3`` matrix of the observed informations. 
"""
function observed_information_item(response::AbstractResponse)
    observed_information_item(response.val, response.examinee.latent, response.item.parameters)
end

expected_information_item



