"""
     probability(latent_val::Float64, parameters::Parameters1PL)

# Description
It computes the probability (ICF) of a correct response for item `parameters` under the 1PL model at `latent_val` point.

# Arguments
- **`latent_val::Float64`** : Required. The point in the latent space in which compute the probability. 
- **`parameters::Parameters1PL`** : Required. A 1-parameter logistic parameters object. 

# Output
A `Float64` scalar. 
"""
function probability(latent_val::Float64, parameters::Parameters1PL)
    1 / (1 + _exp_c(parameters.b - latent_val))
end

"""
     probability(latent::Latent1D, parameters::Parameters1PL)

# Description
It computes the probability (ICF) of a correct response for item `parameters` under the 1PL model at `Latent1D` point.

# Arguments
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters1PL`** : Required. A 1-parameter logistic parameters object. 

# Output
A `Float64` scalar. 
"""
function probability(latent::Latent1D, parameters::Parameters1PL)
    probability(latent.val, parameters)
end

"""
    probability(latent::Latent1D, parameters::Parameters1PL, g_item::Vector{Float64}, g_latent::Vector{Float64})

# Description
It computes the probability (ICF) of a correct response for item `parameters` under the 1PL model at `Latent1D` point.
It updates the gradient vectors if they are not empty.

# Arguments
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters1PL`** : Required. A 1-parameter logistic parameters object. 

# Output
A `Float64` scalar. 
"""
function probability(latent::Latent1D, parameters::Parameters1PL, g_item::Vector{Float64}, g_latent::Vector{Float64})
    p = probability(latent, parameters)
    
    if size(g_item,1)>0
        g_item .= p * (1 - p)
    end
    
    if size(g_latent,1)>0
        g_latent .= - p * (1 - p)
    end
    
    return p
end

"""
     probability(latent_val::Float64, parameters::Parameters2PL)

# Description
It computes the probability (ICF) of a correct response for item `parameters` under the 2PL model at `latent_val` point.

# Arguments
- **`latent_val::Float64`** : Required. The point in the latent space in which compute the probability. 
- **`parameters::Parameters2PL`** : Required. A 2-parameter logistic parameters object. 

# Output
A `Float64` scalar. 
"""
function probability(latent_val::Float64, parameters::Parameters2PL)
    1 / (1 + _exp_c( - parameters.a * (latent_val - parameters.b)))
 end

"""
    probability(latent::Latent1D, parameters::Parameters2PL)

# Description
It computes the probability (ICF) of a correct response for item `parameters` under the 2PL model at `Latent1D` point.

# Arguments
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters2PL`** : Required. A 2-parameter logistic parameters object. 

# Output
A `Float64` scalar. 
"""
function probability(latent::Latent1D, parameters::Parameters2PL)
    probability(latent.val, parameters)
end

"""
    probability(latent::Latent1D, parameters::Parameters2PL, g_item::Vector{Float64}, g_latent::Vector{Float64})

# Description
It computes the probability (ICF) of a correct response for item `parameters` under the 2PL model at `Latent1D` point.
It updates the gradient vectors if they are not empty.

# Arguments
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters2PL`** : Required. A 2-parameter logistic parameters object. 

# Output
A `Float64` scalar. 
"""
function probability(latent::Latent1D, parameters::Parameters2PL, g_item::Vector{Float64}, g_latent::Vector{Float64})
    p = probability(latent, parameters)
    
    if size(g_item,1)>0 || size(g_latent,1)>0
        p1p = p * (1 - p)
        if size(g_item,1)>0
            g_item .= [(latent.val - parameters.b) * p1p ,- parameters.a * p1p ]
        end
        
        if size(g_latent,1)>0
            g_latent .= parameters.a * p1p
        end
    end
    
    return p
end

"""
    probability(latent_val::Float64, parameters::Parameters3PL)

# Description
It computes the probability (ICF) of a correct response for item `parameters` under the 3PL model at `latent_val` point.

# Arguments
- **`latent_val::Float64`** : Required. The point in the latent space in which compute the probability. 
- **`parameters::Parameters3PL`** : Required. A 3-parameter logistic parameters object. 

# Output
A `Float64` scalar. 
"""
function probability(latent_val::Float64, parameters::Parameters3PL)
    parameters.c + (1 - parameters.c) * (1 / (1 + _exp_c( - parameters.a * (latent_val - parameters.b ))))
end

"""
 probability(latent::Latent1D, parameters::Parameters3PL)

# Description
It computes the probability (ICF) of a correct response for item `parameters` under the 3PL model at `Latent1D` point.

# Arguments
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters3PL`** : Required. A 3-parameter logistic parameters object. 

# Output
A `Float64` scalar. 
"""
function probability(latent::Latent1D, parameters::Parameters3PL)
    probability(latent.val, parameters)
end

"""
    probability(latent::Latent1D, parameters::Parameters3PL,  g_item::Vector{Float64}, g_latent::Vector{Float64})

# Description
It computes the probability (ICF) of a correct response for item `parameters` under the 3PL model at `Latent1D` point.
It updates the gradient vectors if they are not empty.

# Arguments
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters3PL`** : Required. A 3-parameter logistic parameters object. 

# Output
A `Float64` scalar. 
"""
function probability(latent::Latent1D, parameters::Parameters3PL,  g_item::Vector{Float64}, g_latent::Vector{Float64})
    p = probability(latent, parameters)
    
    if size(g_item,1)>0 || size(g_latent,1)>0
        q1c = (1 - p) / (1 - parameters.c)
        
        if size(g_item,1)>0
            g_item .= [(latent.val - parameters.b) * q1c * (p - parameters.c), - parameters.a * q1c * (p - parameters.c) , q1c]
        end    
        
        #by Kim's book
        if size(g_latent,1)>0
            g_latent .= parameters.a * (p - parameters.c) * q1c
        end
        
    end
    
    return p
end