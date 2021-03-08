##########################################
#   1-dimensional latent and 1PL pars    #
##########################################
"""
```julia
__probability(latent_val::Float64, parameters::Parameters1PL)
```

# Description

It computes the probability (ICF) of a correct response for item `parameters` under the 1PL model at `latent_val` point.
It follows the parametrization \$a(θ - b)\$.

# Arguments

- **`latent_val::Float64`** : Required. The point in the latent space in which compute the probability. 
- **`parameters::Parameters1PL`** : Required. A 1-parameter logistic parameters object. 

# Output

A `Float64` scalar. 
"""
function __probability(latent_val::Float64, parameters::Parameters1PL)
   return 1 / (1 + _exp_c(parameters.b - latent_val))::Float64
end

"""
```julia
_probability(latent::Latent1D, parameters::Parameters1PL)
```

# Description

It computes the probability (ICF) of a correct response for item `parameters` under the 1PL model at `Latent1D` point.
It follows the parametrization \$a(θ - b)\$.

# Arguments

- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters1PL`** : Required. A 1-parameter logistic parameters object. 

# Output

A `Float64` scalar. 
"""
function _probability(latent::Latent1D, parameters::Parameters1PL)
   return __probability(latent.val, parameters)::Float64
end

"""
```julia
_probability(latent::Latent1D, parameters::Parameters1PL, g_item::Vector{Float64}, g_latent::Vector{Float64})
```

# Description

It computes the probability (ICF) of a correct response for item `parameters` under the 1PL model at `Latent1D` point.
It updates the gradient vectors if they are not empty.
It follows the parametrization \$a(θ - b)\$.

# Arguments

- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters1PL`** : Required. A 1-parameter logistic parameters object. 
- **`g_item::Vector{Float64}`** : Optional. Vector of gradients with respect to item parameters.
- **`g_latent::Vector{Float64}`** : Optional. Vector of gradients with respect to the 1-dimensional latent.

# Output

A `Float64` scalar. Updates gradient vectors.
"""
function _probability(
    latent::Latent1D,
    parameters::Parameters1PL,
    g_item::Vector{Float64},
    g_latent::Vector{Float64},
)
    p = _probability(latent, parameters)

    if size(g_item, 1) > 0
        g_item .= p * (1 - p)
    end

    if size(g_latent, 1) > 0
        g_latent .= -p * (1 - p)
    end

    return p::Float64
end

##########################################
#   1-dimensional latent and 2PL pars    #
##########################################
"""
```julia
__probability(latent_val::Float64, parameters::Parameters2PL)
```

# Description

It computes the probability (ICF) of a correct response for item `parameters` under the 2PL model at `latent_val` point.
It follows the parametrization \$a(θ - b)\$.

# Arguments

- **`latent_val::Float64`** : Required. The point in the latent space in which compute the probability. 
- **`parameters::Parameters2PL`** : Required. A 2-parameter logistic parameters object. 

# Output

A `Float64` scalar. 
"""
function __probability(latent_val::Float64, parameters::Parameters2PL)
    return _sig_c(parameters.a * (latent_val - parameters.b))::Float64
end

"""
```julia
_probability(latent::Latent1D, parameters::Parameters2PL)
```

# Description

It computes the probability (ICF) of a correct response for item `parameters` under the 2PL model at `Latent1D` point.
It follows the parametrization \$a(θ - b)\$.

# Arguments
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters2PL`** : Required. A 2-parameter logistic parameters object. 

# Output

A `Float64` scalar. 
"""
function _probability(latent::Latent1D, parameters::Parameters2PL)
    return __probability(latent.val, parameters)::Float64
end

"""
```julia
_probability(latent::Latent1D, parameters::Parameters2PL, g_item::Vector{Float64}, g_latent::Vector{Float64})
```

# Description

It computes the probability (ICF) of a correct response for item `parameters` under the 2PL model at `Latent1D` point.
It updates the gradient vectors if they are not empty.
It follows the parametrization \$a(θ - b)\$.

# Arguments

- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters2PL`** : Required. A 2-parameter logistic parameters object. 
- **`g_item::Vector{Float64}`** : Optional. Vector of gradients with respect to item parameters.
- **`g_latent::Vector{Float64}`** : Optional. Vector of gradients with respect to the 1-dimensional latent.

# Output

A `Float64` scalar. Updates the gradient vectors.

# Example

_Probability of a correct response is computed for each examinee and each item._

"""
function _probability(
    latent::Latent1D,
    parameters::Parameters2PL,
    g_item::Vector{Float64},
    g_latent::Vector{Float64},
)
    p = _probability(latent, parameters)

    if size(g_item, 1) > 0 || size(g_latent, 1) > 0
        p1p = p * (1 - p)
        if size(g_item, 1) > 0
            g_item .= [(latent.val - parameters.b) * p1p, -parameters.a * p1p]
        end

        if size(g_latent, 1) > 0
            g_latent .= parameters.a * p1p
        end
    end

    return p::Float64
end


##########################################
#   1-dimensional latent and 3PL pars    #
##########################################
"""
```julia
__probability(latent_val::Float64, parameters::Parameters3PL)
```

# Description

It computes the probability (ICF) of a correct response for item `parameters` under the 3PL model at `latent_val` point.
It follows the parametrization \$a(θ - b)\$.

# Arguments

- **`latent_val::Float64`** : Required. The point in the latent space in which compute the probability. 
- **`parameters::Parameters3PL`** : Required. A 3-parameter logistic parameters object. 

# Output

A `Float64` scalar. 
"""
function __probability(latent_val::Float64, parameters::Parameters3PL)
    return parameters.c +
    (1 - parameters.c) * (1 / (1 + _exp_c(-parameters.a * (latent_val - parameters.b))))::Float64
end

"""
```julia
_probability(latent::Latent1D, parameters::Parameters3PL)
```

# Description

It computes the probability (ICF) of a correct response for item `parameters` under the 3PL model at `Latent1D` point.
It follows the parametrization \$a(θ - b)\$.

# Arguments

- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters3PL`** : Required. A 3-parameter logistic parameters object. 

# Output

A `Float64` scalar. 
"""
function _probability(latent::Latent1D, parameters::Parameters3PL)
    return __probability(latent.val, parameters)::Float64
end

"""
```julia
_probability(latent::Latent1D, parameters::Parameters3PL,  g_item::Vector{Float64}, g_latent::Vector{Float64})
```

# Description

It computes the probability (ICF) of a correct response for item `parameters` under the 3PL model at `Latent1D` point.
It updates the gradient vectors if they are not empty.
It follows the parametrization \$a(θ - b)\$.

# Arguments

- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 
- **`parameters::Parameters3PL`** : Required. A 3-parameter logistic parameters object. 
- **`g_item::Vector{Float64}`** : Optional. Vector of gradients with respect to item parameters.
- **`g_latent::Vector{Float64}`** : Optional. Vector of gradients with respect to the 1-dimensional latent.

# Output

A `Float64` scalar. Updates the gradient vectors.
"""
function _probability(
    latent::Latent1D,
    parameters::Parameters3PL,
    g_item::Vector{Float64},
    g_latent::Vector{Float64},
)
    p = _probability(latent, parameters)

    if size(g_item, 1) > 0 || size(g_latent, 1) > 0
        q1c = (1 - p) / (1 - parameters.c)

        if size(g_item, 1) > 0
            g_item .= [
                (latent.val - parameters.b) * q1c * (p - parameters.c),
                -parameters.a * q1c * (p - parameters.c),
                q1c,
            ]
        end

        #by Kim's book
        if size(g_latent, 1) > 0
            g_latent .= parameters.a * (p - parameters.c) * q1c
        end

    end

    return p::Float64
end

##########################################
#   N-dimensional latent and NPL pars    #
##########################################
"""
```julia
__probability(latent_vals::Vector{Float64}, parameters::ParametersNPL)
```

# Description

It computes the probability (ICF) of a correct response for item `parameters` under the 3PL model at `latent_vals` points.
N-dimensional latent.
It follows the parametrization \$a(θ - b)\$.

# Arguments

- **`latent_vals::Vector{Float64}`** : Required. The points in the latent space in which compute the probability. 
- **`parameters::ParametersNPL`** : Required. A 2-parameter logistic parameters object suitbale for N-dimensional latent. 

# Output

A `Float64` scalar. 
"""
function __probability(latent_vals::Vector{Float64}, parameters::ParametersNPL)
    return _sig_c(latent_vals' * parameters.a - parameters.b)::Float64
end


"""
```julia
_probability(latent::LatentND, parameters::ParametersNPL)
```

# Description

It computes the probability (ICF) of a correct response for item `parameters` under the 3PL model at `Latent1D` point.
It updates the gradient vectors if they are not empty.
It follows the parametrization \$a(θ - b)\$.

# Arguments

- **`latent::LatentND`** : Required. A N-dimensional latent variable. 
- **`parameters::ParametersNPL`** : Required. A 2-parameter logistic parameters object suitable for a N-dimensional latent. 

# Output

A `Float64` scalar. 
"""
function _probability(latent::LatentND, parameters::ParametersNPL)
    return _sig_c(latent.val' * parameters.a - parameters.b)::Float64
end

##########################################
#   Externals  for examinees and items   #
##########################################

"""
```julia
probability(examinee::AbstractExaminee, item::AbstractItem)
```

# Description

It computes the probability (ICF) that an `examinee` answers correctly at `item`.

# Arguments

- **`examinee::AbstractExaminee`** : Required. An Examinee. 
- **`item::AbstractItem`** : Required. An Item. 

# Output

A `Float64` scalar. 
"""
function probability(examinee::AbstractExaminee, item::AbstractItem)
    return _probability(examinee.latent, item.parameters)::Float64
end


"""
```julia
probability(examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})
```

# Description

It computes the probability (ICF) that a vector of `examinees` answers correctly at a vector of `items`.

# Arguments

- **`examinees::Vector{<:AbstractExaminee}`** : Required. A vector of `Examinee`s. 
- **`items::Vector{<:AbstractItem}`** : Required. A vector of `Item`s. 

# Output

A I x N `Float64` matrix. 
"""
function probability(examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})
    return mapreduce(e -> map(i -> probability(e, i), items), hcat, examinees)::Matrix{Float64}
end

"""
```julia
probability(items::Vector{<:AbstractItem}, examinees::Vector{<:AbstractExaminee})
```

# Description

It computes the probability (ICF) that a vector of `examinees` answers correctly at a vector of `items`.

# Arguments

- **`examinees::Vector{<:AbstractExaminee}`** : Required. A vector of `Examinee`s. 
- **`items::Vector{<:AbstractItem}`** : Required. A vector of `Item`s. 

# Output

A I x N `Float64` matrix. 
"""
function probability(items::Vector{<:AbstractItem}, examinees::Vector{<:AbstractExaminee})
    return probability(examinees, items)::Matrix{Float64}
end

##########################################
#         Externals for matrix data      #
##########################################
"""
```julia
probability(parameters_matrix::Matrix{Float64}, latent_matrix::Matrix{Float64})
```

# Description

It computes the probability (ICF) of a correct response for item parameters and latents values provided in matrix form.
Not suitable for 3PL models, for such a kind of model use probability_3PL().
It follows the parametrization ``a \\theta - b ``.

# Arguments

- **`parameters_matrix::Matrix{Float64}`** : Required. A `I x (n_latents + 1)` matrix with item parameters. intercept (b) must be in first column, latents coefficients (a_j) in next columns (2, ..., n_latents + 1). 
- **`latent_matrix::Matrix{Float64}`** : Required. A `n_latents x N` or `(n_latents + 1) x N` matrix with latents values. 

# Output

A `I x N` `Float64` matrix. 
"""
function probability(parameters_matrix::Matrix{Float64}, latents_matrix::Matrix{Float64})
    if size(latents_matrix, 1) == (size(parameters_matrix, 2)-1)
        latents_matrix = vcat(.-ones(Float64, size(latents_matrix, 2))', latents_matrix)
    elseif size(latents_matrix, 1) != size(parameters_matrix, 2)
        error("The latents matrix must have n_pars-1 columns (the intercept is automatically added) or n_pars columns.")
    end
    return _sig_c.(_gemmblasAB(parameters_matrix, latents_matrix))::Matrix{Float64}
end

"""
```julia
probability_3PL(parameters_matrix::Matrix{Float64}, latent_matrix::Matrix{Float64})
```

# Description

Only for models which has guessing parameter (c) in last row of parameters_matrix. It computes the probability (ICF) of a correct response for item parameters and latents values provided in matrix form.
It follows the parametrization ``a \\theta - b ``.

# Arguments

- **`parameters_matrix::Matrix{Float64}`** : Required. A `I x (n_latents + 1)` matrix with item parameters. intercept (b) must be in first column, latents coefficients (a_j) in next columns (2, ..., n_latents + 1). 
- **`latent_matrix::Matrix{Float64}`** : Required. A `n_latents x N` matrix with latents values. 

# Output

A `I x N` `Float64` matrix. 
"""
function probability_3PL(
    parameters_matrix::Matrix{Float64},
    latents_matrix::Matrix{Float64},
)
    parameters_core = parameters_matrix[:, 1:(end-1)]
    if size(latents_matrix, 1) == (size(parameters_core, 2)-1)
        latents_matrix = vcat(ones(Float64, size(latents_matrix, 2))', latents_matrix)
    elseif size(latents_matrix, 1) != size(parameters_core, 2)
        error("The latents matrix must have n_pars-2 columns (the intercept is automatically added) or n_pars-1 columns.")
    end
    ret = _sig_c.(_gemmblasAB(parameters_core, latents_matrix))
    return _matrix_rows_vec(ret, parameters_matrix[:, end], (x, y) -> y + (1 - y) * x)::Matrix{Float64}
end