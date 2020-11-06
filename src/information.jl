## 1D Latent Informations


"""
    information_latent(latents_matrix::Matrix{Float64}, parameters_matrix::Matrix{Float64})

# Description
It computes the information function (IIF) for item parameters at latents values provided in matrix form.
Not suitable for 3PL models, for such a kind of model use information_latent_3PL().
It follows the parametrization \$aθ - b\$.

# Arguments
- **`latents_matrix::Matrix{Float64}`** : Required. A `n_latents x N` matrix with latents values. 
- **`parameters_matrix::Matrix{Float64}`** : Required. A `(n_latents + 1) x I` matrix with item parameters. intercept (b) must be in first row, latents coefficients (a_j) in next rows (2, ..., n_latents + 1). 
    
# Output
A `I x N` `Float64` matrix. 
"""
function information_latent(
    latents_matrix::Matrix{Float64},
    parameters_matrix::Matrix{Float64},
)
    p = probability(parameters_matrix, latents_matrix)
    if size(parameters_matrix, 2) > 1
        return _matrix_cols_vec(p, parameters_matrix[:, end], (x, y) -> _p1p(x) * y^2)
    else
        return _p1p.(p)
    end
end

"""
    information_latent_3PL(latents_matrix::Matrix{Float64}, parameters_matrix::Matrix{Float64})

# Description
Only for models which has guessing parameter (c) in last row of parameters_matrix.
It computes the information function (IIF) for item parameters at latents values provided in matrix form.
It follows the parametrization \$aθ - b\$.

# Arguments
- **`latents_matrix::Matrix{Float64}`** : Required. A `n_latents x N` matrix with latents values. 
- **`parameters_matrix::Matrix{Float64}`** : Required. A `(n_latents + 1) x I` matrix with item parameters. intercept (b) must be in first row, latents coefficients (a_j) in next rows (2, ..., n_latents + 1). 

# Output
A `I x N` `Float64` matrix. 
"""
function information_latent_3PL(
    latents_matrix::Matrix{Float64},
    parameters_matrix::Matrix{Float64},
)
    p = probability_3PL(parameters_matrix, latents_matrix)
    return _matrix_cols_vec(p, parameters_matrix[:, 2], (x, y) -> y^2 * (1 - x) * x) .*
           _matrix_cols_vec(p, parameters_matrix[:, end], (x, y) -> ((x - y) / (1 - y))^2)
end

"""
    information_latent(latent::Latent1D, parameters::Parameters1PL)

# Description
It computes the information (-second derivative of the likelihood) with respect to the 1-dimensional latent variable under the 1PL model.
It follows the parametrization \$a(θ - b)\$.

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
It follows the parametrization \$a(θ - b)\$.

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
It follows the parametrization \$a(θ - b)\$.

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

"""
    information_latent(examinee::AbstractExaminee, items::Vector{<:AbstractItem})

# Description
An abstraction of `information_latent(latent::AbstractLatent, parameters::AbstractParameters)` on an examinee and items.
It follows the parametrization \$a(θ - b)\$.

# Arguments
- **`examinee::AbstractExaminee`** : Required. 
- **`items::Vector{<:AbstractItem}`** : Required. 

"""
function information_latent(
    examinee::AbstractExaminee,
    items::Vector{<:AbstractItem},
)
    [information_latent(examinee.latent, i.parameters) for i in items]
end

"""
    information_latent(examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})

# Description
An abstraction of `information_latent(latent::AbstractLatent, parameters::AbstractParameters)` on examinees and items.
It follows the parametrization \$a(θ - b)\$.

# Arguments
- **`examinees::Vector{<:AbstractExaminee}`** : Required. 
- **`items::Vector{<:AbstractItem}`** : Required. 

"""
function information_latent(
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
)
    [information_latent(e.latent, i.parameters) for i in items, e in examinees]
end

## Item Expected Informations

"""
    expected_information_item(parameters::Parameters1PL, latent::Latent1D)

# Description
It computes the expected information (-second derivative of the likelihood) with respect to the difficulty parameter of the 1PL model.
It follows the parametrization \$a(θ - b)\$.

# Arguments
- **`parameters::Parameters1PL`** : Required. A 1-parameter logistic parameters object. 
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 

# Output
A `Float64` scalar. 
"""
function expected_information_item(parameters::Parameters1PL, latent::Latent1D)
    p = probability(latent, parameters)
    return (latent.val - parameters.b)^2 * p * (1 - p)
end

"""
    expected_information_item(parameters::Parameters2PL, latent::Latent1D)

# Description
It computes the expected information (-second derivative of the likelihood) with respect to the 2 parameters of the 2PL model.
It follows the parametrization \$a(θ - b)\$.

# Arguments
- **`parameters::Parameters1PL`** : Required. A 2-parameter logistic parameters object. 
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 

# Output
A ``2 \\times 2`` matrix of the expected informations. 
"""
function expected_information_item(parameters::Parameters2PL, latent::Latent1D)
    p = probability(latent, parameters)
    i_aa = (1 - p) * p * (latent.val - parameters.b)^2
    i_ab = -parameters.a * (1 - p) * p * (latent.val - parameters.b)
    i_bb = parameters.a^2 * (1 - p) * p
    return [i_aa i_ab; i_ab i_bb]
end

"""
    expected_information_item(parameters::Parameters3PL, latent::Latent1D)

# Description
It computes the expected information (-second derivative of the likelihood) with respect to the 3 parameters of the 3PL model. 
It follows the parametrization \$a(θ - b)\$.

# Arguments
- **`parameters::Parameters1PL`** : Required. A 3-parameter logistic parameters object. 
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 

# Output
A ``3 \\times 3`` matrix of the expected informations. 
"""
function expected_information_item(parameters::Parameters3PL, latent::Latent1D)
    p = probability(latent, parameters)
    den = p * (1 - parameters.c)^2
    i_aa = (1 - p) * (p - parameters.c) * (latent.val - parameters.b)^2 / den
    i_ab =
        -parameters.a * (1 - p) * (p - parameters.c)^2 * (latent.val - parameters.b) / den
    i_ac = i_aa * (p - parameters.c) * (latent.val - parameters.b) / den
    i_bc = -parameters.a * (1 - p) * (p - parameters.c) / den
    i_bb = -parameters.a * i_bc * (p - parameters.c)
    i_cc = (1 - p) / den
    return [i_aa i_ab i_ac; i_ab i_bb i_bc; i_ac i_bc i_cc]::Matrix{Float64}
end

"""
    expected_information_item(items::Vector{<:AbstractItem}, examinees::Vector{<:AbstractExaminee})

# Description
Abstraction of expected_information_item(latent, parameters) on Vector{<:AbstractExaminee} and items::Vector{<:AbstractItem}.
It follows the parametrization \$a(θ - b)\$.

# Arguments
- **`items::Vector{<:AbstractItem}`** : Required. Size I x 1.
- **`examinees::Vector{<:AbstractExaminee}`** : Required. Size N x 1.

# Output
A \$N x I\$ matrix of the expected informations matrices. 
"""
function expected_information_item(
    items::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
)
    [expected_information_item(i.parameters, e.latent) for i in items, e in examinees]::Matrix{Matrix{Float64}}
end

## Item Observed Informations

"""
    observed_information_item(parameters::Parameters3PL, latent::Latent1D, response_val::Float64)

# Description
It computes the observed information (-second derivative of the likelihood) with respect to the 3 parameters of the 3PL model. 
It follows the parametrization \$a(θ - b)\$.

# Arguments
- **`parameters::Parameters3PL`** : Required. A 3-parameter logistic parameters object. 
- **`response_val::Float64`** : Required. A scalar response. 
- **`latent::Latent1D`** : Required. A 1-dimensional `Latent1D` latent variable. 

# Output
A ``3 \\times 3`` matrix of the observed informations. 
"""
function observed_information_item(
    parameters::Parameters3PL,
    latent::Latent1D,
    response_val::Float64,
)
    p = probability(latent, parameters)
    i = (1 - p) * (p - parameters.c)
    h = (response_val * parameters.c - p^2) * i
    j = response_val * i
    den = ((1 - parameters.c) * p)^2
    i_aa = -h * (latent.val - parameters.b)^2 / den
    i_ab =
        (
            (parameters.a * (latent.val - parameters.b) * h) +
            (p * (response_val - p) * (p - parameters.c))
        ) / den
    i_ac = j * (latent.val - parameters.b) / den
    i_bc = parameters.a * j / den
    i_bb = -parameters.a^2 * h / den
    i_cc = (response_val - 2 * response_val * p + p^2) / den
    return [i_aa i_ab i_ac; i_ab i_bb i_bc; i_ac i_bc i_cc]
end

"""
    observed_information_item(items::Vector{<:AbstractItem}, examinees::Vector{<:AbstractExaminee}, responses::Vector{<:AbstractResponse})

# Description
Abstraction to response, item and examinee of `observed_information_item(response_val::Float64, latent::Latent1D, parameters::Parameters3PL)`.
It follows the parametrization \$a(θ - b)\$.
"""
function observed_information_item(
    items::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{<:AbstractResponse},
)
    mapreduce(responses, +) do r
        observed_information_item(
            r,
            get_examinee_by_id(r.examinee_id, examinees),
            get_item_by_id(r.item_id, items),
        )
    end
end

"""
    D_method(item::Item1PL, examinee::AbstractExaminee)

# Description
It calls the function `expected_information_item` and returns its result.

# Arguments
  - **`item::Item1PL`**: The 1PL item.
  - **`examinee::AbstractExaminee`**: The examinee at which computing the information.

# Output
It returns a `Float64` scalar.
"""
function D_method(item::Item1PL, examinee::AbstractExaminee)
    return expected_information_item(item.parameters, examinee.latent)
end

"""
    D_method(item::Item2PL, examinee::AbstractExaminee)

# Description
Computes the determinant of the expected information matrix for a 2PL item and a generic type examinee.

# Arguments
  - **`item::Item2PL`**: The 2PL item.
  - **`examinee::AbstractExaminee`**: The examinee at which computing the information.

# Output
It returns a `Float64` scalar.
"""
function D_method(item::Item2PL, examinee::AbstractExaminee)
    return LinearAlgebra.det(expected_information_item(item.parameters, examinee.latent))
end

"""
    D_method(item::Item3PL, examinee::AbstractExaminee)

# Description
Computes the determinant of the expected information matrix for a 3PL item and a generic type examinee.

# Arguments
  - **`item::Item3PL`**: The 3PL item.
  - **`examinee::AbstractExaminee`**: The examinee at which computing the information.

# Output
It returns a `Float64` scalar.
"""
function D_method(item::Item3PL, examinee::AbstractExaminee)
    return LinearAlgebra.det(expected_information_item(item.parameters, examinee.latent))
end


"""
    A_method(item::Item1PL, examinee::AbstractExaminee)

# Description
It calls the function `expected_information_item` and returns its result.

# Arguments
  - **`item::Item1PL`**: The 1PL item.
  - **`examinee::AbstractExaminee`**: The examinee at which computing the information.

# Output
It returns a `Float64` scalar.
"""
function A_method(item::Item1PL, examinee::AbstractExaminee)
    return expected_information_item(item.parameters, examinee.latent)
end

"""
    A_method(item::Item2PL, examinee::AbstractExaminee)

# Description
Computes the trace of the expected information matrix for a 2PL item and a generic type examinee.

# Arguments
  - **`item::Item2PL`**: The 2PL item.
  - **`examinee::AbstractExaminee`**: The examinee at which computing the information.

# Output
It returns a `Float64` scalar.
"""
function A_method(item::Item2PL, examinee::AbstractExaminee)
    return LinearAlgebra.tr(expected_information_item(item.parameters, examinee.latent))
end

"""
    A_method(item::Item3PL, examinee::AbstractExaminee)

# Description
Computes the trace of the expected information matrix for a 3PL item and a generic type examinee.

# Arguments
  - **`item::Item3PL`**: The 3PL item.
  - **`examinee::AbstractExaminee`**: The examinee at which computing the information.

# Output
It returns a `Float64` scalar.
"""
function A_method(item::Item3PL, examinee::AbstractExaminee)
    return LinearAlgebra.det(expected_information_item(item.parameters, examinee.latent))
end


"""
    D_gain_method(item::Item1PL, examinee::AbstractExaminee)

# Description
It calls the function `expected_information_item` and returns its result.

# Arguments
  - **`item::Item1PL`**: The 1PL item.
  - **`examinee::AbstractExaminee`**: The examinee at which computing the information.

# Output
It returns a `Float64` scalar.
"""
function D_gain_method(item::Item1PL, examinee::AbstractExaminee)
    old_exp_info = copy(item.parameters.expected_information)
    return expected_information_item(item.parameters, examinee.latent)
end

"""
    D_gain_method(item::Item2PL, examinee::AbstractExaminee)

# Description
Computes the trace of the expected information matrix for a 2PL item and a generic type examinee.

# Arguments
  - **`item::Item2PL`**: The 2PL item.
  - **`examinee::AbstractExaminee`**: The examinee at which computing the information.

# Output
It returns a `Float64` scalar.
"""
function D_gain_method(item::Item2PL, examinee::AbstractExaminee)
    old_exp_info = copy(item.parameters.expected_information)
    return LinearAlgebra.det(old_exp_info + expected_information_item(item.parameters, examinee.latent)) - LinearAlgebra.det(old_exp_info)
end

"""
    D_gain_method(item::Item3PL, examinee::AbstractExaminee)

# Description
Computes the trace of the expected information matrix for a 3PL item and a generic type examinee.

# Arguments
  - **`item::Item3PL`**: The 3PL item.
  - **`examinee::AbstractExaminee`**: The examinee at which computing the information.

# Output
It returns a `Float64` scalar.
"""
function D_gain_method(item::Item3PL, examinee::AbstractExaminee)
    old_exp_info = copy(item.parameters.expected_information)
    return LinearAlgebra.det(old_exp_info + expected_information_item(item.parameters, examinee.latent)) - LinearAlgebra.det(old_exp_info)
end