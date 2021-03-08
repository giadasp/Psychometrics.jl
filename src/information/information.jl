include("item_expected_information.jl")
include("item_observed_information.jl")
include("latent_information.jl")

##########################################
#   Latent information                   #
##########################################
"""
```julia
latent_information(examinee::AbstractExaminee, item::AbstractItem)
```

# Description

An abstraction of `_latent_information(latent::AbstractLatent, parameters::AbstractParametersBinary)` on an examinee and an item.
It follows the parametrization \$a(θ - b)\$.

# Arguments

- **`examinee::AbstractExaminee`** : Required. 
- **`item::AbstractItem`** : Required. 

# Output
A `Float64` scalar.
"""
function latent_information(examinee::AbstractExaminee, item::AbstractItem)
    _latent_information(examinee.latent, item.parameters)::Float64
end

"""
```julia
latent_information(examinee::AbstractExaminee, items::Vector{<:AbstractItem})
```

# Description

An abstraction of `_latent_information(latent::AbstractLatent, parameters::AbstractParametersBinary)` on an examinee and items.
It follows the parametrization \$a(θ - b)\$.

# Arguments

- **`examinee::AbstractExaminee`** : Required. 
- **`items::Vector{<:AbstractItem}`** : Required. 

# Output
A `Float64` vector.

# Example

Compute the Fisher information for the latent/s of the examinees (second derivatives of the likelihood with respect to ``\\theta``) and each item.

``E_\\theta_n \\[ I(\\theta_n | b_i) \\] ``

```julia
examinee = Examinee(); #default examinee factory
items = [Item() for i = 1 : 30]; #default item factory
latent_information(examinee, items) #compute the information wrt θ   
```
"""
function latent_information(examinee::AbstractExaminee, items::Vector{<:AbstractItem})
    mapreduce( i -> _latent_information(examinee.latent, i.parameters), vcat, items)::Vector{Float64}
end

"""
```julia
latent_information(
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
)
```

# Description

An abstraction of `_latent_information(latent::AbstractLatent, parameters::AbstractParametersBinary)` on examinees and items.
It follows the parametrization \$a(θ - b)\$.

# Arguments

- **`examinees::Vector{<:AbstractExaminee}`** : Required. 
- **`items::Vector{<:AbstractItem}`** : Required. 

# Output
A ``N \\times I`` matrix of latent informations (matrices or scalars). 

```@example
examinees = [Examinee(); for n = 1 : 100]; #default examinee factory
items = [Item() for i = 1 : 30]; #default item factory
latent_information(examinees, items) #compute the information wrt θs   
```
"""
function latent_information(
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
)
    return [_latent_information(e.latent, i.parameters) for i in items, e in examinees]
end

##########################################
#   Item expected information            #
##########################################
"""
```julia
item_expected_information(
    item::AbstractItem,
    examinee::AbstractExaminee,
)
```

# Description

Abstraction of _item_expected_information(latent, parameters) on Vector{<:AbstractExaminee} and items::Vector{<:AbstractItem}.
It follows the parametrization \$a(θ - b)\$.

# Arguments

- **`item::AbstractItem`** : Required. 
- **`examinee::AbstractExaminee`** : Required.

# Output
A matrix of expected informations (matrices for 2PL and 3PL or scalars 1PL). 
"""
function item_expected_information(
    item::AbstractItem,
    examinee::AbstractExaminee,
)
    return _item_expected_information(item.parameters, examinee.latent)
end

"""
```julia
item_expected_information(
    items::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
)
```

# Description

Abstraction of _item_expected_information(latent, parameters) on Vector{<:AbstractExaminee} and items::Vector{<:AbstractItem}.
It follows the parametrization \$a(θ - b)\$.

# Arguments

- **`items::Vector{<:AbstractItem}`** : Required. Size I x 1.
- **`examinees::Vector{<:AbstractExaminee}`** : Required. Size N x 1.

# Output
A ``N \\times I`` matrix of expected informations (matrices or scalars). 

```@example
examinee = Examinee(1); # default examinee random factory (1-D latent)
item_1PL = Item(1, Parameters1PL()); # default 1PL item random factory
response_1PL = answer(examinee, item_1PL) # generate responses
item_exp_info_1PL = item_expected_information(item_1PL, examinee)

item_2PL = Item(1, Parameters2PL()); # default 2PL item random factory
response_2PL = answer(examinee, item_2PL) # generate responses
item_exp_info_2PL = item_expected_information(item_2PL, examinee)
"""
function item_expected_information(
    items::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
)
    return [
        _item_expected_information(i.parameters, e.latent) for i in items, e in examinees
    ]
end

##########################################
#   Item observed information            #
##########################################

"""
```julia
item_observed_information(
    item::AbstractItem,
    examinee::AbstractExaminee,
    response::AbstractResponse,
)
```

# Description

It computes the item observed information for an `examinee` who answered `response` .

```@example
examinee = Examinee(1); # default examinee random factory (1-D latent)
item_2PL = Item(1, Parameters2PL()); # default 2PL item random factory
response_2PL = answer(examinee, item_2PL) # generate responses
item_obs_info_2PL = item_observed_information(item_2PL, examinee, response_2PL)
item_exp_info_2PL = item_expected_information(item_2PL, examinee)
# for 2PL and 1PL item parameters, the observed information is equal to the expected information.
@assert item_obs_info_2PL == item_exp_info_2PL

examinee = Examinee(1); # default examinee random factory (1-D latent)
item_3PL = Item(1, Parameters3PL()); # default 3PL item random factory
response_3PL = answer(examinee, item_3PL) # generate responses
# for 3PL item parameters, the observed information is different from the expected information.
item_obs_info_3PL = item_observed_information(item_3PL, examinee, response_3PL)
item_exp_info_3PL = item_expected_information(item_3PL, examinee)
```
"""
function item_observed_information(
    item::AbstractItem,
    examinee::AbstractExaminee,
    response::AbstractResponse,
)
    return  _item_observed_information(item.parameters, examinee.latent, response.val)
end

"""
```julia
item_observed_information(
    items::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{<:AbstractResponse},
)
```

# Description

It computes the sum of the item observed informations across the responses.
Items (latents) must be of the same type.

```@example
examinees = [Examinee(n) for n = 1 : 100]; #default examinee factory (1-D latent)
items_2PL = [Item(i, Parameters2PL()) for i = 1 : 30]; #2PL item factory
items_3PL = [Item(i + 30, string(i + 30), Parameters3PL()) for i = 1 : 30]; #3PL item factory, id and idx must be different
items = vcat([items_2PL, items_3PL]...) # create vector of items

responses = answer(examinees, items) # generate responses
#obs_item_info = item_observed_information(items, examinees, responses) # do not run
#^ This returns an error since Julia cannot sum ``2 \\times 2`` matrices with ``3 \\times 3`` matrices.
# Instead do this ⌄.
obs_items_info_2PL = item_observed_information(items_2PL, examinees, responses)
obs_items_info_3PL = item_observed_information(items_3PL, examinees, responses)
```
"""
function item_observed_information(
    items::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{<:AbstractResponse},
)
    return mapreduce( r ->
        item_observed_information(
            r,
            get_examinee_by_id(r.examinee_id, examinees),
            get_item_by_id(r.item_id, items),
        ), 
        +,
        responses)
end

##########################################
#   Externals for matrix data            #
##########################################
"""
```julia
latent_information(
    latents_matrix::Matrix{Float64},
    parameters_matrix::Matrix{Float64},
)
```

# Description

It computes the information function (IIF) for item parameters at latents values provided in matrix form.
Not suitable for 3PL models, for such a kind of model use `latent_information_3PL()`.
It follows the parametrization ``a \\theta - b ``.
See the docs of [`latent_information(examinee::AbstractExaminee, item::AbstractItem)`](@ref) for details.

# Arguments

- **`latents_matrix::Matrix{Float64}`** : Required. A ``n_latents \\times N`` matrix with latents values. 
- **`parameters_matrix::Matrix{Float64}`** : Required. A ``(n_latents + 1) \\times I`` matrix with item parameters. intercept (b) must be in first row, latents coefficients ``(a_j)`` in next rows ``(2, \\ldots, n_latents + 1)``. 
    
# Output

A ``I \\time N`` `Float64` matrix. 
"""
function latent_information(
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
```julia
latent_information_3PL(
    latents_matrix::Matrix{Float64},
    parameters_matrix::Matrix{Float64},
)
```

# Description

Only for models which has guessing parameter (c) in last row of parameters_matrix.
It computes the information function (IIF) for item parameters at latents values provided in matrix form.
It follows the parametrization ``a \\theta - b ``.

# Arguments

- **`latents_matrix::Matrix{Float64}`** : Required. A `n_latents x N` matrix with latents values. 
- **`parameters_matrix::Matrix{Float64}`** : Required. A `(n_latents + 1) x I` matrix with item parameters. intercept (b) must be in first row, latents coefficients (a_j) in next rows (2, ..., n_latents + 1). 

# Output

A `I x N` `Float64` matrix. 
"""
function latent_information_3PL(
    latents_matrix::Matrix{Float64},
    parameters_matrix::Matrix{Float64},
)
    p = probability_3PL(parameters_matrix, latents_matrix)
    return _matrix_cols_vec(p, parameters_matrix[:, 2], (x, y) -> y^2 * (1 - x) * x) .*
           _matrix_cols_vec(p, parameters_matrix[:, end], (x, y) -> ((x - y) / (1 - y))^2)
end