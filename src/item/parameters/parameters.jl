# Types and Structs
"""
    AbstractParameters

# Description

An abstract type which, at the moment, has only the abstract `AbstractParametersBinary` as subtype.
Thus, it works only with binary (dichotomous) responses. The available mutables for the latter are `Parameters1PL`, `Parameters2PL`, `Parameters3PL`.
They contain the details about the item parameters under the 1-parameter logistic (1PL), 2-parameters logistic (2PL) and 3-parameter logistic (3PL) IRT models, respectively.

# Example

The object `Parameters2PL` has the fields difficulty `b::Float64` and discrimination `a::Float64`. It is possible to define the Bayesian priors and posterior by assigning multivariate distributions from the package `Distributions` to the fields `prior::Distributions.MultivariateDistribution` and `posterior::Distributions.MultivariateDistribution`.
"""
abstract type AbstractParameters end

include("binary/binary.jl")

# Assign prior

"""
    _add_prior!(parameters::AbstractParameters, prior::Distributions.Distribution) 

# Description
It assigns the prior `prior` to a `AbstractParameters` instance.

# Arguments
- **`parameters::AbstractParameters`** : Required. Any type of parameters object. 
- **`prior::Distributions.Distribution`** : Required. A <n>-variate probability distribution where <n> > 1 and is the numebr of item parameters in `parameters`. 

# Examples
    parameters2PL = Parameters2PL()
    bivariate_normal = Distributions.MultivariateNormal([0,0], LinearAlgebra.I(2))
    _add_prior!(parameters2PL, bivariate_normal)
"""
function add_prior!(
    parameters::AbstractParameters,
    prior::Distributions.MultivariateDistribution,
)
    parameters.prior .= prior
end

"""
    _add_prior!(parameters::AbstractParameters, priors::Vector{Distributions.UnivariateDistribution}) 

# Description
It transforms the vector `priors` of univariate distributions to their products and assign it to `AbstractParameters` instance.

# Arguments
- **`parameters::AbstractParameters`** : Required. Any type of parameters object. 
- **`priors::Vector{Distributions.UnivariateDistribution}`** : Required. A vector of probability distributions. The size of the vector must be the same as the number of item parameters. 

# Examples
    parameters2PL = Parameters2PL()
    a_dist = Distributions.Normal(0,1)
    b_dist = Distributions.Normal(0,1)
    _add_prior!(parameters2PL, [a_dist, b_dist])
"""
function add_prior!(
    parameters::AbstractParameters,
    priors::Vector{Distributions.UnivariateDistribution},
)
    parameters.prior .= Distributions.Product(priors)
end


# Assign posterior distribution

"""
    _add_posterior!(parameters::AbstractParameters, posterior::Distributions.Distribution) 

# Description
It assigns the <n>-variate `posterior` distribution to a `AbstractParameters` instance with <n> parameters.

# Arguments
- **`parameters::AbstractParameters`** : Required. Any type of parameters object. 
- **`posterior::Distributions.Distribution`** : Required. A <n>-variate probability distribution where <n> > 1 and is the numebr of item parameters in `parameters`. 

# Examples
    parameters2PL = Parameters2PL()
    bivariate_normal = Distributions.MultivariateNormal([0,0], LinearAlgebra.I(2))
    _add_posterior!(parameters2PL, bivariate_normal)
"""
function _add_posterior!(
    parameters::AbstractParameters,
    posterior::Distributions.Distribution,
)
    parameters.posterior .= posterior
end

"""
    _add_posterior!(parameters::AbstractParameters, priors::Vector{Distributions.UnivariateDistribution}) 

# Description
It transforms the vector `posteriors` of univariate distributions to their products and assign it to `AbstractParameters` instance.

# Arguments
- **`parameters::AbstractParameters`** : Required. Any type of parameters object. 
- **`posteriors::Vector{Distributions.UnivariateDistribution}`** : Required. A vector of probability distributions. The size of the vector must be the same as the number of `parameters`. 

# Examples
    parameters2PL = Parameters2PL()
    a_dist = Distributions.Normal(0,1)
    b_dist = Distributions.Normal(0,1)
    _add_posterior!(parameters2PL, [a_dist, b_dist])
"""
function _add_posterior!(
    parameters::AbstractParameters,
    posteriors::Vector{Distributions.UnivariateDistribution},
)
    parameters.posterior .= Distributions.Product(posteriors)
end
