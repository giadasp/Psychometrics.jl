# Types and Structs

abstract type AbstractParameters end

"""
    Parameters1PL

Contains info about the difficulty of an item under the 1-parameter logistic model.
"""
mutable struct Parameters1PL <: AbstractParameters
    b::Float64
    bounds_b::Vector{Float64}
    prior::Distributions.UnivariateDistribution
    posterior::Distributions.UnivariateDistribution
    chain::Vector{Float64}
    expected_information::Float64

    Parameters1PL(b, bounds_b, prior, posterior, chain, expected_information) = new(b, bounds_b, prior, posterior, chain, expected_information)

    # Default Initializers
    Parameters1PL(; distribution_b = Distributions.Normal(0, 1),  bounds_b = [-6.0, 6.0], chain_length = 1000) = new(truncate_rand(distribution_b, bounds_b), bounds_b, distribution_b, distribution_b, zeros(Float64, 0), 1.0)
end

## 2PL 

mutable struct Parameters2PL <: AbstractParameters
    a::Float64
    bounds_a::Vector{Float64}
    b::Float64
    bounds_b::Vector{Float64} 
    prior::Distributions.MultivariateDistribution 
    posterior::Distributions.MultivariateDistribution
    chain::Vector{Vector{Float64}}
    expected_information::Matrix{Float64}

    Parameters2PL(a, bounds_a, b, bounds_b, prior, posterior, chain, expected_information) = new(a, bounds_a, b, bounds_b, prior, posterior, chain, expected_information)

    # Random Initializers
    function Parameters2PL(posterior, bounds_a, bounds_b) 
        pars = truncate_rand(posterior, [bounds_a, bounds_b])
        new(pars[1], bounds_a, pars[2], bounds_b, posterior, posterior, Vector{Vector{Float64}}(undef, 0), [1.0 0.0; 0.0 1.0])
    end

    function Parameters2PL() 
        posterior =  Distributions.Product([Distributions.LogNormal(0, 1), Distributions.Normal(0, 1)])
        bounds_a = [1e-5, 5.0]
        bounds_b = [-6.0, 6.0]
        pars = truncate_rand(posterior, [bounds_a, bounds_b])
        new(pars[1], bounds_a, pars[2], bounds_b, posterior, posterior, Vector{Vector{Float64}}(undef, 0), [1.0 0.0; 0.0 1.0])
    end

end

## 3PL

mutable struct Parameters3PL <: AbstractParameters
    a::Float64
    bounds_a::Vector{Float64}
    b::Float64
    bounds_b::Vector{Float64}
    c::Float64
    bounds_c::Vector{Float64}
    prior::Distributions.MultivariateDistribution
    posterior::Distributions.MultivariateDistribution
    chain::Vector{Vector{Float64}}
    expected_information::Matrix{Float64}

    Parameters3PL(a, bounds_a, b, bounds_b, c, bounds_c, prior, posterior, chain, expected_information) = new(a, bounds_a, b, bounds_b, c, bounds_c, prior, posterior, chain, expected_information)

    # Random Initializers
    
    function Parameters3PL(posterior, bounds_a, bounds_b, bounds_c) 
        pars = truncate_rand(posterior, [bounds_a, bounds_b, bounds_c])
        new(pars[1], bounds_a, pars[2], bounds_b, pars[3], bounds_c, posterior, posterior, Vector{Vector{Float64}}(undef, 0), [1.0 0.0; 0.0 1.0])
    end

    function Parameters3PL() 
        posterior =  Distributions.Product([Distributions.LogNormal(0, 1), Distributions.Normal(0, 1), Distributions.Uniform(0, 1)])
        bounds_a = [1e-5, 5.0]
        bounds_b = [-6.0, 6.0]
        bounds_c = [1e-5, 1.0 - 1e-5]
        pars = truncate_rand(posterior, [bounds_a, bounds_b, bounds_c])
        new(pars[1], bounds_a, pars[2], bounds_b, posterior, posterior, Vector{Vector{Float64}}(undef, 0), [1.0 0.0; 0.0 1.0])
    end
end

mutable struct ParametersNPL <: AbstractParameters
    names::Vector{String}
    vals::Vector{Vector{Float64}}
    bounds::Vector{Vector{Float64}}
    prior::Distributions.MultivariateDistribution
    posterior::Distributions.MultivariateDistribution
    chain::Vector{Vector{Float64}}
    expected_information::Matrix{Float64}

    ParametersNPL(names, vals, bounds, prior, posterior, chain, expected_information) = new(names, vals, bounds, prior, posterior, chain, expected_information)

    # Random Initializers
    
    function ParametersNPL(posterior, bounds)
        N = size(bounds,1)
        pars = truncate_rand(posterior, bounds)
        new(string.("par_",collect(1:N)), pars, bounds, posterior, posterior,[zeros(Float64, 0) for n=1:N], LinearAlgebra.I(N))
    end

end

# Assign prior

"""
    add_prior!(parameters::AbstractParameters, prior::Distributions.Distribution) 

# Description
It assigns the prior `prior` to a `AbstractParameters` instance.

# Arguments
- **`parameters::AbstractParameters`** : Required. Any type of parameters object. 
- **`prior::Distributions.Distribution`** : Required. A probability distribution. It must be the same dimension as `parameters`. 

# Examples
    parameters2PL = Parameters2PL()
    bivariate_normal = Distributions.MultivariateNormal([0,0], LinearAlgebra.I(2))
    add_prior!(parameters2PL, bivariate_normal)
"""
function add_prior!(parameters::AbstractParameters, prior::Distributions.Distribution) 
    parameters.prior .= prior
end

"""
    add_prior!(parameters::AbstractParameters, priors::Vector{Distributions.Distribution}) 

# Description
It transforms the vector of univariate priors `priors` to their products and assign it to `AbstractParameters` instance.

# Arguments
- **`parameters::AbstractParameters`** : Required. Any type of parameters object. 
- **`priors::Vector{Distributions.Distribution}`** : Required. A vector of probability distributions. The size of the vector must be the same as `parameters`. 

# Examples
    parameters2PL = Parameters2PL()
    a_dist = Distributions.Normal(0,1)
    b_dist = Distributions.Normal(0,1)
    add_prior!(parameters2PL, [a_dist, b_dist»])
"""
function add_prior!(parameters::AbstractParameters, priors::Vector{Distributions.Distribution}) 
    parameters.prior .= Distributions.Product(priors)
end

function add_posterior!(parameters::AbstractParameters, posterior::Distributions.Distribution) 
    parameters.posterior .= posterior
end

function add_posterior!(parameters::AbstractParameters, posteriors::Vector{Distributions.Distribution}) 
    parameters.posterior .= Distributions.Product(posteriors)
end
