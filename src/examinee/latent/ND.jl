
"""
LatentND <: AbstractLatent

# Description
A mutable describing a multivariate latent variable.

# Fields
- **`names::Vector{String}`**
- **`val::Vector{Float64}`**
- **`bounds::Vector{Vector{Float64}}`**
- **`prior::Distributions.MultivariateDistribution`**
- **`posterior::Distributions.MultivariateDistribution`**
- **`chain::Vector{Vector{Float64}}`**
- **`expected_information::Matrix{Float64}`**

# Factories
LatentND(names, val, bounds, prior, posterior, chain, expected_information) = new(names, val, bounds, prior, posterior, chain, expected_information)
Creates a new N-dimensional latent variable with custom fields.

# Random Initializers
LatentND()

Randomly generates a value for the N-dimensional latent variable and assigns a default standardized Gaussian prior and posterior
LatentND(dist, bounds)

Randomly generates a value for the N-dimensional latent variable and assigns a custom multivariate distribution to prior and posterior with specific bounds.
"""
mutable struct LatentND <: AbstractLatent
    names::Vector{String}
    val::Vector{Float64}
    bounds::Vector{Vector{Float64}}
    prior::Distributions.MultivariateDistribution
    posterior::Distributions.MultivariateDistribution
    chain::Vector{Vector{Float64}}
    expected_information::Matrix{Float64}

    function LatentND(names, val, bounds, prior, posterior, chain, expected_information)
        new(names, val, bounds, prior, posterior, chain, expected_information)
    end

    # Random Initializers

    function LatentND(
        dist::Distributions.MultivariateDistribution,
        bounds::Vector{Vector{Float64}},
    )
        N = size(bounds, 1)
        val = truncate_rand(dist, bounds)
        new(string.("L_", collect(1:N)), val, bounds, dist, dist, [val], LinearAlgebra.I(N))
    end
end
