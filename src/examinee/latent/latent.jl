abstract type AbstractLatent end

include("1D.jl")
include("ND.jl")

"""
    Latent <: AbstractLatent

# Description
Generic latent variable struct.

# Fields
    - **`val::Vector{Float64}`**
    - **`bounds::Vector{Vector{Float64}}`**
    - **`prior::Union{Distributions.ContinuousUnivariateDistribution, Distributions.MultivariateDistribution}`**
    - **`posterior::Union{Distributions.ContinuousUnivariateDistribution, Distributions.MultivariateDistribution}`**
    - **`chain::Vector{Vector{Float64}} `**
    - **`expected_information::Matrix{Float64}`**

# Factories
    Latent(names, val, bounds, prior, posterior, chain, expected_information) = new(val, bounds, prior, posterior, chain, expected_information)
Creates a new generic latent variable with custom fields.

# Random Initializers
    Latent(bounds, posterior)

Randomly generates a value for the generic latent variable and assigns a custom univariate or multivariate distribution to prior and posterior with specific bounds.
"""
mutable struct Latent <: AbstractLatent
    names::Vector{String}
    val::Vector{Float64}
    bounds::Vector{Vector{Float64}}
    prior::Union{
        Distributions.ContinuousUnivariateDistribution,
        Distributions.MultivariateDistribution,
    }
    posterior::Union{
        Distributions.ContinuousUnivariateDistribution,
        Distributions.MultivariateDistribution,
    }
    chain::Vector{Vector{Float64}}
    expected_information::Matrix{Float64}

    function Latent(names, val, bounds, prior, posterior, chain, expected_information)
        new(names, val, bounds, prior, posterior, chain, expected_information)
    end

    function Latent(
        bounds,
        dist::Union{
            Distributions.ContinuousUnivariateDistribution,
            Distributions.MultivariateDistribution,
        },
    )
        N = 1
        val = truncate_rand(dist, bounds)
        new(string.("L_", collect(1:N)), val, bounds, dist, dist, [val], LinearAlgebra.I(N))
    end
end


# Structs
