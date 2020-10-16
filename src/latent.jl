abstract type AbstractLatent end

# Structs
"""
    Latent1D <: AbstractLatent

# Description
1-dimensional latent variable struct.

# Fields
    - **`val::Float64`**
    - **`bounds::Vector{Float64}`**
    - **`prior::Distributions.ContinuousUnivariateDistribution`**
    - **`posterior::Distributions.ContinuousUnivariateDistribution`**
    - **`chain::Vector{Float64}`**
    - **`expected_information::Float64`**

# Factories
    Latent1D(val, bounds, prior, posterior, chain, expected_information) = new(val, bounds, prior, posterior, chain, expected_information)
Creates a new 1-dimensional latent variable with custom fields.

# Random Initializers
    Latent1D()

Randomly generates a value for the 1-dimensional latent variable and assigns a default standardized Gaussian prior and posterior
    Latent1D(bounds, posterior)

Randomly generates a value for the 1-dimensional latent variable and assigns a custom univariate distribution to prior and posterior with specific bounds.
"""
mutable struct Latent1D <: AbstractLatent
    val::Float64
    bounds::Vector{Float64}
    prior::Distributions.ContinuousUnivariateDistribution
    posterior::Distributions.ContinuousUnivariateDistribution
    chain::Vector{Float64}
    expected_information::Float64
    Latent1D(val, bounds, prior, posterior, chain, expected_information) =
        new(val, bounds, prior, posterior, chain, expected_information)

    # Random Initializers
    function Latent1D()
        bounds = [-6.0, 6.0]
        dist = Distributions.Normal(0.0, 1.0)
        val = truncate_rand(dist, bounds)[1]
        new(val, bounds, dist, dist, [val], 1.0)
    end
    function Latent1D(dist::Distributions.UnivariateDistribution, bounds::Vector{Float64})
        val = truncate_rand(dist, bounds)[1]
        new(val, bounds, dist, dist, [val], 1.0)
    end
end

"""
    LatentND <: AbstractLatent

# Description
N-dimensional latent variable struct.

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
    LatentND(bounds, posterior)

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
