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
    Latent1D(dist, bounds)

Same as `Latent1D(dist, bounds)` but with value equal to `val`
    Latent1D(val)

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
    function Latent1D(val::Float64)
        latent = Latent1D()
        latent.val = val
        return latent
    end
    function Latent1D(dist::Distributions.UnivariateDistribution, bounds::Vector{Float64})
        val = truncate_rand(dist, bounds)[1]
        new(val, bounds, dist, dist, [val], 1.0)
    end
end

"""
    _empty_chain!(latent::Latent1D)
"""
function _empty_chain!(latent::Latent1D)
    latent.chain = Float64[]
end

"""
    _set_val!(latent::Latent1D, val::Float64)
"""
function _set_val!(latent::Latent1D, val::Float64)
    latent.val = copy(val)
end

"""
    _set_val_from_chain!(latent::Latent1D)
"""
function _set_val_from_chain!(latent::Latent1D)
    latent.val = latent.chain[end]
end

"""
    _update_estimate!(latent::Latent1D; sampling = true)
"""
function _update_estimate!(latent::Latent1D; sampling = true)
    chain_size = size(latent.chain, 1)
    if sampling
        latent.val =
            sum(latent.chain[(chain_size-min(999, chain_size - 1)):end]) /
            min(1000, chain_size)
    else
        latent.val = sum(latent.chain) / chain_size
    end
end

"""
    _chain_append!(latent::Latent1D; sampling = false)
"""
function _chain_append!(latent::Latent1D; sampling = false)
    val = Distributions.rand(latent.posterior)
    if (sampling && size(latent.chain, 1) >= 1000)
        latent.chain[Random.rand(1:1_000)] = val
    else
        push!(latent.chain, val)
    end
    return val::Float64
end
