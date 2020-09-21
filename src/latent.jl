abstract type AbstractLatent end

# Structs

mutable struct Latent1D <: AbstractLatent
    val::Float64
    bounds::Vector{Float64}
    prior::Distributions.ContinuousUnivariateDistribution
    posterior::Distributions.ContinuousUnivariateDistribution
    chain::Vector{Float64}
    expected_information::Float64
    # Random Initializers
    function Latent1D(;univariate_distribution = Distributions.Normal(0.0, 1.0), bounds = [-6.0, 6.0])
        val = truncate_rand(univariate_distribution, bounds)[1]
        new(val,
            bounds,
            univariate_distribution,
            univariate_distribution,
            zeros(Float64, 0),
            1.0)
    end
end
