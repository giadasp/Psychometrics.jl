abstract type AbstractLatent end

# Structs

mutable struct Latent1D <: AbstractLatent
    val::Float64
    bounds::Vector{Float64} 
    prior::Distributions.ContinuousUnivariateDistribution 
    posterior::Distributions.ContinuousUnivariateDistribution
    chain::Vector{Float64}
    expected_information::Float64
    Latent1D(val,bounds,prior,posterior,chain,expected_information) = new(val,bounds,prior,posterior,chain,expected_information)

    # Random Initializers
    function Latent1D()
        bounds = [-6.0,6.0]
        posterior = Distributions.Normal(0.0, 1.0)
        val = truncate_rand(posterior, bounds)[1]
        new(val,
            bounds,
            posterior,
            posterior,
            zeros(Float64, 0),
            1.0)
    end
    function Latent1D(bounds, posterior)
        val = truncate_rand(posterior, bounds)[1]
        new(val,
            bounds,
            posterior,
            posterior,
            zeros(Float64, 0),
            1.0)
    end
end

mutable struct LatentND <: AbstractLatent
    names::Vector{String}
    val::Vector{Float64}
    bounds::Vector{Vector{Float64}} 
    prior::Distributions.ContinuousUnivariateDistribution 
    posterior::Distributions.ContinuousUnivariateDistribution
    chain::Vector{Vector{Float64}} 
    expected_information::Matrix{Float64}

    function LatentND(names,val,bounds,prior,posterior,chain,expected_information)
    new(names,val,bounds,prior,posterior,chain,expected_information)
    end
    
    function LatentND(bounds, posterior)
        N = size(bounds,1)
        val = truncate_rand(posterior, bounds)
        new(string.("L_",collect(1:N)),
            val,
            bounds,
            posterior,
            posterior,
            [zeros(Float64, 0) for n=1:N],
            LinearAlgebra.I(N))
    end
end
