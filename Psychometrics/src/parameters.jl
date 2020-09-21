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
    function Parameters2PL(;univariate_distribution_a = Distributions.LogNormal(0, 1),
    bounds_a = [1e-5, 5.0] , univariate_distribution_b = Distributions.Normal(0, 1),
    bounds_b = [-6.0, 6.0], bivariate_distribution = Distributions.Product(univariate_distribution_a,univariate_distribution_b)
    ) 
        pars = truncate_rand(bivariate_distribution, bounds)
        new(pars[1], bounds_a, pars[2], bounds_b, bivariate_distribution, bivariate_distribution, Vector{Vector{Float64}}(undef, 0), [1.0 0.0; 0.0 1.0])
    end

    function Parameters2PL(; bivariate_distribution = Distributions.Product([LogNormal(0,1),Normal(0, 1)]), bounds = [[1e-5, 5.0], [-6.0, 6.0]])
        pars = truncate_rand(bivariate_distribution, bounds)
        new(pars[1][1], bounds[1], pars[1][2], bounds[2], bivariate_distribution, bivariate_distribution, Vector{Vector{Float64}}(undef, 0), Distributions.cov(bivariate_distribution))
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
    
    function Parameters3PL(;univariate_distribution_a = Distributions.LogNormal(0, 1),  bounds_a = [1e-5, 5.0],  
        univariate_distribution_b = Distributions.Normal(0, 1), bounds_b = [-6.0, 6.0],
        univariate_distribution_c = Distributions.Uniform(0, 1),  bounds_c = [1e-5, 1.0 - 1e-5], 
    )
        trivariate_distribution = Distributions.Product([univariate_distribution_a, univariate_distribution_b, univariate_distribution_c])
        pars = truncate_rand(trivariate_distribution, bounds)
        new(pars[1], bounds_a, pars[2], bounds_b, pars[3], bounds_c, trivariate_distribution, trivariate_distribution, zeros(Int64, 0))
    end

    function Parameters3PL(; trivariate_distribution = Distributions.Product([LogNormal(0,1), Normal(0, 1), Uniform(0, 1)]), bounds = [[1e-5, 5.0], [-6.0, 6.0], [1e-5, 1.0 - 1e-5]]
        )
        pars = truncate_rand(trivariate_distribution, bounds)
        new(pars[1][1], bounds[1], pars[1][2], bounds[2], pars[1][3], bounds[3], trivariate_distribution, trivariate_distribution, Vector{Vector{Float64}}(undef, 0), Distributions.cov(trivariate_distribution))
    end
end


