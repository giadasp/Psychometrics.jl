# 2PL 
"""
    Parameters2PL <: AbstractParametersBinary

Contains info about the difficulty of an item under the 2-parameter logistic model.
"""
mutable struct Parameters2PL <: AbstractParametersBinary
    a::Float64
    bounds_a::Vector{Float64}
    b::Float64
    bounds_b::Vector{Float64}
    prior::Distributions.MultivariateDistribution
    posterior::Distributions.MultivariateDistribution
    chain::Vector{Vector{Float64}}
    expected_information::Matrix{Float64}
    calibrated::Bool

    Parameters2PL(
        a,
        bounds_a,
        b,
        bounds_b,
        prior,
        posterior,
        chain,
        expected_information,
        calibrated,
    ) = new(
        a,
        bounds_a,
        b,
        bounds_b,
        prior,
        posterior,
        chain,
        expected_information,
        calibrated,
    )

    # Random Initializers
    function Parameters2PL(
        bivariate_dist::Distributions.MultivariateDistribution,
        bounds_a,
        bounds_b,
    )
        pars = truncate_rand(bivariate_dist, [bounds_a, bounds_b])
        Parameters2PL(
            pars[1][1],
            bounds_a,
            pars[1][2],
            bounds_b,
            bivariate_dist,
            bivariate_dist,
            [pars[1]],
            [1.0 0.0; 0.0 1.0],
            true,
        )
    end

    function Parameters2PL()
        Parameters2PL(
            Distributions.Product([
                Distributions.LogNormal(0, 0.25),
                Distributions.Normal(0, 1),
            ]),
            [1e-5, 5.0],
            [-6.0, 6.0],
        )
    end

end
