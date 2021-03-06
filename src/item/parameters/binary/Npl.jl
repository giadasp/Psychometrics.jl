## NPL
"""
    ParametersNPL <: AbstractParametersBinary

# Description

Contains info about the difficulty of an item under the N-parameter logistic model.
"""
mutable struct ParametersNPL <: AbstractParametersBinary
    a::Vector{Float64}
    bounds_a::Vector{Vector{Float64}}
    b::Float64
    bounds_b::Vector{Float64}
    prior::Distributions.MultivariateDistribution
    posterior::Distributions.MultivariateDistribution
    chain::Vector{Vector{Float64}}
    expected_information::Matrix{Float64}
    calibrated::Bool

    ParametersNPL(
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

    function ParametersNPL(
        N_variate_dist::Distributions.MultivariateDistribution,
        bounds::Vector{Vector{Float64}},
    )
        N = size(bounds, 1)
        pars = truncate_rand(N_variate_dist, bounds)
        new(
            string.("par_", collect(1:N)),
            pars[2:end],
            bounds[2:end],
            pars[1],
            bounds[1],
            N_variate_dist,
            N_variate_dist,
            [pars[1]],
            LinearAlgebra.I(N),
            true,
        )
    end

end
