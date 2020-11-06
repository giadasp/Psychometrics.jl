#


mutable struct ParametersNPL <: AbstractParameters
    a::Vector{Float64}
    bounds_a::Vector{Vector{Float64}}
    b::Float64
    bounds_b::Vector{Float64}
    prior::Distributions.MultivariateDistribution
    posterior::Distributions.MultivariateDistribution
    chain::Vector{Vector{Float64}}
    expected_information::Matrix{Float64}

    ParametersNPL(a, bounds_a, b, bounds_b, prior, posterior, chain, expected_information) =
        new(a, bounds_a, b, bounds_b, prior, posterior, chain, expected_information)

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
        )
    end

end