
"""
`Parameters1PL <: AbstractParametersBinary`

Contains information of a set of item parameters (values, bounds, priors, posteiors, chains, expected Fisher information, calibrated) under the 1-parameter logistic model.
"""
mutable struct Parameters1PL <: AbstractParametersBinary
    b::Float64
    bounds_b::Vector{Float64}
    prior::Distributions.UnivariateDistribution
    posterior::Distributions.UnivariateDistribution
    chain::Vector{Float64}
    expected_information::Float64
    calibrated::Bool

    Parameters1PL(b, bounds_b, prior, posterior, chain, expected_information, calibrated) =
        new(b, bounds_b, prior, posterior, chain, expected_information, calibrated)

    # Random Initializers
    function Parameters1PL(dist::Distributions.UnivariateDistribution, bounds_b)
        pars = truncate_rand(dist, bounds_b)
        new(pars[1][1], bounds_b, dist, dist, [pars[1]], 1.0, true)
    end

    function Parameters1PL()
        Parameters1PL(Distributions.Normal(0, 1), [-6.0, 6.0], true)
    end
end

"""
`_empty_chain!(parameters::Parameters1PL)`
"""
function _empty_chain!(parameters::Parameters1PL)
    parameters.chain = Vector{Float64}(undef, 0)
end
