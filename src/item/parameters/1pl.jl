
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

Parameters1PL(b, bounds_b, prior, posterior, chain, expected_information) =
    new(b, bounds_b, prior, posterior, chain, expected_information)

# Random Initializers
function Parameters1PL(dist::Distributions.UnivariateDistribution, bounds_b)
    pars = truncate_rand(dist, bounds_b)
    new(pars[1][1], bounds_b, dist, dist, [pars[1]], 1.0)
end

function Parameters1PL()
    Parameters1PL(Distributions.Normal(0, 1), [-6.0, 6.0])
end
end