# 2PL 
"""
    Parameters2PL <: AbstractParametersBinary

Contains information of a set of item parameters (values, bounds, priors, posteiors, chains, expected Fisher information, calibrated) under the 2-parameter logistic model.
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
        return Parameters2PL(
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
        return Parameters2PL(
            Distributions.Product([
                Distributions.LogNormal(0, 0.25),
                Distributions.Normal(0, 1),
            ]),
            [1e-5, 5.0],
            [-6.0, 6.0],
        )
    end

    function Parameters2PL(bounds_a::Vector{Float64}, bounds_b::Vector{Float64})
        par = Parameters2PL()
        par.bounds_a = bounds_a
        par.bounds_b = bounds_b
        return par::Parameters2PL
    end

end

"""
    _empty_chain!(parameters::Parameters2PL)
"""
function _empty_chain!(parameters::Parameters2PL)
    parameters.chain = Vector{Vector{Float64}}(undef, 0)
end

"""
    _set_val!(parameters::Parameters2PL, vals::Vector{Float64})
"""
function _set_val!(parameters::Parameters2PL, vals::Vector{Float64})
    parameters.a = vals[1]
    parameters.b = vals[2]
end

"""
    _set_val_from_chain!(parameters::Parameters2PL)
"""
function _set_val_from_chain!(parameters::Parameters2PL)
    parameters.a = parameters.chain[end][1]
    parameters.b = parameters.chain[end][2]
end

"""
    _update_estimate!(parameters::Parameters2PL; sampling = true)
"""
function _update_estimate!(parameters::Parameters2PL; sampling = true)
    chain_size = size(parameters.chain, 1)
    if sampling
        chain_matrix = hcat(parameters.chain[(chain_size-min(999, chain_size - 1)):end]...)
        vals = [sum(i) / min(1000, chain_size) for i in eachrow(chain_matrix)]
    else
        chain_matrix = hcat(parameters.chain...)
        vals = [sum(i) / chain_size for i in eachrow(chain_matrix)]
    end
    parameters.a = clamp(vals[1], parameters.bounds_a[1], parameters.bounds_a[2])
    parameters.b = clamp(vals[2], parameters.bounds_b[1], parameters.bounds_b[2])
end

"""
    _get_parameters_vals(parameters::Parameters2PL)
"""
function _get_parameters_vals(parameters::Parameters2PL)
    return [parameters.a, parameters.b]
end