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

    Parameters3PL(
        a,
        bounds_a,
        b,
        bounds_b,
        c,
        bounds_c,
        prior,
        posterior,
        chain,
        expected_information,
    ) = new(
        a,
        bounds_a,
        b,
        bounds_b,
        c,
        bounds_c,
        prior,
        posterior,
        chain,
        expected_information,
    )

    # Random Initializers

    function Parameters3PL(
        trivariate_dist::Distributions.MultivariateDistribution,
        bounds_a,
        bounds_b,
        bounds_c,
    )
        pars = truncate_rand(trivariate_dist, [bounds_a, bounds_b, bounds_c])
        new(
            pars[1][1],
            bounds_a,
            pars[1][2],
            bounds_b,
            pars[1][3],
            bounds_c,
            trivariate_dist,
            trivariate_dist,
            [pars[1]],
            [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
        )
    end

    function Parameters3PL()
        Parameters3PL(
            Distributions.Product([
                Distributions.LogNormal(0, 0.25),
                Distributions.Normal(0, 1),
                Distributions.Uniform(0, 1),
            ]),
            [1e-5, 5.0],
            [-6.0, 6.0],
            [1e-5, 1.0 - 1e-5],
        )
    end
end