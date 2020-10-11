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

    Parameters1PL(b, bounds_b, prior, posterior, chain, expected_information) =
        new(b, bounds_b, prior, posterior, chain, expected_information)

    # Random Initializers
    function Parameters1PL(dist::Distributions.UnivariateDistribution, bounds_b)
        pars = truncate_rand(dist, bounds_b)
        new(pars[1][1], bounds_b, dist, dist, Vector{Vector{Float64}}(undef, 0), 1.0)
    end

    function Parameters1PL()
        Parameters1PL(Distributions.Normal(0, 1), [-6.0, 6.0])
    end
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

    Parameters2PL(a, bounds_a, b, bounds_b, prior, posterior, chain, expected_information) =
        new(a, bounds_a, b, bounds_b, prior, posterior, chain, expected_information)

    # Random Initializers
    function Parameters2PL(
        bivariate_dist::Distributions.MultivariateDistribution,
        bounds_a,
        bounds_b,
    )
        pars = truncate_rand(bivariate_dist, [bounds_a, bounds_b])
        new(
            pars[1][1],
            bounds_a,
            pars[1][2],
            bounds_b,
            bivariate_dist,
            bivariate_dist,
            Vector{Vector{Float64}}(undef, 0),
            [1.0 0.0; 0.0 1.0],
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
            Vector{Vector{Float64}}(undef, 0),
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
            [zeros(Float64, 0) for n = 1:N],
            LinearAlgebra.I(N),
        )
    end

end

# Assign prior

"""
    add_prior!(parameters::AbstractParameters, prior::Distributions.Distribution) 

# Description
It assigns the prior `prior` to a `AbstractParameters` instance.

# Arguments
- **`parameters::AbstractParameters`** : Required. Any type of parameters object. 
- **`prior::Distributions.Distribution`** : Required. A <n>-variate probability distribution where <n> > 1 and is the numebr of item parameters in `parameters`. 

# Examples
    parameters2PL = Parameters2PL()
    bivariate_normal = Distributions.MultivariateNormal([0,0], LinearAlgebra.I(2))
    add_prior!(parameters2PL, bivariate_normal)
"""
function add_prior!(
    parameters::AbstractParameters,
    prior::Distributions.MultivariateDistribution,
)
    parameters.prior .= prior
end

"""
    add_prior!(parameters::AbstractParameters, priors::Vector{Distributions.UnivariateDistribution}) 

# Description
It transforms the vector `priors` of univariate distributions to their products and assign it to `AbstractParameters` instance.

# Arguments
- **`parameters::AbstractParameters`** : Required. Any type of parameters object. 
- **`priors::Vector{Distributions.UnivariateDistribution}`** : Required. A vector of probability distributions. The size of the vector must be the same as the number of item parameters. 

# Examples
    parameters2PL = Parameters2PL()
    a_dist = Distributions.Normal(0,1)
    b_dist = Distributions.Normal(0,1)
    add_prior!(parameters2PL, [a_dist, b_dist])
"""
function add_prior!(
    parameters::AbstractParameters,
    priors::Vector{Distributions.UnivariateDistribution},
)
    parameters.prior .= Distributions.Product(priors)
end


# Assign posterior distribution

"""
    add_posterior!(parameters::AbstractParameters, posterior::Distributions.Distribution) 

# Description
It assigns the <n>-variate `posterior` distribution to a `AbstractParameters` instance with <n> parameters.

# Arguments
- **`parameters::AbstractParameters`** : Required. Any type of parameters object. 
- **`posterior::Distributions.Distribution`** : Required. A <n>-variate probability distribution where <n> > 1 and is the numebr of item parameters in `parameters`. 

# Examples
    parameters2PL = Parameters2PL()
    bivariate_normal = Distributions.MultivariateNormal([0,0], LinearAlgebra.I(2))
    add_posterior!(parameters2PL, bivariate_normal)
"""
function add_posterior!(
    parameters::AbstractParameters,
    posterior::Distributions.Distribution,
)
    parameters.posterior .= posterior
end

"""
    add_posterior!(parameters::AbstractParameters, priors::Vector{Distributions.UnivariateDistribution}) 

# Description
It transforms the vector `posteriors` of univariate distributions to their products and assign it to `AbstractParameters` instance.

# Arguments
- **`parameters::AbstractParameters`** : Required. Any type of parameters object. 
- **`posteriors::Vector{Distributions.UnivariateDistribution}`** : Required. A vector of probability distributions. The size of the vector must be the same as the number of `parameters`. 

# Examples
    parameters2PL = Parameters2PL()
    a_dist = Distributions.Normal(0,1)
    b_dist = Distributions.Normal(0,1)
    add_posterior!(parameters2PL, [a_dist, b_dist])
"""
function add_posterior!(
    parameters::AbstractParameters,
    posteriors::Vector{Distributions.UnivariateDistribution},
)
    parameters.posterior .= Distributions.Product(posteriors)
end
