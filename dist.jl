using Distributions

import Base.+
import Base.-
import Base.*
import Base./
+(dist::Normal{Float64}, x::Float64) = Normal(location(dist) + x, scale(dist))
-(dist::Normal{Float64}, x::Float64) = Normal(location(dist) - x, scale(dist))
*(dist::Normal{Float64}, x::Float64) = Normal(location(dist) * x, scale(dist) * x)
/(dist::Normal{Float64}, x::Float64) = Normal(location(dist) / x, scale(dist) / x)

+(dist::Uniform{Float64}, x::Float64) = Uniform(location(dist) + x, scale(dist))
-(dist::Uniform{Float64}, x::Float64) = Uniform(location(dist) - x, scale(dist))
*(dist::Uniform{Float64}, x::Float64) = Uniform(location(dist), scale(dist) * x)
/(dist::Uniform{Float64}, x::Float64) = Uniform(location(dist), scale(dist) / x)

*(dist::LogNormal{Float64}, x::Float64) = LogNormal(location(dist)+log(x), scale(dist) )
/(dist::LogNormal{Float64}, x::Float64) = LogNormal(location(dist)-log(x), scale(dist) )

# Truncation

function truncate_rand(distribution::Distributions.UnivariateDistribution, bounds::Vector{Float64}; n = 1)
    return map(x -> min(max(rand(distribution), bounds[1]), bounds[2]), [1:n])
end

function truncate_rand(distribution::Distributions.MultivariateDistribution, bounds::Vector{Vector{Float64}}; n = 1)
     return map( y ->  map(rand(distribution), bounds) do x, bound
        min(max(x, bound[1]), bound[2])
            end, [1:n])
end

# Assign prior

function prior!(parameters::AbstractParameters, prior::Distributions.Distribution) 
    parameters.prior .= prior
end

function prior!(parameters::AbstractParameters, priors::Vector{Distributions.Distribution}) 
    parameters.prior .= Distributions.Product(priors)
end

function posterior!(parameters::AbstractParameters, posterior::Distributions.Distribution) 
    parameters.posterior .= posterior
end

function posterior!(parameters::AbstractParameters, posteriors::Vector{Distributions.Distribution}) 
    parameters.posterior .= Distributions.Product(posteriors)
end
