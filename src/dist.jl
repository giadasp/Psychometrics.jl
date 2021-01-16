import Base.+
import Base.-
import Base.*
import Base./

+(dist::Distributions.Normal{Float64}, x::Float64) =
    Distributions.Normal(Distributions.location(dist) + x, Distributions.scale(dist))
-(dist::Distributions.Normal{Float64}, x::Float64) =
    Distributions.Normal(Distributions.location(dist) - x, Distributions.scale(dist))
*(dist::Distributions.Normal{Float64}, x::Float64) =
    Distributions.Normal(Distributions.location(dist) * x, Distributions.scale(dist) * x)
/(dist::Distributions.Normal{Float64}, x::Float64) =
    Distributions.Normal(Distributions.location(dist) / x, Distributions.scale(dist) / x)

+(dist::Distributions.Uniform{Float64}, x::Float64) =
    Distributions.Uniform(Distributions.location(dist) + x, Distributions.scale(dist))
-(dist::Distributions.Uniform{Float64}, x::Float64) =
    Distributions.Uniform(Distributions.location(dist) - x, Distributions.scale(dist))
*(dist::Distributions.Uniform{Float64}, x::Float64) =
    Distributions.Uniform(Distributions.location(dist), Distributions.scale(dist) * x)
/(dist::Distributions.Uniform{Float64}, x::Float64) =
    Distributions.Uniform(Distributions.location(dist), Distributions.scale(dist) / x)

*(dist::Distributions.LogNormal{Float64}, x::Float64) = Distributions.LogNormal(
    Distributions.location(dist) + log(x),
    Distributions.scale(dist),
)
/(dist::Distributions.LogNormal{Float64}, x::Float64) = Distributions.LogNormal(
    Distributions.location(dist) - log(x),
    Distributions.scale(dist),
)

# Truncation

function truncate_rand(
    distribution::Distributions.UnivariateDistribution,
    bounds::Vector{Float64};
    n::Int64 = 1,
)
    return map(_ -> min(max(rand(distribution), bounds[1]), bounds[2]), 1:n)
end

function truncate_rand(
    distribution::Distributions.MultivariateDistribution,
    bounds::Vector{Vector{Float64}};
    n::Int64 = 1,
)
    return map(y -> map(rand(distribution), bounds) do x, bound
        min(max(x, bound[1]), bound[2])
    end, 1:n)
end
