# Truncated Inverse Gaussian distribution

TruncatedInverseGaussian(μ::Float64, λ::Float64, a::Float64, b::Float64) =
    Distributions.Truncated(Distributions.InverseGaussian(μ, λ), a, b)
TruncatedInverseGaussian(μ::Real, λ::Real, a::Real, b::Real) =
    TruncatedInverseGaussian(Float64(μ), Float64(λ), Float64(a), Float64(b))

# ### statistics


function Distributions.minimum(
    d::Distributions.Truncated{Distributions.InverseGaussian{T},Distributions.Continuous},
) where {T<:Real}
    min(d.lower, Distributions.minimum(d.untruncated))
end
function Distributions.maximum(
    d::Distributions.Truncated{Distributions.InverseGaussian{T},Distributions.Continuous},
) where {T<:Real}
    max(d.lower, Distributions.maximum(d.untruncated))
end

## sampling

### sampler for upper truncated inverse gaussian

function Distributions.rand(
    rng::Distributions.AbstractRNG,
    d::Distributions.Truncated{Distributions.InverseGaussian{T},Distributions.Continuous},
) where {T<:Real}
    mu = d.untruncated.μ
    lambda = d.untruncated.λ
    R = d.upper
    if d.lower == 0.0
        #upper Truncated
        X = R + 1
        if (mu > R)
            alpha = 0.0
            while (Random.rand(rng) > alpha)
                ## X = R + 1
                ## while (X > R) {
                ##     X = 1.0 / rgamma(1, 0.5, rate=0.5)
                ## }
                E = [Random.randexp(rng) for i = 1:2]
                while (E[1]^2 > 2 * E[2] / R)
                    E = [Random.randexp(rng) for i = 1:2]
                end
                X = R / (1 + R * E[1])^2
                alpha = exp(-0.5 * (1 / mu)^2 * X)
            end
        else
            while (X > R)
                lambda = 1.0
                Y = Random.randn(rng)^2
                X =
                    mu + 0.5 * mu^2 / lambda * Y -
                    0.5 * mu / lambda * sqrt(4 * mu * lambda * Y + (mu * Y)^2)
                if (Random.randexp(rng) > mu / (mu + X))
                    X = mu^2 / X
                end
            end
        end
        return X
    else
        error("The sampler is available only for upper TruncatedInverseGaussian.")
    end
end
