import Random: GLOBAL_RNG
import Random
import SpecialFunctions: loggamma, gamma
#import Random:rand

# Truncated Inverse Gaussian distribution

TruncatedInverseGaussian(μ::Float64, λ::Float64, a::Float64, b::Float64) =
    Distributions.Truncated(Distributions.InverseGaussian(μ, λ), a, b)
TruncatedInverseGaussian(μ::Real, λ::Real, a::Real, b::Real) =
    TruncatedInverseGaussian(Float64(μ), Float64(λ), Float64(a), Float64(b))

# ### statistics

minimum(
    d::Distributions.Truncated{Distributions.InverseGaussian{T},Distributions.Continuous},
) where {T<:Real} = d.lower
maximum(
    d::Distributions.Truncated{Distributions.InverseGaussian{T},Distributions.Continuous},
) where {T<:Real} = d.upper

## sampling

### sampler for upper truncated inverse gaussian

function Distributions.rand(
    rng::Distributions.AbstractRNG,
    d::Distributions.Truncated{Distributions.InverseGaussian{T},Distributions.Continuous},
) where {T<:Real}
    μ = d.untruncated.μ
    λ = d.untruncated.λ
    if d.lower == 0.0
        X = d.upper + 1
        if (μ > d.upper)
            alpha = 0.0
            while (Random.rand(rng) > alpha)
                ## X = R + 1
                ## while (X > R) {
                ##     X = 1.0 / rgamma(1, 0.5, rate=0.5);
                ## }
                E = Random.randexp(rng, 2)
                while E[1]^2 > (2 * E[2] / d.upper)
                    E = Random.randexp(rng, 2)
                end
                X = d.upper / (1 + d.upper * E[1])^2
                alpha = _exp_c(-0.5 / (μ)^2 * X)
            end
        else
            while (X > d.upper)
                Y = Random.randn(rng)^2
                X = μ * (1 + 0.5 * μ / λ * Y - 0.5 / λ * sqrt(4 * μ * λ * Y + (μ * Y)^2))
                if (Random.rand(rng) > μ / (μ + X))
                    X = μ^2 / X
                end
            end
        end
        return X
    else
        error("The sampler is available only for upper TruncatedInverseGaussian.")
    end
end
