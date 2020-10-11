#See Philippe, *Simulation of right and left truncated gamma
# distributions by mixtures* and Dagpunar *Sampling of variates from
# a truncated gamma distribution*
# for pdf, cdf and quantile only right truncation is supported, left bound is 0.0
# for rand both left and right truncation is supported

TruncatedGamma(α::Float64, θ::Float64, lower::Float64, upper::Float64) =
    Distributions.Truncated(Distributions.Gamma(α, θ), lower, upper)
TruncatedGamma(α::Float64, θ::Float64, lower::Real, upper::Real) =
    TruncatedGamma(Float64(α), Float64(θ), lower, Float64(upper))

# ### statistics

minimum(
    d::Distributions.Truncated{Distributions.Gamma{T},Distributions.Continuous},
) where {T<:Real} = d.lower
maximum(
    d::Distributions.Truncated{Distributions.Gamma{T},Distributions.Continuous},
) where {T<:Real} = d.upper

# density of the right truncated gamma distribution 
function Distributions.pdf(d::Distributions.Truncated{Distributions.Gamma{T},Distributions.Continuous}, x::Real) where {T<:Real} 
    if minimum(Distributions.untruncated(d)) == d.lower && maximum(Distributions.untruncated(d)) == d.upper
        Distributions.pdf(Distributions.Gamma(d.α, d.θ), x)
    elseif minimum(Distributions.untruncated(d)) == d.lower
        # right truncated
        return Distributions.pdf(Distributions.Gamma(d.α, d.θ), x) / Distributions.cdf(Distributions.Gamma(d.α, d.θ), d.upper)
    else
        # left truncated
        println("left truncation still not supported.")
        return nothing
    end
end

function Distributions.cdf(d::Distributions.Truncated{Distributions.Gamma{T},Distributions.Continuous}, x::Real) where {T<:Real}
    if minimum(Distributions.untruncated(d)) == d.lower && maximum(Distributions.untruncated(d)) == d.upper
        Distributions.cdf(Distributions.Gamma(d.α, d.θ), x)
    elseif minimum(Distributions.untruncated(d)) == d.lower
        # right truncated
        if (x > d.upper)
            x = d.upper
        end
        return Distributions.cdf(Distributions.untruncated(d), x) / Distributions.cdf(Distributions.untruncated(d), d.upper)
    else
        # left truncated
        println("left truncation still not supported.")
        return nothing
    end
end

function Distributions.quantile(d::Distributions.Truncated{Distributions.Gamma{T},Distributions.Continuous}, x::Real) where {T<:Real} 
    if minimum(Distributions.untruncated(d)) == d.lower && maximum(Distributions.untruncated(d)) == d.upper
        Distributions.quantile(Distributions.Gamma(d.α, d.θ), x)
    elseif minimum(Distributions.untruncated(d)) == d.lower
        # right truncated
        if (d.α == 0.0)
            return 0.0
        end
    
        val =  x * Distributions.cdf(Distributions.untrucated(d), d.upper) 
        return Distributions.quantile(Distributions.untrucated(d), val)
        else
        # left truncated
        println("left truncation still not supported.")
        return nothing
    end
end

#Sampling

# optimal number of components for p = 0.95 fixed
function ncomp_optimal(θ::Float64)
  q = 1.644853626951
  ans = 0.25 * (q * sqrt(q * q + 4. * θ)^2)
  return trunc(Int64, ans)::Int64
end

#random number generation from the gamma with right truncation point t = 1,
# i.e. TG^-(α,θ,0.0,1.0) 
function rng_right_tgamma_t1(rng::Distributions.AbstractRNG, d::Distributions.Truncated{Distributions.Gamma{T},Distributions.Continuous}) where {T<:Real}
  n = ncomp_optimal(d.θ)
  wl  = zeros(Float64, n + 2)
  wlc = similar(wl)

  wl[1] = 1.0; wlc[1] = 1.0
  for i in 2:(n+1)
    wl[i]  = wl[i-1] * d.θ / (d.α + i)
    wlc[i] = wlc[i-1] + wl[i]
  end
  wlc = wlc ./ wlc[end-1]
  y = 1.0
  yy = 1.0
  for  i in 2:(n+1)
    yy *= d.θ / i
    y  += yy
  end
  cont = true
  z = 0.0
  while (cont)
    u = Distributions.rand(rng)
    j = 1
    while (u > wlc[j])
        j += 1;
        x = Distributions.rand(rng, Distributions.Beta(a,j))
        u = Distributions.rand(rng)
        z = 1.0
        zz = 1.0
    for i in 2:(n+1)
      zz *= (1 - x) * d.θ / i
      z  += zz
    end
    z = _exp_c(-d.θ * x) * y / z
    if (u <= z)
      cont = false
    end
end
  end
  return z::Float64
end

function ltgamma(rng::Distributions.AbstractRNG, d::Distributions.Truncated{Distributions.Gamma{T},Distributions.Continuous}) where {T<:Real}
    trunc = d.upper
    shape = d.α
    b = 1/d.θ * trunc

    if (shape == 1) 
        return Distributions.rand(rng, Distributions.Exponential(1.0)) / (1/d.θ) + trunc
    end

    d1 = b-shape
    d3 = shape - 1
    c0 = 0.5 * (d1 + sqrt(d1*d1 + 4 * b)) / b

    x = 0.0
    accept = false
    while (!accept) 
        x = b + Distributions.rand(rng, Distributions.Exponential(1.0)) / c0
        u = Distributions.rand(rng)
        l_rho = d3 * _log_c(x) - x * (1-c0)
        l_M   = d3 * _log_c(d3 / (1-c0)) - d3
        accept = _log_c(u) <= (l_rho - l_M)
    end

    return trunc * (x/b)
end

# random number generation from the gamma with right truncation point#
# i.e. TG^-(α,θ,0.0,upper)
function Distributions.rand(rng::Distributions.AbstractRNG, d::Distributions.Truncated{Distributions.Gamma{T},Distributions.Continuous}) where {T<:Real}
    if minimum(Distributions.untruncated(d)) == d.lower && maximum(Distributions.untruncated(d)) == d.upper
        Distributions.rand(rng, Distributions.Gamma(d.α, d.θ))
    elseif minimum(Distributions.untruncated(d)) == d.lower
        # right truncated
        d1 = TruncatedGamma(d.α, d.θ * d.upper, 0.0, d.upper)
        return rng_right_tgamma_t1(rng, d1) * d.upper      
    else
        # left truncated
        return ltgamma(rng, d)
    end
end



