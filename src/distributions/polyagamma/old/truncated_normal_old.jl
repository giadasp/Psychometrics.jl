#From Chopin's algorithm in N. Chopin, "Fast simulation of truncated Gaussian distributions",
#Stat Comput (2011) 21:275-288

#------------------------------------------------------------
#include("constants.jl")

const N_tn = 4001# Index of the right tail
const yl0 = 0.053513975472# y_l of the leftmost rectangle
const ylN = 0.000914116389555# y_l of the rightmost rectangle


#------------------------------------------------------------
# Compute y_l from y_k
function yl(k::Int64)
    if (k == 1)
        return yl0
    elseif (k == N_tn)
        return ylN
    elseif (k <= 1954)
        return YU[k-1]
    else
        return YU[k+1]
    end
end
#------------------------------------------------------------
# Rejection algorithm with lower truncated exponential proposal
function rtexp(rng::Distributions.AbstractRNG, a::Float64, b::Float64)
    stop = false
    TWOASQ = 2 * (a^2)
    expab = _exp_c(-a * (b - a)) - 1
    z = 0.0
    while !stop
        z = _log_c(1 + Distributions.rand(rng) * expab)
        e = -_log_c(Distributions.rand(rng))
        stop = (TWOASQ * e > z^2)
    end
    return a - z / a
end


# Design variables
const XMIN = -2.00443204036# Left bound
const XMAX = 3.48672170399# Right bound
const KMIN = 5# if kb-ka < kmin then use lower rejection algorithm
const INVH = 1631.73284006# = 1/h, h being the minimal interval range
const I0 = 3271# = - floor(X_TN(0)/h)
const ALPHA = 1.837877066409345 # = _log_c(2*__PI)
const XSIZE = size(X_TN, 1)
const SQ2 = 7.071067811865475e-1# = 1/sqrt(2)
const SQPI = 1.772453850905516# = sqrt(__PI)

function Distributions.std(
    d::Distributions.Truncated{Distributions.Normal{T},Distributions.Continuous},
) where {T<:Real}
    d.untruncated.σ
end

function Distributions.var(
    d::Distributions.Truncated{Distributions.Normal{T},Distributions.Continuous},
) where {T<:Real}
    d.untruncated.σ^2
end

"""
Distributions.rand(rng::Distributions.AbstractRNG, d::Distributions.Truncated{Distributions.Normal{T},Distributions.Continuous}) where {T<:Real}

# Description
Pseudorandom numbers from lower and/or upper truncated Gaussian distribution.
The Gaussian has parameters μ (default 0) and σ (default 1) and is truncated on the interval `[lower, upper]`.

# Output
Returns a sample of `d`.
"""

function Distributions.rand(
    rng::Distributions.AbstractRNG,
    d::Distributions.Truncated{Distributions.Normal{T},Distributions.Continuous},
) where {T<:Real}
    stop = false
    d0 = d.untruncated
    μ = Distributions.mean(d0)
    σ = Distributions.std(d0)
    lower = d.lower
    upper = d.upper
    # Scaling
    if (μ != 0 || σ != 1)
        lower = (lower - μ) / σ
        upper = (upper - μ) / σ
    end

    #-----------------------------

    # Check if lower < upper
    if lower >= upper
        error("upper must be greater than lower!")
        # Check if |lower| < |upper|
    elseif (abs(lower) > abs(upper))
        r = -rand(rng, Distributions.TruncatedNormal(μ, σ, -upper, -lower))# Pair (r,p)
    # If lower in the right tail (lower > XMAX), use rejection algorithm with lower truncated exponential proposal
    elseif (lower > XMAX)
        r = rtexp(rng, lower, upper)
        # If lower in the left tail (lower < XMIN), use rejection algorithm with lower Gaussian proposal
    elseif (lower > XMIN)
        stop = false
        while !stop
            r = Distributions.rand(Distributions.Normal(0, 1))
            stop = (r >= lower) && (r <= upper)
        end
        # In other cases (XMIN < lower < XMAX), use Chopin's algorithm
    else
        # Compute ka
        i = max(1, Int64(I0 + (floor(lower * INVH) + 1)))
        ka = NCELL[i]

        # Compute kb
        if (upper >= XMAX)
            kb = N_tn
        else
            i = max(1, Int64(I0 + (floor(upper * INVH) + 1)))
            kb = NCELL[i]
        end

        # If |upper-lower| is small, use rejection algorithm with lower truncated exponential proposal
        if (abs(kb - ka) < KMIN)
            r = rtexp(rng, lower, upper)
            stop = true
        end

        while (!stop)
            # Sample integer between ka and kb
            k = Int64(floor(Distributions.rand(rng) * (kb - ka + 1)) + ka + 1) #was floor
            if (k >= N_tn)
                # Right tail
                lbound = X_TN[XSIZE]
                z = -_log_c(Distributions.rand(rng))
                e = -_log_c(Distributions.rand(rng))
                z = z / lbound
                if ((z^2 <= 2 * e) && (z < upper - lbound))
                    # Accept this proposition, otherwise reject
                    r = lbound + z
                    stop = true
                end
            elseif (k <= ka + 1) || ((k >= kb - 1) && (upper < XMAX))

                # Two leftmost and rightmost regions
                sim = X_TN[k] + (X_TN[k+1] - X_TN[k]) * Distributions.rand(rng)

                if ((sim >= lower) && (sim <= upper))
                    # Accept this proposition, otherwise reject
                    simy = YU[k] * Distributions.rand(rng)
                    if ((simy < yl(k)) || (sim * sim + 2 * _log_c(simy) + ALPHA) < 0)
                        r = sim
                        stop = true
                    end
                end
            else # All the other boxes
                u = Distributions.rand(rng)
                simy = YU[k] * u
                d = X_TN[k+1] - X_TN[k]
                ylk = yl(k)
                if (simy < ylk)# That's what happens most of the time 
                    r = X_TN[k] + u * d * YU[k] / ylk
                    stop = true
                else
                    sim = X_TN[k] + d * Distributions.rand(rng)
                    # Otherwise, check you're below the pdf curve
                    if ((sim * sim + 2 * _log_c(simy) + ALPHA) < 0)
                        r = sim
                        stop = true
                    end
                end
            end
        end
    end

    #-----------------------------

    # Scaling
    if (μ != 0 || σ != 1)
        r = r * σ + μ
    end
    # Compute the probability
    #Z = sqpi * sq2 * σ * ( gsl_sf_erf(upper *s q2) - gsl_sf_erf(lower*sq2) )
    #p = _exp_c(-(((r-μ)/σ)^2)/2) / Z 

    return r
end
