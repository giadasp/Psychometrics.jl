#  Pseudorandom numbers from a truncated Gaussian distribution.

#  This implements the zigguart algorithm from
#  N. Chopin, "Fast simulation of truncated Gaussian distributions",
#  Stat Comput (2011) 21:275-288
#
#  The code is based on the implementation by Guillaume Dollé, Vincent Mazet
#  available from http:#miv.u-strasbg.fr/mazet/rtnorm/. In particular, the
#  rtchopin_twosided and rtchopin_onesided are derived from their code. 
#  Andreas Dzemski adapted these functions to the R environment on Sep 13, 2018 by 
#  changing the random number generation using the GNU Scientific library to 
#  native R functions.
#

#------------------------------------------------------------
# Pseudorandom numbers from a truncated Gaussian distribution
# The Gaussian has parameters μ (default 0) and σ (default 1)
# and is truncated on the interval [lower,upper].

include("constants.jl")

function Distributions.rand(
    rng::Distributions.AbstractRNG,
    d::Distributions.Truncated{Distributions.Normal{T},Distributions.Continuous},
) where {T<:Real}
    lower = d.lower
    upper = d.upper
    mu = d.untruncated.μ
    sigma = d.untruncated.σ

    a = (lower - mu) / sigma
    b = (upper - mu) / sigma

    # return(lower[0])

    if (abs(a) > abs(b))

        r = -Distributions.rand(rng, Distributions.TruncatedNormal(mu, sigma, -b, -a))

    elseif (a == -Inf)
        r = Distributions.rand(rng, Distributions.Normal(0, 1))
        # If a in the right tail (a > xmax), use rejection algorithm with a truncated exponential proposal
    elseif (a > _XMAX)
        if (b == Inf)
            r = c_rtexp_onesided(rng, a)
        else
            r = c_rt_exp_c(rng, a, b)
        end
        # If a in the left tail (a < xmin), use rejection algorithm with a Gaussian proposal
    elseif (a < _XMIN)

        if (b == Inf)
            r = c_rtnaive_onesided(rng, a)
        else
            r = c_rtnaive(rng, a, b)
        end
        # In other cases (xmin < a < xmax), use Chopin's algorithm
    else

        if (b == Inf)
            r = rtchopin_onesided(rng, a)
        else
            r = rtchopin_twosided(rng, a, b)
        end
    end

    return sigma * r + mu
end

#------------------------------------------------------------
# Compute y_l from y_k #IDX_OK
function yl(k::Int64)
    if (k == 1) # 1 was 0
        return 0.053513975472   # y_l of the leftmost rectangle
    elseif (k == _N - 1) #_N was _N - 1
        return 0.000914116389555  # y_l of the rightmost rectangle
    elseif (k <= 1954) # 1954 was 1953
        return _YU[k-1]
    else
        return _YU[k+1]
    end
end

#------------------------------------------------------------
# Rejection algorithm with a truncated exponential proposal
function c_rt_exp_c(rng::Distributions.AbstractRNG, a::Float64, b::Float64)

    stop = false
    twoasq = 2 * a^2
    expab = _exp_c(-a * (b - a)) - 1
    z = 0.0
    while (!stop)
        z = _log_c(1 + Random.rand(rng) * expab)
        e = -_log_c(Random.rand(rng))
        stop = (twoasq * e > z^2)
    end
    return a - z / a
end

#------------------------------------------------------------
# One-sided rejection algorithm with a truncated exponential proposal
function c_rtexp_onesided(rng::Distributions.AbstractRNG, a::Float64)

    stop = false
    lambda = 0.5 * (a + sqrt(a^2 + 4))
    z = 0.0
    while (!stop)
        z = a - _log_c(Random.rand(rng)) / lambda
        stop = (-2 * _log_c(Random.rand(rng)) > (z - lambda)^2)
    end
    return z
end

#------------------------------------------------------------
# Naive accept reject (two-sided)
function c_rtnaive(rng::Distributions.AbstractRNG, a::Float64, b::Float64)
    stop = false
    r = 0.0
    while (!stop)
        r = Distributions.rand(rng, Distributions.Normal(0, 1))
        stop = (r >= a) && (r <= b)
    end
    return r
end

#------------------------------------------------------------
# Naive accept reject  (one-sided)
function c_rtnaive_onesided(rng::Distributions.AbstractRNG, a::Float64)
    stop = false
    r = 0.0
    while (!stop)
        r = Distributions.rand(rng, Distributions.Normal(0, 1))
        stop = (r >= a)
    end

    return r
end

#------------------------------------------------------------
# Rejection algorithm with a truncated Rayleigh proposal
function rtrayleigh(rng::Distributions.AbstractRNG, a::Float64)

    stop = false
    asq = (a^2) * 0.5
    v = 0.0
    x = 0.0
    while (!stop)
        v = Random.rand(rng)
        x = asq - _log_c(Random.rand(rng))
        stop = (v^2 * x < a)
    end

    return sqrt(2 * x)
end

#------------------------------------------------------------
# Chopin's one-sided ziggurat algorithm
function rtchopin_onesided(rng::Distributions.AbstractRNG, a::Float64)

    stop = false

    if (a < _XMIN)  #if a is too small, use simple reject algorithm

        r = c_rtnaive_onesided(rng, a)
        stop = true

    elseif (a > _XMAX) # if a is too large, use Devroye's algorithm

        r = c_rtexp_onesided(rng, a)
        stop = true

    else
        i = Int64(_I0 + floor(a * _INVH)) #was without +1
        ka = _NCELL[i] + 1#was without +1
        while (!stop)
            # sample integer between ka and N 
            k = Int64(floor(Random.rand(rng) * (_N - ka + 1)) + ka)  # was without +1
            if (k >= _N)  #right tail (last box on the right) # was == 
                r = c_rtexp_onesided(rng, _XCONST[_N])
                stop = true
            elseif (k <= ka + 1)
                sim = _XCONST[k] + (_XCONST[k+1] - _XCONST[k]) * Random.rand(rng)
                if (sim >= a)
                    simy = _YU[k] * Random.rand(rng)
                    if ((simy < yl(k)) || (sim * sim + 2.0 * _log_c(simy) + _ALPHA) < 0)
                        r = sim
                        stop = true
                    end
                end
            else

                u = Random.rand(rng)
                simy = _YU[k] * u
                d = _XCONST[k+1] - _XCONST[k]
                ylk = yl(k)
                if (simy < ylk)  # That's what happens most of the time
                    r = _XCONST[k] + u * d * _YU[k] / ylk
                    stop = true
                else

                    sim = _XCONST[k] + d * Random.rand(rng)

                    # Otherwise, check you're below the pdf curve
                    if ((sim * sim + 2 * _log_c(simy) + _ALPHA) < 0)

                        r = sim
                        stop = true
                    end
                end
            end
        end # end while loop
    end

    return r
end #IDXOK

#------------------------------------------------------------
# Chopin's two-sided ziggurat algorithm
function rtchopin_twosided(rng::Distributions.AbstractRNG, a::Float64, b::Float64)

    kmin = 5                                 # if kb-ka < kmin then use a rejection algorithm
    stop = false

    # Compute ka
    i = Int64(_I0 + floor(a * _INVH)) # was without +1
    ka = _NCELL[i] + 1 #was without +1

    # Compute kb
    if (b >= _XMAX)
        kb = _N
    else
        i = Int64(_I0 + floor(b * _INVH))  # was without +1
        kb = _NCELL[i] + 1 # was without +1
    end

    # If |b-a| is small, use rejection algorithm with a truncated exponential proposal
    if (abs(kb - ka) < kmin)
        r = c_rt_exp_c(rng, a, b)
        stop = true
    end

    while (!stop)

        # Sample integer between ka and kb
        k = Int64(ceil(Distributions.rand(rng, Distributions.Uniform(ka, kb))))  # was without +1

        if (k >= _N) # was ==

            # Right tail
            lbound = _XCONST[_N]
            z = -_log_c(Random.rand(rng))
            e = -_log_c(Random.rand(rng))
            z = z / lbound

            if ((z^2 <= 2 * e) && (z < b - lbound))

                # Accept this proposition, otherwise reject
                r = lbound + z
                stop = true
            end

        elseif ((k <= ka + 1) || (k >= kb - 1 && b < _XMAX))


            # Two leftmost and rightmost regions
            sim = _XCONST[k] + (_XCONST[k+1] - _XCONST[k]) * Random.rand(rng)

            if ((sim >= a) && (sim <= b))

                # Accept this proposition, otherwise reject
                simy = _YU[k] * Random.rand(rng)
                if ((simy < yl(k)) || (sim * sim + 2 * _log_c(simy) + _ALPHA) < 0)

                    r = sim
                    stop = true
                end
            end
        else # All the other boxes

            u = Random.rand(rng)
            simy = _YU[k] * u
            d = _XCONST[k+1] - _XCONST[k]
            ylk = yl(k)
            if (simy < ylk)  # That's what happens most of the time

                r = _XCONST[k] + u * d * _YU[k] / ylk
                stop = true
            else

                sim = _XCONST[k] + d * Random.rand(rng)

                # Otherwise, check you're below the pdf curve
                if ((sim * sim + 2 * _log_c(simy) + _ALPHA) < 0)

                    r = sim
                    stop = true
                end
            end
        end
    end

    return r
end
