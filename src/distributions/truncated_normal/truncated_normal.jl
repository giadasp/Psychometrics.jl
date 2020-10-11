#Pseudorandom numbers from lower truncated Gaussian distribution.
#
#This implements an extension of Chopin's algorithm detailed in
#N. Chopin, "Fast siμlation of truncated Gaussian distributions",
#Stat Comput (2011) 21:275-288
#
#Copyright (C) 2012 Guillaume Dollé, Vincent Mazet
#(LSIIT, CNRS/Université de Strasbourg)
#Version 2012-07-04, Contact: vincent.mazet@unistra.fr
#
#06/07/2012:
#- first launch of rtnorm.cpp
#
#Licence: GNU General Public License Version 2
#This program is free software you can redistribute it and/or modify it
#under the terms of the GNU General Public License as published by the
#Free Software Foundation either version 2 of the License, or (at your
#option) any later version. This program is distributed in the hope that
#it will be useful, but WITHOUT ANY WARRANTY without even the implied
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
#GNU General Public License for more details. You should have received lower
#copy of the GNU General Public License along with this program if not,
#see http:#www.gnu.org/licenses/old-licenses/gpl-2.0.txt
#
#Depends: LibGSL
#OS: Unix based system

#------------------------------------------------------------
include("constants.jl")

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
const ALPHA = 1.837877066409345 # = _log_c(2*pi)
const XSIZE = size(X_TN, 1)
const SQ2 = 7.071067811865475e-1# = 1/sqrt(2)
const SQPI = 1.772453850905516# = sqrt(pi)

#------------------------------------------------------------
# Pseudorandom numbers from lower truncated Gaussian distribution
# The Gaussian has parameters μ (default 0) and σ (default 1)
# and is truncated on the interval [lower,upper].
# Returns the random variable r.
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
        error("*** B μst be greater than A ! ***")
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
        i = max(1, Int64(I0 + floor(lower * INVH)))
        ka = NCELL[i]

        # Compute kb
        if (upper >= XMAX)
            kb = N_tn
        else
            i = max(1, Int64(I0 + floor(upper * INVH)))
            kb = NCELL[i]
        end

        # If |upper-lower| is small, use rejection algorithm with lower truncated exponential proposal
        if (abs(kb - ka) < KMIN)
            r = rtexp(rng, lower, upper)
            stop = true
        end

        while (!stop)
            # Sample integer between ka and kb
            k = clamp(Int64(floor(Distributions.rand(rng) * (kb - ka + 1)) + ka), 1, XSIZE) #was floor
            if (k == N_tn)
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
