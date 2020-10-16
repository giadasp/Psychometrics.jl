# (C) Nicholas Polson, James Scott, Jesse Windle, 2012-2019

# This file is part of BayesLogit.

# BayesLogit is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# BayesLogit is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# BayesLogit.  If not, see <https:#www.gnu.org/licenses/>.



#include "truncated_norm.h"
#include "simple_RNG_wrapper.h"
const RCHECK = 1000
const SQRT2PI = 2.50662827

function flat(rng::Distributions.AbstractRNG, a::Float64, b::Float64)
    return a + rand(rng) * (b - a)::Float64
end

function texpon_rate(rng::Distributions.AbstractRNG, left::Float64, rate::Float64)
    if (rate < 0)
        println("texpon_rate: rate < 0, return 0")
    end
    # return left - _log_c(rand(rng)) / rate
    return Distributions.rand(rng, Distributions.Exponential(1 / rate)) + left::Float64
end

function texpon_rate(
    rng::Distributions.AbstractRNG,
    left::Float64,
    right::Float64,
    rate::Float64,
)
    if (left == right)
        return left
    end
    if (left > right)
        println("texpon_rate: left > right, return 0.")
        return
    end
    if (rate < 0)
        println("texpon_rate: rate < 0, return 0.")
        return 0.0
    end
    b = 1 - _exp_c(rate * (left - right))
    y = 1 - b * Distributions.rand(rng)
    return (left - _log_c(y) / rate)::Float64
end

function alphastar(left::Float64)
    return 0.5 * (left + sqrt(left * left + 4))::Float64
end

function lowerbound(left::Float64)
    astar = alphastar(left)
    lbound = left + _exp_c(0.5 * +0.5 * left * (left - astar)) / astar
    return lbound::Float64
end

function tnorm(rng::Distributions.AbstractRNG, left::Float64)
    count = 1

    if (left < 0)  # Accept/Reject Normal
        while (true)
            ppsl = Random.randn(rng)
            if (ppsl > left)
                return ppsl
            end
            count += 1
            #check_R_interupt(count++)
            if (count > RCHECK * 1000)
                println("left < 0 count: %i\n", count)
            end
        end
    else  # Accept/Reject Exponential
        # return tnorm_tail(left) # Use Devroye.
        astar = alphastar(left)
        while (true)
            ppsl = texpon_rate(rng, left, astar)
            rho = _exp_c(-0.5 * (ppsl - astar) * (ppsl - astar))
            if (Distributions.rand(rng) < rho)
                return ppsl
            end
            count += 1
            #check_R_interupt(count++)
            if (count > RCHECK * 1000)
                println("left > 0 count: ", count)
            end
        end
    end
end

function tnorm(rng::Distributions.AbstractRNG, left::Float64, right::Float64)
    # The most difficult part of this algorithm is figuring out all the
    # various cases.  An outline is summarized in the Appendix.

    # Check input

    if (right < left)
        println("Warning: left: $left, right:$right.")
        println("tnorm: parameter problem.", 0.5 * (left + right))
    end

    count = 1

    if (left >= 0)
        lbound = lowerbound(left)
        if (right > lbound)  # Truncated Exponential.
            astar = alphastar(left)
            while (true)
                ppsl = texpon_rate(rng, left, right, astar)
                rho = _exp_c(-0.5 * (ppsl - astar) * (ppsl - astar))
                if (rand(rng) < rho)
                    return ppsl
                end
                if (count > RCHECK * 10)
                    println("left >= 0, right > lbound count: ", count)
                end
            end
        else
            while (true)
                ppsl = flat(rng, left, right)
                rho = _exp_c(0.5 * (left * left - ppsl * ppsl))
                if (rand(rng) < rho)
                    return ppsl
                end
                count += 1
                #check_R_interupt(count++)
                if (count > RCHECK * 10)
                    println("left >= 0, right <= lbound count: ", count)
                end
            end
        end
    elseif (right >= 0)
        if ((right - left) < SQRT2PI)
            while (true)
                ppsl = flat(rng, left, right)
                rho = _exp_c(-0.5 * ppsl * ppsl)
                if (rand(rng) < rho)
                    return ppsl
                end
                count += 1
                #check_R_interupt(count++)
                if (count > RCHECK * 10)
                    println("First, left < 0, right >= 0, count:", count)
                end
            end
        else
            while (true)
                ppsl = Random.randn(rng)
                if (left < ppsl && ppsl < right)
                    return ppsl
                end
                count += 1
                #check_R_interupt(count++)
                if (count > RCHECK * 10)
                    println("Second, left < 0, right > 0, count: ", count)
                end
            end
        end
    else
        return -1 * tnorm(rng, -1.0 * right, -1.0 * left)
    end
end

function tnorm(rng::Distributions.AbstractRNG, left::Float64, mu::Float64, sd::Float64)
    newleft = (left - mu) / sd
    return (mu + tnorm(rng, newleft) * sd)::Float64
end

function tnorm(
    rng::Distributions.AbstractRNG,
    left::Float64,
    right::Float64,
    mu::Float64,
    sd::Float64,
)
    if (left == right)
        return left
    end

    newleft = (left - mu) / sd
    newright = (right - mu) / sd

    # I want to check this here as well so we can see what the input was.
    # It may be more elegant to try and catch tdraw.
    if (newright < newleft)
        println("left, right, mu, sd: ", left, right, mu, sd)
        println("nleft, nright: ", newleft, newright)
        println("tnorm: parameter problem.", 0.5 * (left + right))
    end

    tdraw = tnorm(rng, newleft, newright)
    draw = mu + tdraw * sd

    # It may be the case that there is some numerical error and that the draw
    # ends up out of bounds.
    if (draw < left || draw > right)
        println("Error in tnorm: draw not in bounds.\n")
        println("left, right, mu, sd: ", left, right, mu, sd)
        println("nleft, nright, tdraw, draw: \n", newleft, newright, tdraw, draw)
        println("Aborting and returning average of left and right.\n", 0.5 * (left + right))
    end

    return draw::Float64
end

function rtinvchi2(rng::Distributions.AbstractRNG, scale::Float64, trunc::Float64)
    R = trunc / scale
    # double X = 0.0
    # # I need to consider using a different truncated normal sampler.
    # double E1 = r.expon_rate(1.0) double E2 = r.expon_rate(1.0)
    # while ( (E1*E1) > (2 * E2 / R)) {
    #   # printf("E %g %g %g %g\n", E1, E2, E1*E1, 2*E2/R)
    #   E1 = r.expon_rate(1.0) E2 = r.expon_rate(1.0)
    # }
    # # printf("E %g %g \n", E1, E2)
    # X = 1 + E1 * R
    # X = R / (X * X)
    # X = scale * X
    E = tnorm(rng, 1 / sqrt(R))
    X = scale / (E * E)
    return X
end
