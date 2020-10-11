# -*- mode: c++ c-basic-offset: 4 -*-
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



#include "inverse_gaussian.h"
#include "simple_RNG_wrapper.h"

function igauss(rng::Distributions.AbstractRNG, mu::Float64, lambda::Float64)
    # See R code for specifics.
    mu2 = mu * mu
    Y = Distributions.rand(rng, Distributions.Distributions.Normal())
    Y *= Y
    W = mu + 0.5 * mu2 * Y / lambda
    X = W - sqrt(W * W - mu2)
    if (Distributions.rand(rng) > mu / (mu + X))
        X = mu2 / X
    end
    return X
end


function p_igauss(x::Float64, mu::Float64, lambda::Float64)
    # z = 1 / mean
    z = 1 / mu
    b = sqrt(lambda / x) * (x * z - 1)
    a = sqrt(lambda / x) * (x * z + 1) * -1.0
    # double y = p_Distributions.rand(rng, Distributions.Normal())(b, false) + exp(2 * lambda * z) * p_Distributions.rand(rng, Distributions.Normal())(a, false)
    y =
        Distributions.cdf(Distributions.Normal(), b) +
        _exp_c(2 * lambda * z + _log_c(Distributions.cdf(Distributions.Normal(), a)))
    return y
end
