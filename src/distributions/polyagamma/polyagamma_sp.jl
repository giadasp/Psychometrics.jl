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

struct FD
    val::Float64
    der::Float64
    FD() = new(0.0, 0.0)
    FD(val, der) = new(val, der)
end

struct Line
    slope::Float64
    icept::Float64
    Line() = new(0.0, 0.0)
    Line(slope, icept) = new(slope, icept)
end

struct PolyaGammaSPSampler <:
       Distributions.Sampleable{Distributions.Univariate,Distributions.Continuous}
    h::Float64
    z::Float64
    maxiter::Int64
    PolyaGammaSPSampler(h, z) = new(h, z, 200)
    PolyaGammaSPSampler(h, z, maxiter) = new(h, z, maxiter)
end


function _rtigauss_sp(
    rng::Distributions.AbstractRNG,
    mu::Float64,
    lambda::Float64,
    trunc::Float64,
)
    X = trunc + 1
    if (trunc < mu)
        alpha = 0.0
        while Random.rand(rng) > alpha
            X = rtinvchi2_sp(rng, lambda, trunc)
            alpha = _exp_c(lambda / (2 * mu^2) * X)
        end
    else
        while X > trunc
            X = _igauss_sp(rng, mu, lambda)
        end
    end
    return X
end

function alphastar(left::Float64)
    return 0.5 * (left + sqrt(left * left + 4))
end

function _texpon_rate(left::Float64, rate::Float64)
    if (rate < 0)
        println("texpon_rate: rate < 0, return 0", 0.0)
        return 0.0
    end
    return Random.randexp(1 / rate) + left
end

function _tnorm_left_sp(rng::Distributions.AbstractRNG, left::Float64)
    if (left < 0)
        while (true)
            ppsl = Random.randn(rng)
            if (ppsl > left)
                return ppsl
            end
        end
    else
        astar = alphastar(left)
        while (true)
            ppsl = _texpon_rate(left, astar)
            rho = _exp_c(-0.5 * (ppsl - astar)^2)
            if (Random.rand(rng) < rho)
                return ppsl
            end
        end
    end
end

function rtinvchi2_sp(rng::Distributions.AbstractRNG, scale::Float64, trunc::Float64)
    R = trunc / scale
    E = _tnorm_left_sp(rng, 1 / sqrt(R))
    X = scale / (E * E)
    return X
end

function _igauss_sp(rng::Distributions.AbstractRNG, mu::Float64, lambda::Float64)
    mu2 = mu * mu
    Y = (Random.randn(rng)^2)
    W = mu + 0.5 * mu2 * Y / lambda
    X = W - sqrt(W * W - mu2)
    if (Random.rand(rng) > (mu / (mu + X)))
        X = mu2 / X
    end
    return X
end

#y_eval
# function y_func_sp(v::Float64)
#     tol = 1e-6
#     y = 0.0
#     r = sqrt(abs(v))
#     if (v > tol)
#         y = tan(r) / r
#     elseif (v < -1 * tol)
#         y = tanh(r) / r
#     else
#         y = 1 + (1 / 3) * v + (2 / 15) * (v^2) + (17 / 315) * (v^3)
#     end
#     return y
# end

function cos_rt_sp(v::Float64)
    y = 0.0
    r = sqrt(abs(v))
    if (v >= 0)
        y = cos(r)
    else
        y = cosh(r)
    end
    return y
end

function delta_func_sp(x::Float64, mid::Float64)
    if (x >= mid)
        return FD(_log_c(x) - _log_c(mid), 1 / x)
    else
        return FD(0.5 * (1 - 1 / x) - 0.5 * (1 - 1 / mid), 0.5 / (x^2))
    end
end

function phi_func_sp(x::Float64, z::Float64)
    v = v_eval(x, 1e-9, 1000)
    #println("v =", v)
    u = 0.5 * v
    t = u + 0.5 * z^2
    return v, FD(_log_c(cosh(abs(z))) - _log_c(cos_rt_sp(v)) - t * x, -t)
end

function tangent_to_eta_sp(x::Float64, z::Float64, mid::Float64)
    eta = FD()
    delta = FD()
    phi = FD()

    v, phi = phi_func_sp(x, z)
    delta = delta_func_sp(x, mid)

    eta.val = phi.val - delta.val
    eta.der = phi.der - delta.der

    return Line(eta.der, eta.val - eta.der * x)
end

function sp_approx_sp(x::Float64, n::Float64, z::Float64)
    v = v_eval(x, 1e-9, 1000)
    u = 0.5 * v
    z2 = z^2
    t = u + 0.5 * z2
    # double m  = y_func(-1 * z2)

    phi = _log_c(cosh(z)) - _log_c(cos_rt_sp(v)) - t * x

    K2 = 0.0
    if (abs(v) >= 1e-6)
        K2 = (x^2) + (1 - x) / v
    else
        K2 = (x^2) - 1 / 3 - (2 / 15) * v
    end
    log_spa = 0.5 * _log_c(0.5 * n / __PI) - 0.5 * _log_c(K2) + n * phi
    return _exp_c(log_spa)
    #spa = sqrt(((n / (2 * __PI))/ K2) * _exp_c(phi*n)
    #return spa
end

function ltgamma_sp(
    rng::Distributions.AbstractRNG,
    shape::Float64,
    rate::Float64,
    trunc::Float64,
)
    a = shape
    b = rate * trunc

    if (trunc <= 0)
        return 0.0
    end
    if (shape < 1)
        return 0.0
    end

    if (shape == 1)
        return Random.randexp(rng) / rate + trunc
    end

    d1 = b - a
    d3 = a - 1
    c0 = 0.5 * (d1 + sqrt(d1 * d1 + 4 * b)) / b
    x = 0.0
    accept = false

    while (!accept)
        x = b + Random.randexp(rng) / c0
        u = Random.rand(rng)

        l_rho = d3 * _log_c(x) - x * (1 - c0)
        l_M = d3 * _log_c(d3 / (1 - c0)) - d3

        accept = _log_c(u) <= (l_rho - l_M)
    end

    return trunc * (x / b)
end

function p_igauss_sp(x::Float64, mu::Float64, lambda::Float64)
    z = 1 / mu
    b = sqrt(lambda / x) * (x * z - 1)
    a = -sqrt(lambda / x) * (x * z + 1)
    return Distributions.cdf(Distributions.Normal(), b) +
           (_exp_c(2 * lambda * z) * Distributions.cdf(Distributions.Normal(), a))
end
# function pgamma(X::Float64, SHAPE::Float64, RATE::Float64)
#     shape = SHAPE
#     rate = 1/RATE
#     x = X
#     y = RCall.rcopy(RCall.R" pgamma($x, shape = $shape, rate = $rate) ")
#     return y
# end
## draw with saddle point method
function Distributions.rand(rng::Distributions.AbstractRNG, s::PolyaGammaSPSampler)
    n = copy(s.h)
    z = copy(s.z)

    if (n < 1)
        println("PolyaGammaSP::draw: n must be >= 1.0")
        return -1.0
    end
    z = 0.5 * abs(z)

    xl = y_eval(-z^2)    # Mode of phi - Left point.
    md = xl * 1.1          # Mid point.
    xr = xl * 1.2          # Right point.

    # Inflation constants
    # double vmd  = yv.v_func(md)
    vmd = v_eval(md, 1e-9, 1000)
    K2md = 0.0
    m2 = md^2

    if (abs(vmd) >= 1e-6)
        K2md = m2 + (1 - md) / vmd
    else
        K2md = m2 - (1 / 3) - (2 / 15) * vmd
    end
    al = m2 * md / K2md
    ar = m2 / K2md

    # Tangent lines info.
    ll = tangent_to_eta_sp(xl, z, md)
    lr = tangent_to_eta_sp(xr, z, md)
    rl = -ll.slope
    rr = -lr.slope
    il = copy(ll.icept)
    ir = copy(lr.icept)
    lcn = 0.5 * _log_c(0.5 * n / __PI)
    rt2rl = sqrt(2 * rl)
    # # to cross-reference R script
    # double term1, term2, term3, term4, term5
    # term1 = _exp_c((0.5 * _log_c(al))
    # term2 = _exp_c((- n * rt2rl + n * il + 0.5 * n * 1.0/md)
    # term3 = p_igauss( md, 1.0/rt2rl, n)
    # printf("l terms 1-3: %g, %g, %g\n", term1, term2, term3)

    wl =
        _exp_c(0.5 * _log_c(al) - n * rt2rl + n * il + 0.5 * n * 1 / md) *
        p_igauss_sp(md, 1 / rt2rl, n)
    #Distributions.cdf(Distributions.InverseGaussian(1.0 / rl, Float64(n)), md)
    #Distributions.cdf(Distributions.InverseGaussian(1.0 / rt2rl, Float64(n)), md)

    # # to cross-reference R script
    # term1 = _exp_c((0.5 * _log_c(ar))
    # term2 = _exp_c((lcn)
    # term3 = _exp_c((- n * _log_c(n * rr) + n * ir - n * _log_c(md))
    # term4 = _exp_c((loggamma(n))
    # term5 = (1.00 - p_gamma_rate(md, n, n*rr, false))
    # printf("r terms 1-5: %g, %g, %g, %g, %g\n", term1, term2, term3, term4, term5)
    #wr = (sqrt(ar) * sqrt(n/(2*__PI)) * (_exp_c(ir) / (md*n*rr)^n * gamma(n))) * (1.00 - Distributions.cdf(Distributions.Gamma(n, 1/(n * rr)), md))

    wr =
        _exp_c(
            0.5 * _log_c(ar) +
            lcn +
            (-n * _log_c(n * rr) + n * ir - n * _log_c(md) + loggamma(n)),
            #) * pgamma(md, n, 1/n*rr)
        ) * (1 - Distributions.cdf(Distributions.Gamma(n, 1 / (n * rr)), md))
    #(1.00 - Distributions.cdf(Distributions.Gamma(n, n*rr), md))
    # or
    #TODO p_gamma_rate problem rate/scale paramater
    # yv.upperIncompleteGamma(md, n, n*rr)

    # printf("wl, wr, lcn: %g, %g, %g\n", wl, wr, lcn)

    wt = wl + wr
    pl = wl / wt

    # Sample
    go = true
    iter = 0
    X = 2.0
    F = 0.0

    while (go && (iter <= s.maxiter))
        # Put first so check on first pass. 
        #if (iter % 1000 == 0) R_CheckUserInterrupt() end
        iter += 1
        if (Distributions.rand(rng) < pl)
            # X = Distributions.rand(
            #     rng,
            #     TruncatedInverseGaussian(1.0 / rl, Float64(n), 0.0, md),
            # )
            X = _rtigauss_sp(rng, 1 / rt2rl, n, md)
            phi_ev = n * (il - rl * X) + 0.5 * n * ((1 - 1 / X) - (1 - 1 / md))
            F = _exp_c(0.5 * _log_c(al) + lcn - 1.5 * _log_c(X) + phi_ev)
        else
            #X = Distributions.rand(rng, TruncatedGamma(n, 1 / (n * rr), md, Inf))
            X = ltgamma_sp(rng, n, n * rr, md)
            phi_ev = n * (ir - rr * X) + n * (_log_c(X) - _log_c(md))
            F = _exp_c(0.5 * _log_c(ar) + lcn + phi_ev) / X
        end

        spa = sp_approx_sp(X, n, z)

        if (F * Random.rand(rng) < spa)
            go = false
        end

    end

    return n * 0.25 * X
    #d = n * 0.25 * X
    #return iter
end
