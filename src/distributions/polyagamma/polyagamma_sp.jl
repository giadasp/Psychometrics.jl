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



#include "PolyaGammaApproxSP.h"
#include "InvertY.h"
#include <stdexcept>

#------------------------------------------------------------------------------

# double v_secant(double y, double vb, double va, double tol, int maxiter)
# {
#   double yb = y_func(vb)
#   double ya = y_func(va)

#   if (yb > ya) fprintf(stderr, "v_secant: yb(elow) > ya(above).\n")

#   int iter = 0
#   double ydiff = tol + 1.0
#   double vstar, ystar

#   while (abs(ydiff) > tol && iter < maxiter) {
#     iter = iter + 1
#     double m = (ya - yb) / (va - vb)
#     vstar = (y - yb) / m + vb
#     ystar = y_func(vstar)
#     ydiff = y - ystar
#     if (ystar < y) {
#       vb = vstar
#       yb = ystar
#     end else {
#       va = vstar
#       ya = ystar
#     end
#     # printf("y, v, ydiff: %g, %g, %g\n", ystar, vstar, ydiff)
#   end

#   if (iter >= maxiter) fprintf(stderr, "v_secant: reached maxiter.\n")

#   return vstar
# end

# double v_func(double y) {
#   double lowerb = -100
#   double upperb = 2.22

#   double v = 0.0
#   if (y < 0.1)
#     v = -1.0 / (y*y)
#   elseif (y > 8.285225) {
#     v = atan(y * 0.5 * π)
#     v = v * v
#   end
#   else
#     v = v_secant(y, lowerb, upperb, 1e-8, 10000)
#   return v
# end


mutable struct FD
    val::Float64
    der::Float64
    FD() = new(0.0, 0.0)
end

mutable struct Line
    slope::Float64
    icept::Float64
    Line() = new(0.0, 0.0)
end

struct PolyaGammaSPSampler <: Distributions.Sampleable{Distributions.Univariate,Distributions.Continuous}
    h::Float64
    z::Float64
    maxiter::Int64
end

function rtigauss_sp(
    rng::Distributions.AbstractRNG,
    mu::Float64,
    lambda::Float64,
    trunc::Float64,
)
    # mu = abs(mu)
    X = trunc + 1.00
    if (trunc < mu) # mu > t
        alpha = 0.0
        while (Distributions.rand(rng)() > alpha)
            X = rtinvchi2(rng, lambda, trunc)
            alpha = _exp_c(-0.5 * lambda / (mu * mu) * X)
        end
        # printf("rtigauss, part i: %g\n", X)
    else
        while (X > trunc)
            X = igauss(rng, mu, lambda)
        end
        # printf("rtigauss, part ii: %g\n", X)
    end
    return X::Float64
end

function y_func_sp(v::Float64)
    tol = 1e-6
    y = 0.0
    r = sqrt(abs(v))
    if (v > tol)
        y = tan(r) / r
    elseif (v < -1 * tol)
        y = tanh(r) / r
    else
        y = 1 + (1 / 3) * v + (2 / 15) * v * v + (17 / 315) * v * v * v
    end
    return y
end

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

function delta_func_sp!(delta::FD, x::Float64, mid::Float64)
    if (x >= mid)
        delta.val = _log_c(x) - _log_c(mid)
        delta.der = 1.00 / x
    else
        delta.val = 0.5 * (1 - 1.00 / x) - 0.5 * (1 - 1.00 / mid)
        delta.der = 0.5 / (x * x)
    end
end

function phi_func_sp!(phi::FD, x::Float64, z::Float64)
    # double v = yv.v_func(x)
    v = (x)
    u = 0.5 * v
    t = u + 0.5 * z * z

    phi.val = _log_c(cosh(abs(z))) - _log_c(cos_rt_sp(v)) - t * x
    phi.der = -1.00 * t

    return v
end

function tangent_to_eta_sp!(tl::Line, x::Float64, z::Float64, mid::Float64)
    eta = FD()
    delta = FD()
    phi = FD()

    v = phi_func_sp!(phi, x, z)
    delta_func_sp!(delta, x, mid)

    eta.val = phi.val - delta.val
    eta.der = phi.der - delta.der

    tl.slope = eta.der
    tl.icept = eta.val - eta.der * x

    return v
end

function sp_approx_sp(x::Float64, n::Float64, z::Float64)
    # double v  = yv.v_func(x)
    v = (x)
    u = 0.5 * v
    z2 = z * z
    t = u + 0.5 * z2
    # double m  = y_func(-1 * z2)

    phi = _log_c(cosh(z)) - _log_c(cos_rt_sp(v)) - t * x

    K2 = 0.0
    if (abs(v) >= 1e-6)
        K2 = x * x + (1 - x) / v
    else
        K2 = x * x - 1 / 3 - (2 / 15) * v
    end
    log_spa = 0.5 * _log_c(0.5 * n / π) - 0.5 * _log_c(K2) + n * phi
    return _exp_c(log_spa)
end


## draw with saddle point method
function Distributions.rand(rng::Distributions.AbstractRNG, s::PolyaGammaSPSampler)
    #int PolyaGammaApproxSP::draw(double& d, double n, double z, int maxiter)
    n = s.h
    z = s.z

    if (n < 1)
        println("PolyaGammaApproxSP::draw: n must be >= 1.0\n")
        return -1.0
    end
    z = 0.5 * abs(z)

    xl = y_func_sp(-1 * z * z)    # Mode of phi - Left point.
    md = xl * 1.01          # Mid point.
    xr = xl * 1.02          # Right point.

    # Inflation constants
    # double vmd  = yv.v_func(md)
    vmd = v_eval(md)
    K2md = 0.0

    if (abs(vmd) >= 1e-6)
        K2md = md * md + (1 - md) / vmd
    else
        K2md = md * md - 1 / 3 - (2 / 15) * vmd
    end
    m2 = md * md
    al = m2 * md / K2md
    ar = m2 / K2md

    # Tangent lines info.
    ll = Line()
    lr = Line()


    tangent_to_eta_sp!(ll, xl, z, md)
    tangent_to_eta_sp!(lr, xr, z, md)

    rl = -1.0 * ll.slope
    rr = -1.0 * lr.slope
    il = ll.icept
    ir = lr.icept
    lcn = 0.5 * _log_c(0.5 * n / π)
    rt2rl = sqrt(2 * rl)
    # # to cross-reference R script
    # double term1, term2, term3, term4, term5
    # term1 = _exp_c((0.5 * _log_c(al))
    # term2 = _exp_c((- n * rt2rl + n * il + 0.5 * n * 1.0/md)
    # term3 = p_igauss( md, 1.0/rt2rl, n)
    # printf("l terms 1-3: %g, %g, %g\n", term1, term2, term3)

    wl =
        _exp_c(0.5 * _log_c(al) - n * rt2rl + n * il + 0.5 * n * 1.0 / md) *
        p_igauss(md, 1.0 / rt2rl, n)

    # # to cross-reference R script
    # term1 = _exp_c((0.5 * _log_c(ar))
    # term2 = _exp_c((lcn)
    # term3 = _exp_c((- n * _log_c(n * rr) + n * ir - n * _log_c(md))
    # term4 = _exp_c((loggamma(n))
    # term5 = (1.00 - p_gamma_rate(md, n, n*rr, false))
    # printf("r terms 1-5: %g, %g, %g, %g, %g\n", term1, term2, term3, term4, term5)

    wr =
        _exp_c(
            0.5 * _log_c(ar) +
            lcn +
            (-n * _log_c(n * rr) + n * ir - n * _log_c(md) + loggamma(n)),
        ) * (1.00 - Distributions.cdf(Distributions.Gamma(n, 1 / (n * rr)), md))
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

    while (go && iter < s.maxiter)
        # Put first so check on first pass. 
        #if (iter % 1000 == 0) R_CheckUserInterrupt() end
        iter += 1
        if (Distributions.rand(rng)() < pl)
            X = rtigauss(rng, 1.0 / rt2rl, n, md)
            phi_ev = n * (il - rl * X) + 0.5 * n * ((1.0 - 1.0 / X) - (1.0 - 1.0 / md))
            F = _exp_c(0.5 * _log_c(al) + lcn - 1.05 * _log_c(X) + phi_ev)
        else
            X = Distributions.rand(rng, Distributions.TruncatedGamma(n, 1/(n * rr), md, Inf))
            phi_ev = n * (ir - rr * X) + n * (_log_c(X) - _log_c(md))
            F = _exp_c(0.5 * _log_c(ar) + lcn + phi_ev) / X
        end

        spa = sp_approx_sp(X, n, z)

        if (F * Distributions.rand(rng)() < spa)
            go = false
        end

    end

    # return n * 0.25 * X
    d = n * 0.25 * X
    return iter
end

