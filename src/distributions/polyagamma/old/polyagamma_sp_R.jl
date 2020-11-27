## Copyright 2013 Nick Polson, James Scott, and Jesse Windle.

## This file is part of BayesLogit, distributed under the GNU General Public
## License version 3 or later and without ANY warranty, implied or otherwise.

################################################################################
## CALCULATE SADDLE POINT APPROXIMATION ##
################################################################################

mutable struct FD
    val::Float64
    der::Float64
    FD() = new(0.0, 0.0)
    FD(val, der) = new(val, der)
end

mutable struct Line
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
    PolyaGammaSPSampler(h, z) = new(h, z, 1000)
    PolyaGammaSPSampler(h, z, maxiter) = new(h, z, maxiter)
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

function sp_approx_1(x::Float64, n::Float64, z::Float64)
    v = v_eval(x, 1e-9, 1000)
    u = v / 2
    t = u + z^2 / 2
    m = y_eval(-z^2)

    phi = _log_c(cosh(z)) - _log_c(cos_rt_sp(v)) - t * x

    K2 = x^2 + (1 - x) / (2 * u)
    if (u < 1e-5) && (u > -1e-5)
        K2 = x^2 - (1 / 3) - (2 / 15) * (2 * u)
    end
    spa = sqrt((n / (2 * __PI)) / K2) * _exp_c(phi * n)
    return spa
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
    u = v / 2
    t = u + (z^2 / 2)
    return FD(_log_c(cosh(abs(z))) - _log_c(cos_rt_sp(v)) - t * x, -t)
end

function tangent_lines_eta(x::Float64, z::Float64, mid::Float64)
    eta = FD()
    delta = FD()
    phi = FD()

    phi = phi_func_sp(x, z)
    delta = delta_func_sp(x, mid)

    eta.val = phi.val - delta.val
    eta.der = phi.der - delta.der

    return Line(eta.der, eta.val - eta.der * x)
end
################################################################################
## rrtigauss ##
################################################################################

## pigauss - cumulative distribution function for Inv-Gauss(mu, lambda).
## NOTE: note using mu = mean, using Z = 1/mu.
##------------------------------------------------------------------------------
#Distributions.pdf(InverseGaussian(Z, lambda), x)

function rrtinvch2_1(rng::Distributions.AbstractRNG, scale::Float64, trnc::Float64)
    R = trnc / scale
    #E = rtnorm(1, 0, 1, left=1/sqrt(R), right=Inf)
    E = Distributions.rand(rng, Distributions.TruncatedNormal(0.0, 1.0, 1 / sqrt(R), Inf))
    X = scale / E^2
    return X
end

function rigauss_1(rng::Distributions.AbstractRNG, mu::Float64, lambda::Float64)
    nu = Random.randn(rng)
    y = nu^2
    x =
        mu + 0.5 * mu^2 * y / lambda -
        0.5 * mu / lambda * sqrt(4 * mu * lambda * y + (mu * y)^2)
    if (Random.rand(rng) > mu / (mu + x))
        x = mu^2 / x
    end
    x
end

function rrtigauss_1(
    rng::Distributions.AbstractRNG,
    mu::Float64,
    lambda::Float64,
    trnc::Float64,
)
    ## trnc is truncation point
    accept = false
    X = trnc + 1
    if (trnc < mu)
        alpha = 0.0
        while (!accept)
            X = rrtinvch2_1(rng, lambda, trnc)
            l_alpha = -0.5 * lambda / mu^2 * X
            accept = _log_c(Random.rand(rng)) < l_alpha
        end
        ## cat("rtigauss.ch, part i:", X, "\n");
    else  ## trnc >= mu
        while (X > trnc)
            Y = Random.randn(rng)^2
            X =
                mu + 0.5 * mu^2 / lambda * Y -
                0.5 * mu / lambda * sqrt(4 * mu * lambda * Y + (mu * Y)^2)
            if (Random.rand(rng) > (mu / (mu + X)))
                X = mu^2 / X
            end
        end
    end
    return X
end
function rltgamma_dagpunar_1(
    rng::Distributions.AbstractRNG,
    shape::Float64,
    rate::Float64,
    trnc::Float64,
)
    ## y ~ Ga(shape, rate, trnc)
    ## x = y/t
    ## x ~ Ga(shape, rate t, 1)
    a = shape
    b = rate * trnc

    if (shape == 1)
        return (Random.randexp(rng) / rate + trnc)
    end

    d1 = b - a
    d3 = a - 1
    c0 = 0.5 * (d1 + sqrt(d1^2 + 4 * b)) / b

    x = 0
    accept = false

    while (!accept)
        x = b + Random.randexp(rng) / c0
        u = Random.rand(rng)

        l_rho = d3 * _log_c(x) - x * (1 - c0)
        l_M = d3 * _log_c(d3 / (1 - c0)) - d3

        accept = _log_c(u) <= (l_rho - l_M)
    end
    return trnc * x / b
end
function pigauss(x::Float64, Z::Float64, lambda::Float64)

    ## I believe this works when Z = 0
    ##Z = 1/mu
    b = sqrt(lambda / x) * (x * Z - 1)
    a = sqrt(lambda / x) * (x * Z + 1) * -1.0
    y =
        Distributions.cdf(Distributions.Normal(), b) +
        exp(2 * lambda * Z) * Distributions.cdf(Distributions.Normal(), a)
    # y2 = 2 * pnorm(-1.0 / sqrt(x));
    return y
end
##------------------------------------------------------------------------------
## Generate sample for J^*(n,z) using SP approx.
function sp_sampler_1(
    rng::Distributions.AbstractRNG,
    n::Float64,
    z::Float64;
    maxiter = 1000,
)

    xl = y_eval(-z^2)
    mid = xl * 1.1
    xr = xl * 1.2
    v_mid = v_eval(mid, 1e-9, maxiter)
    K2_mid = mid^2 + (1 - mid) / v_mid
    al = mid^3 / K2_mid
    ar = mid^2 / K2_mid
    ll = tangent_lines_eta(xl, z, mid)
    lr = tangent_lines_eta(xr, z, mid)
    #tl = tangent_lines_eta(xl, xr, z, mid)
    rl = -ll.slope
    rr = -lr.slope
    il = copy(ll.icept)
    ir = copy(lr.icept)

    wl =
        sqrt(al) *
        _exp_c(-n * sqrt(2 * rl) + n * il + 0.5 * n - 0.5 * n * (1 - 1 / mid)) *
        pigauss(mid, sqrt(2 * rl), n)
    #Distributions.cdf(Distributions.InverseGaussian(sqrt(2*rl), n), mid)  
    wr =
        sqrt(ar * n / (__PI * 2)) *
        _exp_c(-n * _log_c(n * rr) + n * ir - n * _log_c(mid) + loggamma(n)) *
        (1 - Distributions.cdf(Distributions.Gamma(n, 1 / (n * rr)), mid))

    wt = wl + wr
    pl = wl / wt

    go = true
    iter = 0
    X = 2
    FX = 0

    while (go && iter <= maxiter)
        iter = iter + 1

        if (wt * Random.rand(rng) < wl)
            ## sample left
            X = rrtigauss_1(rng, 1 / sqrt(2 * rl), n, mid)
            ## while (X > mid) X = rigauss.1(1/sqrt(2*rl), n)
            phi_ev = n * (-rl * X + il) + 0.5 * n * ((1 - 1 / X) - (1 - 1 / mid))
            FX = sqrt(al * n / (2 * __PI)) * X^(-1.5) * _exp_c(phi_ev)

        else
            ## sample right
            X = rltgamma_dagpunar_1(rng, n, n * rr, mid)

            #X = Distributions.rand(rng, TruncatedGamma(n, 1 / (n * rr), mid, Inf))

            phi_ev = n * (-rr * X + ir) + n * (_log_c(X) - _log_c(mid))
            FX = sqrt(ar * 0.5 * n / __PI) * _exp_c(phi_ev) / X
        end

        spa = sp_approx_1(X, n, z)

        if (FX * Random.rand(rng) < spa)
            go = false
        end
    end

    return X
end


## draw with saddle point method
function Distributions.rand(rng::Distributions.AbstractRNG, s::PolyaGammaSPSampler)
    n = s.h
    z = 0.5 * s.z

    x = sp_sampler_1(rng, n, z)

    return 0.25 * n * x
end
