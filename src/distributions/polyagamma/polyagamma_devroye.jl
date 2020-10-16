const FOURPISQ = 4 * π^2
const HALFPISQ = 0.5 * π * π
const __TRUNC = 0.64
const __TRUNC_RECIP = 1.0 / __TRUNC

struct PolyaGammaDevRoye1Sampler <:
       Distributions.Sampleable{Distributions.Univariate,Distributions.Continuous}
    h::Float64
    z::Float64
end

struct PolyaGammaDevRoyeSampler <:
       Distributions.Sampleable{Distributions.Univariate,Distributions.Continuous}
    h::Float64
    z::Float64
end

## Calculate coefficient n in density of PG(1.0, 0.0), i.e. J* from Devroye.

function _a(n::Int64, x::Float64)
    K = (n + 0.5) * π
    y = 0
    if (x > __TRUNC)
        y = K * exp(-0.5 * K * K * x)
    elseif (x > 0)
        #expnt = -1.5 * (_log_c(0.5 * π)  + _log_c(x)) + _log_c(K) - 2.0 * (n+0.5)*(n+0.5) / x
        #y = _exp_c(expnt)
        y = (0.5 * π * x)^(-1.5) * K * _exp_c(-2.0 * (n + 0.5) * (n + 0.5) / x)
    end
    return y
end

function _mass_texpon(z::Float64)
    fz = pi^2 / 8 + z^2 / 2
    b = sqrt(1.0 / __TRUNC) * (__TRUNC * z - 1)
    a = sqrt(1.0 / __TRUNC) * (__TRUNC * z + 1) * -1.0

    x0 = _log_c(fz) + fz * __TRUNC
    xb = x0 - z + _log_c(Distributions.cdf(Distributions.Normal(), b))
    xa = x0 + z + _log_c(Distributions.cdf(Distributions.Normal(), a))

    qdivp = 4 / π * (_exp_c(xb) + _exp_c(xa))

    return 1.0 / (1.0 + qdivp)
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
    #E = tnorm(rng, 1 / sqrt(R))
    E = Distributions.rand(rng, Distributions.TruncatedNormal(0.0, 1.0, 1 / sqrt(R), Inf))
    X = scale / (E * E)
    return X
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
        while (Distributions.rand(rng) > alpha)
            X = rtinvchi2(rng, lambda, trunc)
            alpha = _exp_c(-0.5 * lambda / (mu * mu) * X)
        end
        # printf("rtigauss, part i: %g\n", X)
    else
        while (X > trunc)
            #X = igauss(rng, mu, lambda)
            X = Distributions.rand(rng, Distributions.InverseGaussian(mu, lambda))
        end
        # printf("rtigauss, part ii: %g\n", X)
    end
    return X::Float64
end
## Samples from PG(1, z)
function Distributions.rand(rng::Distributions.AbstractRNG, s::PolyaGammaDevRoye1Sampler)
    Z = abs(s.z) * 0.5
    ## PG(1,z) = 1/4 J*(1,Z/2)
    fz = (pi^2 + (4 * s.z^2)) / 8
    X = 0.0
    cont = true
    while cont
        #num.trials = num.trials + 1;
        if (Random.rand(rng) < _mass_texpon(Z))
            ## Truncated Exponential
            X = _TRUNC + Random.randexp(rng) / fz
        else
            ## Truncated Inverse Normal
            X = Distributions.rand(rng, TruncatedInverseGaussian(1 / Z, 1.0, 0.0, _TRUNC))
            #X = rtigauss_sp(rng, 1 / Z, 1.0, _TRUNC)
        end
        S = _a(0, X)
        Y = Random.rand(rng) * S
        n = 0
        while cont
            n += 1
            #total_iter += 1
            if (iseven(n))
                S -= _a(n, X)
                if (Y <= S)
                    cont = false
                end
            else
                S += _a(n, X)
                if (Y > S)
                    cont = false
                end
            end
        end
        if (Y <= S)
            cont = false
        end
    end
    return 0.25 * X
end

## Sample from PG(h, z) using Devroye-like method.
## h is a natural number and z is a positive real.
##--------------------------------------------------------------
function Distributions.rand(rng::Distributions.AbstractRNG, s::PolyaGammaDevRoyeSampler)
    devroye1sampler = PolyaGammaDevRoye1Sampler(s.h, s.z)
    return mapreduce(x -> Distributions.rand(rng, devroye1sampler), +, 1:s.h)
end
