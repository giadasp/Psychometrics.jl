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
        expnt = -1.5 * (_log_c(0.5 * π)  + _log_c(x)) + _log_c(K) - 2.0 * (n+0.5)*(n+0.5) / x
        y = _exp_c(expnt)
        #y = (0.5 * π * x)^(-1.5) * K * _exp_c(-2.0 * (n + 0.5) * (n + 0.5) / x)
    end
    return y
end

function _mass_texpon(z::Float64)
    fz = pi^2 / 8 + z^2 / 2
    b = sqrt(1.0 / __TRUNC) * (__TRUNC * z - 1)
    a = -sqrt(1.0 / __TRUNC) * (__TRUNC * z + 1) 

    x0 = _log_c(fz) + fz * __TRUNC
    xb = (x0 / z * Distributions.cdf(Distributions.Normal(), b))
    xa = (x0 * z * Distributions.cdf(Distributions.Normal(), a))

    qdivp = 4 / π * (xb + xa)

    return 1 / (1 + qdivp)
end

# function rtinvchi2_dr(rng::Distributions.AbstractRNG, scale::Float64, trunc::Float64)
#     R = trunc / scale
#     E = tnorm_left_sp(rng, 1 / sqrt(R))
#     #E = Distributions.rand(rng, Distributions.TruncatedNormal(0.0, 1.0, 1 / sqrt(R), Inf))
#     X = scale / (E * E)
#     return X
# end

function rtigauss_dr(rng::Distributions.AbstractRNG, Z::Float64)
    Z = abs(Z)
    t = __TRUNC
    X = t + 1.0
    if (__TRUNC_RECIP > Z)
        alpha = 0.0
        while (Random.rand(rng) > alpha)
            E1 = Random.randexp(rng)
            E2 = Random.randexp(rng)
            while (E1 * E1 > 2 * E2 / t)
                E1 = Random.randexp(rng)
                E2 = Random.randexp(rng)
            end
            X = 1 + E1 * t
            X = t / (X * X)
            alpha = _exp_c(-0.5 * Z * Z * X)
        end
    else
        mu = 1.0 / Z
        while (X > t)
            Y = Random.randn(rng)
            Y *= Y
            half_mu = 0.5 * mu
            mu_Y = mu * Y
            X = mu + half_mu * mu_Y - half_mu * sqrt(4 * mu_Y + mu_Y * mu_Y)
            if (Random.rand(rng) > mu / (mu + X))
                X = mu * mu / X
            end
        end
    end
    return X
end

## Samples from PG(1, z)
function Distributions.rand(rng::Distributions.AbstractRNG, s::PolyaGammaDevRoye1Sampler)
    Z = abs(s.z) * 0.5
    ## PG(1,z) = 1/4 J*(1,Z/2)
    fz = (pi^2 + (4 * s.z^2)) / 8
    X = 0.0
    cont = true
    while cont
        #num.trials = num.trials + 1
        if (Random.rand(rng) < _mass_texpon(Z))
            ## Truncated Exponential
            X = _TRUNC + Random.randexp(rng) / fz
        else
            ## Truncated Inverse Normal
            #X = Distributions.rand(rng, TruncatedInverseGaussian(1 / Z, 1.0, 0.0, _TRUNC))
            X = rtigauss_dr(rng, Z)
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
