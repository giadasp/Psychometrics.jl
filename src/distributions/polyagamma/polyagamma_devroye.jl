# Adaptation of BayesLogit C code by Nicholas Polson, James Scott, Jesse Windle, 2012-2019

const FOURPISQ = 4 * __PI^2
const HALFPISQ = 0.5 * __PI * __PI
const __TRUNC = 0.64
const __TRUNC_RECIP = 1 / __TRUNC

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
    K = (n + 0.5) * __PI
    y = 0.0
    if (x > __TRUNC)
        y = K * _exp_c(-K * K * x / 2)
    elseif (x > 0)
        expnt = -1.5 * (_log_c(0.5 * __PI) + _log_c(x)) + _log_c(K) - 2.0 * (n + 0.5)^2 / x
        y = _exp_c(expnt)
        #y = 1 / (__PI * x / 2)^(1.5) * K * 1 /(_exp_c((n + 0.5)^2 / x)^2)
    end
    return y
end

function _mass_texpon(z::Float64)
    fz = __PI^2 / 8 + z^2 / 2
    b = sqrt(1.0 / __TRUNC) * (__TRUNC * z - 1)
    a = -sqrt(1.0 / __TRUNC) * (__TRUNC * z + 1)

    x0 = _log_c(fz) + fz * __TRUNC
    xb = x0 - z + _log_c(Distributions.cdf(Distributions.Normal(), b))
    xa = x0 + z + _log_c(Distributions.cdf(Distributions.Normal(), a))

    qdivp = 4 / __PI * (_exp_c(xb) + _exp_c(xa))

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
    #Z = abs(Z)
    t = __TRUNC
    X = t + 1.0
    if (__TRUNC_RECIP > Z)
        alpha = 0.0
        while (Random.rand(rng) > alpha)
            E1 = Random.randexp(rng)
            E2 = Random.randexp(rng)
            while (E1^2 > (2 * E2 / t))
                E1 = Random.randexp(rng)
                E2 = Random.randexp(rng)
            end
            X = t / (1 + E1 * t)^2
            alpha = _exp_c(-0.5 * Z^2 * X)
        end
    else
        mu = 1 / Z
        while (X > t)
            Y = (Random.randn(rng)^2)
            half_mu = mu / 2
            mu_Y = mu * Y
            X = mu + half_mu * mu_Y - half_mu * sqrt(4 * mu_Y + mu_Y^2)
            if (Random.rand(rng) > (mu / (mu + X)))
                X = mu^2 / X
            end
        end
    end
    return X
end

## Samples from PG(1, z)
function Distributions.rand(rng::Distributions.AbstractRNG, s::PolyaGammaDevRoye1Sampler)
    Z = abs(s.z) * 0.5
    ## PG(1,z) = 1/4 J*(1,Z/2)
    fz = (__PI^2 / 8) + (0.5 * Z^2)
    X = 0.0
    while true
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
        cont = true
        while cont
            n += 1
            if (mod(n, 2) == 1)
                S = S - _a(n, X)
                if (Y <= S)
                    return 0.25 * X
                end
            else
                S = S + _a(n, X)
                if (Y > S)
                    cont = false
                end
            end
        end
        # if (Y <= S)
        #     return 0.25 * X
        # end
    end
end

## Sample from PG(h, z) using Devroye-like method.
## h is a natural number and z is a positive real.
##--------------------------------------------------------------
function Distributions.rand(rng::Distributions.AbstractRNG, s::PolyaGammaDevRoyeSampler)
    devroye1sampler = PolyaGammaDevRoye1Sampler(s.h, s.z)
    return mapreduce(x -> Distributions.rand(rng, devroye1sampler), +, 1:s.h)
end
