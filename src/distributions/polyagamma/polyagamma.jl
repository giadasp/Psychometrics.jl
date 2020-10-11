include("invert_Y.jl")
include("truncated_norm.jl")
include("inverse_gaussian.jl")
include("polyagamma_sp.jl")
include("polyagamma_gamma_sum.jl")
include("polyagamma_normal.jl")


struct PolyaGamma{T1<:Real,T2<:Real} <: Distributions.ContinuousUnivariateDistribution
    h::T1
    z::T2
    PolyaGamma{T1,T2}(h, z) where {T1,T2} = new{T1,T2}(h, z)
end

function PolyaGamma(h::T1, z::T2; check_args = true) where {T1<:Real,T2<:Real}
    check_args && Distributions.@check_args(PolyaGamma, h > zero(h))
    return PolyaGamma{T1,T2}(h, z)
end

#### Outer constructors
PolyaGamma(h::T) where {T<:Real} = PolyaGamma(h, one(T))
PolyaGamma() = PolyaGamma(1, 1.0, check_args = false)

#@distr_support PolyaGamma -Inf Inf #TODO

#### Parameters

scale(d::PolyaGamma) = d.h
tilting(d::PolyaGamma) = d.z

params(d::PolyaGamma) = (d.h, d.z)
partype(::PolyaGamma{T1,T2}) where {T1,T2} = (T1, T2)

#### Statistics

mean(d::PolyaGamma) = d.h / (2 * d.z) * tanh(d.z / 2) # ord.h / (2 * d.z) * ((_exp_c(d.z) - 1) / (1 + _exp_c(d.z)))

var(d::PolyaGamma) = d.h / (4 * d.z^3) * (sinh(d.z) - d.z) * sech(d.z / 2)^2

#### pdf
## DEPRECATED IN BAYESLOGIT
## Calculate coefficient n in density of PG(1.0, 0.0), i.e. J* from Devroye.
##------------------------------------------------------------------------------
function _a_coef(n, x)
    if (x > _TRUNC)
        return pi * (n + 0.5) * _exp_c(-(n + 0.5)^2 * pi^2 * x / 2)
    else
        return (2 / pi / x)^1.5 * pi * (n + 0.5) * _exp_c(-2 * (n + 0.5)^2 / x)
    end
end

## DEPRECATED IN BAYESLOGIT
function _jacobi_logpdf(z::Float64, x::Float64; ntrunc::Int = _TERMS)
    v = mapreduce(n -> (iseven(n) ? 1 : -1) * _a_coef(n, x), +, 0:ntrunc)
    return _log_cosh_c(z) - x * z^2 / 2 + _log_c(v)
end

"""
    The log coefficients of the infinite sum for the density of PG(b, 0).
    See Polson et al. 2013, section 2.3.
"""
function _pg_logcoef(x, b, n)
    loggamma(n + b) - loggamma(n + 1) - loggamma(b) + _log_c(2n + b) -
    _log_c(2π * x^3) / 2 - (2n + b)^2 / 8x
end
"""
   log density of the PG(b, 0) distribution.
    See Polson et al. 2013, section 2.3.
"""
function _pg0_logpdf(x::Float64, b::Float64; ntrunc::Int = _TERMS)
    v = zero(x)
    for n = 0:ntrunc
        v += (iseven(n) ? 1 : -1) * _exp_c(_pg_logcoef(x, b, n))
    end
    return (b - 1) * _log_c(2) + _log_c(abs(v))
end
"""
    log density of the PG(b, c) distribution.
    See Polson et al. 2013, section 2.2 and equation (5).
"""
function _pg_logpdf(b::Float64, c::Float64, x::Float64; ntrunc::Int = _TERMS)
    b * _log_cosh_c(c / 2) - x * c^2 / 2 + _pg0_logpdf(x, b; ntrunc = ntrunc)
end

function logpdf(d::PolyaGamma, x::Float64; ntrunc::Int = _TERMS)
    if d.h == 1
        return _jacobi_logpdf(d.z / 2, 4 * x; ntrunc = ntrunc) + _log_c(4)
    else
        return _pg_logpdf(d.h, d.z, x; ntrunc = ntrunc)
    end
end

# function jacobi_pdf(z, x; ntrunc::Int)
#     v = zero(x)
#     for n in 0:ntrunc
#         v += (iseven(n) ? 1 : -1) * _a_coef(n, x)
#     end
#     return cosh(z) * _exp_c(-x*z^2/2) * v
# end

# """
#     The log coefficients of the infinite sum for the density of PG(b, 0).
#     See Polson et al. 2013, section 2.3.
# """
# function pg_coef(x, b, n)
#     if (isa(b,Integer))
#         g1 = prod((n + b + 1 - i) / (b + 1 - i) for i = 2 : b ) * (2 * n + b) / sqrt(2 * π * x^3)
#        #@show g1
#     else
#         g1 = gamma(n + b) / gamma(n + 1) * (2 * n + b) / sqrt(2 * π * x^3)
#     end
#     g2 = _exp_c(-(2 * n + b)^2 / 8 * x)
#     @show g1*g2
#     return g1 * g2
# end
# """
#    log density of the PG(b, 0) distribution.
#     See Polson et al. 2013, section 2.3.
# """
# function pg0_pdf(x, b; ntrunc::Int)
#     v = zero(x)
#     for n in 0:ntrunc
#         v += (iseven(n) ? 1 : -1) * (pg_coef(x, b, n))
#         #@show v
#     end
#     h = 2^(b-1) * v #/gamma(b)
#     #@show h
#     return h
# end
# """
#     log density of the PG(b, c) distribution.
#     See Polson et al. 2013, section 2.2 and equation (5).
# """
# function pg_pdf(b, c, x; ntrunc::Int)
#      pg0_pdf(x, b; ntrunc=ntrunc) *cosh(c/2)^b  * _exp_c(-x*c^2/2) 
# end

# function Distributions.pdf(d::PolyaGamma, x::Real; ntrunc::Int=_TERMS)
#    if d.h == 1
#         return jacobi_pdf(d.z/2, 4*x; ntrunc=ntrunc) * 4
#     else
#         return pg_pdf(d.h, d.z, x; ntrunc=ntrunc)
#     end
# end
Distributions.pdf(d::PolyaGamma, x::Real; ntrunc::Int = _TERMS) =
    _exp_c(logpdf(d, x; ntrunc = ntrunc))

#### Evaluation & Sampling



struct PolyaGammaDevRoyeSampler <:
       Distributions.Sampleable{Distributions.Univariate,Distributions.Continuous}
    h::Int64
    z::Float64
end

struct PolyaGammaDevRoye1Sampler <:
       Distributions.Sampleable{Distributions.Univariate,Distributions.Continuous}
    h::Float64
    z::Float64
end

# function sampler(d::PolyaGamma{T1,T2}) where {T1,T2}
#      if d.h == 1.0
#         return PolyaGammaDevRoye1Sampler(d.h,d.z)
#     elseif isa(d.h, Integer)
#         return PolyaGammaDevRoyeSampler(d.h,d.z)
#     else
#         return PolyaGammaInvGammaSumSampler(d.h,d.z)
#     end 
# end

function Distributions.rand(rng::Distributions.AbstractRNG, d::PolyaGamma)
    if d.h == 1.0
        return Distributions.rand(rng, PolyaGammaDevRoye1Sampler(float(d.h), d.z))
    elseif d.h == 2.0
        return Distributions.rand(rng, PolyaGammaDevRoyeSampler(d.h, d.z))
    elseif d.h <= 13.0
        return Distributions.rand(rng, PolyaGammaGammaSumSampler(float(d.h), d.z))
    elseif d.h <= 170.0
        return Distributions.rand(rng, PolyaGammaSPSampler(float(d.h), d.z, 100))
    else
        m = pg_m1(d.h, d.z)
        v = pg_m2(d.h, d.z) - m*m
        return Distributions.rand(rng,  Distributions.Normal(m, sqrt(v)))
    end
end

#Utils
function _mass_texpon(Z::Real)
    x = _TRUNC
    K = pi^2 / 8 + Z^2 / 2
    sqrtx = 1 / sqrt(x)
    b = sqrtx * (x * Z - 1)
    a = -1.0 * sqrtx * (x * Z + 1)
    x0 = _log_c(K) + K * _TRUNC
    xb = x0 - Z + Distributions.logcdf(Distributions.Normal(), b)
    xa = x0 + Z + Distributions.logcdf(Distributions.Normal(), a)
    return 1.0 / (1.0 + (4 / pi * (_exp_c(xb) + _exp_c(xa)))) # =p/(p+q)
end


## Samples from PG(1, z)
function Distributions.rand(rng::Distributions.AbstractRNG, s::PolyaGammaDevRoye1Sampler)
    Z = abs(s.z) * 0.5
    ## PG(1,z) = 1/4 J*(1,Z/2)
    fz = pi^2 / 8 + s.z^2 / 2
    ## p = (0.5 * pi) * _exp_c( -1.0 * fz * TRUNC) / fz;
    ## q = 2 * _exp_c(-1.0 * Z) * pigauss(TRUNC, 1.0 / Z, 1.0);
    #num_trials = 0;
    #total_iter = 0;
    X = 0.0
    while true
        #num.trials = num.trials + 1;
        if (Random.rand(rng) < _mass_texpon(Z))
            ## Truncated Exponential
            X = _TRUNC + Random.randexp(rng) / fz
        else
            ## Truncated Inverse Normal
            X = Distributions.rand(rng, TruncatedInverseGaussian(1 / Z, 1.0, 0.0, _TRUNC))
        end

        ## C = cosh(Z) * _exp_c( -0.5 * Z^2 * X )

        ## Don't need to multiply everything by C, since it cancels in inequality.
        S = _a_coef(0, X)
        Y = Random.rand(rng) * S
        n = 0
        while true
            n += 1
            #total_iter += 1
            if (iseven(n))
                S -= _a_coef(n, X)
                (Y <= S) && break
            else
                S += _a_coef(n, X)
                (Y > S) && break
            end
        end
        (Y <= S) && break
    end
    return 0.25 * X
end

## Sample from PG(h, z) using Devroye-like method.
## h is a natural number and z is a positive real.
##--------------------------------------------------------------
function Distributions.rand(rng::Distributions.AbstractRNG, s::PolyaGammaDevRoyeSampler)
    #total_trials = 0
    x = 0.0
    devroye1sampler = PolyaGammaDevRoye1Sampler(s.h, s.z)
    #for j = 1 : s.h
    #       x += Distributions.rand(rng, devroye1sampler)
    #       #total_trials += total_trials_PolyaGammaDevRoye1Sampler;
    #   end
    ## list("x"=x, "rate"=sum(n)/total.trials)
    return sum(Distributions.rand(rng, devroye1sampler, s.h))
end

