include("invert_Y.jl")
include("polyagamma_devroye.jl")
include("polyagamma_sp.jl")
include("polyagamma_alt.jl")
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

Distributions.pdf(d::PolyaGamma, x::Real; ntrunc::Int = _TERMS) =
    _exp_c(logpdf(d, x; ntrunc = ntrunc))


function Distributions.rand(rng::Distributions.AbstractRNG, d::PolyaGamma)
    if d.h == 1.0
        return Distributions.rand(rng, PolyaGammaDevRoye1Sampler(float(d.h), d.z))
    elseif d.h == 2.0
        return Distributions.rand(rng, PolyaGammaDevRoyeSampler(float(d.h), d.z))
    elseif d.h <= 13.0
        return Distributions.rand(rng, PolyaGammaAltSampler(float(d.h), d.z, 200))
    elseif d.h <= 170.0
        return Distributions.rand(rng, PolyaGammaSPSampler(float(d.h), d.z, 100))
    else
        m = pg_m1(d.h, d.z)
        v = pg_m2(d.h, d.z) - m * m
        return Distributions.rand(rng, Distributions.Normal(m, sqrt(v)))
    end
end
