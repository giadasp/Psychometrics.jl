# Adaptation of BayesLogit C code by Nicholas Polson, James Scott, Jesse Windle, 2012-2019 

# double rtinvchi2(double h, double trunc, RNG& r)
# 
#   double h2 = h * h
#   double R = trunc / h2
#   double X = 0.0
#   # I need to consider using a different truncated normal sampler.
#   double E1 = r.expon_rate(1.0) double E2 = r.expon_rate(1.0)
#   while ( (E1*E1) > (2 * E2 / R)) 
#     # printf("E %g %g %g %g\n", E1, E2, E1*E1, 2*E2/R)
#     E1 = r.expon_rate(1.0) E2 = r.expon_rate(1.0)
#   end
#   # printf("E %g %g \n", E1, E2)
#   X = 1 + E1 * R
#   X = R / (X * X)
#   X = h2 * X
#   return X
# end

const PISQ = __PI^2

const trunc_schedule = [
    0.64,
    0.68,
    0.72,
    0.75,
    0.78,
    0.8,
    0.83,
    0.85,
    0.87,
    0.89,
    0.91,
    0.93,
    0.95,
    0.96,
    0.98,
    1,
    1.01,
    1.03,
    1.04,
    1.06,
    1.07,
    1.09,
    1.1,
    1.12,
    1.13,
    1.15,
    1.16,
    1.17,
    1.19,
    1.2,
    1.21,
    1.23,
    1.24,
    1.25,
    1.26,
    1.28,
    1.29,
    1.3,
    1.32,
    1.33,
    1.34,
    1.35,
    1.36,
    1.38,
    1.39,
    1.4,
    1.41,
    1.42,
    1.44,
    1.45,
    1.46,
    1.47,
    1.48,
    1.5,
    1.51,
    1.52,
    1.53,
    1.54,
    1.55,
    1.57,
    1.58,
    1.59,
    1.6,
    1.61,
    1.62,
    1.63,
    1.65,
    1.66,
    1.67,
    1.68,
    1.69,
    1.7,
    1.71,
    1.72,
    1.74,
    1.75,
    1.76,
    1.77,
    1.78,
    1.79,
    1.8,
    1.81,
    1.82,
    1.84,
    1.85,
    1.86,
    1.87,
    1.88,
    1.89,
    1.9,
    1.91,
    1.92,
    1.93,
    1.95,
    1.96,
    1.97,
    1.98,
    1.99,
    2,
    2.01,
    2.02,
    2.03,
    2.04,
    2.05,
    2.07,
    2.08,
    2.09,
    2.1,
    2.11,
    2.12,
    2.13,
    2.14,
    2.15,
    2.16,
    2.17,
    2.18,
    2.19,
    2.21,
    2.22,
    2.23,
    2.24,
    2.25,
    2.26,
    2.27,
    2.28,
    2.29,
    2.3,
    2.31,
    2.32,
    2.33,
    2.35,
    2.36,
    2.37,
    2.38,
    2.39,
    2.4,
    2.41,
    2.42,
    2.43,
    2.44,
    2.45,
    2.46,
    2.47,
    2.48,
    2.49,
    2.51,
    2.52,
    2.53,
    2.54,
    2.55,
    2.56,
    2.57,
    2.58,
    2.59,
    2.6,
    2.61,
    2.62,
    2.63,
    2.64,
    2.65,
    2.66,
    2.68,
    2.69,
    2.7,
    2.71,
    2.72,
    2.73,
    2.74,
    2.75,
    2.76,
    2.77,
    2.78,
    2.79,
    2.8,
    2.81,
    2.82,
    2.83,
    2.84,
    2.85,
    2.87,
    2.88,
    2.89,
    2.9,
    2.91,
    2.92,
    2.93,
    2.94,
    2.95,
    2.96,
    2.97,
    2.98,
    2.99,
    3,
    3.01,
    3.02,
    3.03,
    3.04,
    3.06,
    3.07,
    3.08,
    3.09,
    3.1,
    3.11,
    3.12,
    3.13,
    3.14,
    3.15,
    3.16,
    3.17,
    3.18,
    3.19,
    3.2,
    3.21,
    3.22,
    3.23,
    3.24,
    3.25,
    3.27,
    3.28,
    3.29,
    3.3,
    3.31,
    3.32,
    3.33,
    3.34,
    3.35,
    3.36,
    3.37,
    3.38,
    3.39,
    3.4,
    3.41,
    3.42,
    3.43,
    3.44,
    3.45,
    3.46,
    3.47,
    3.49,
    3.5,
    3.51,
    3.52,
    3.53,
    3.54,
    3.55,
    3.56,
    3.57,
    3.58,
    3.59,
    3.6,
    3.61,
    3.62,
    3.63,
    3.64,
    3.65,
    3.66,
    3.67,
    3.68,
    3.69,
    3.71,
    3.72,
    3.73,
    3.74,
    3.75,
    3.76,
    3.77,
    3.78,
    3.79,
    3.8,
    3.81,
    3.82,
    3.83,
    3.84,
    3.85,
    3.86,
    3.87,
    3.88,
    3.89,
    3.9,
    3.91,
    3.92,
    3.93,
    3.95,
    3.96,
    3.97,
    3.98,
    3.99,
    4,
    4.01,
    4.02,
    4.03,
    4.04,
    4.05,
    4.06,
    4.07,
    4.08,
    4.09,
    4.1,
    4.11,
    4.12,
    4.13,
]


struct PolyaGammaAltSampler <:
       Distributions.Sampleable{Distributions.Univariate,Distributions.Continuous}
    h::Float64
    z::Float64
    maxinner::Int64
end

function a_coef(n::Int64, x::Float64, h::Float64)
    d_n = 2.0 * n + h
    log_out =
        h * _log_c(2.0) - loggamma(h) + loggamma(n + h) - loggamma(n + 1) + _log_c(d_n) -
        0.5 * _log_c(2.0 * __PI * (x^3)) - 0.5 * d_n * d_n / x
    out = _exp_c(log_out)
    return out::Float64
end

function a_coef_recursive(
    n::Float64,
    x::Float64,
    h::Float64,
    coef_h::Float64,
    gnh_over_gn1_gh::Float64,
)
    d_n = 2.0 * n + h
    # gamma_nh_over_n *= (n + h - 1) / n  # Can speed up further by separate function for a0 and an, n > 0.
    if (n != 0)
        gnh_over_gn1_gh *= (n + h - 1) / n
    else
        gnh_over_gn1_gh = 1.0
    end
    coef = coef_h * gnh_over_gn1_gh
    log_kernel = -0.5 * (_log_c(x * x * x) + d_n * d_n / x) + (d_n)
    return coef * _exp_c(log_kernel)::Float64
end

function w_left(x::Float64, h::Float64, z::Float64)
    if (z != 0)
        # p.13 l(x|h)
        return (1 + _exp_c(-2 * abs(z)))^h *
               Distributions.pdf(Distributions.InverseGaussian(h / z, h * h), x)
    else
        # p.11 l(x|h)
        return 2^h * Distributions.pdf(Distributions.InverseGamma(0.5, 0.5 * h * h), x)
    end
end

function w_right(x::Float64, h::Float64, z::Float64)
    if (z != 0)
        # p.13 r(x|h)
        lambda_z = (PISQ + 4 * (z^2)) / 8
        return (__PI / (2 * lambda_z))^h *
               Distributions.pdf(Distributions.Gamma(h, 1 / lambda_z), x)
    else
        # p.11 r(x|h)
        return (4 / __PI)^h * Distributions.pdf(Distributions.Gamma(h, 1 / (PISQ / 8)), x)
    end
    #lambda_z = __PI^2 * 0.125 + 0.5 * z * z
    #p = _exp_c(h * _log_c((__PI/2) / lambda_z)) * (1.0-p_gamma_rate(trunc, h, lambda_z, false))
end


function g_tilde(x::Float64, h::Float64, trunc::Float64)
    if (x > trunc)
        return _exp_c(
            h * _log_c(0.5 * __PI) + (h - 1) * _log_c(x) - PISQ * 0.125 * x - loggamma(h),
        )
    else
        return h * _exp_c(
            h * _log_c(2.0) - 0.5 * _log_c(2.0 * __PI * x * x * x) - 0.5 * h * h / x,
        )
    end
    # out = h * pow(2, h) * pow(2 * __PI * pow(x,3), -0.5) * _exp_c(-0.5 * pow(h,2) / x)
end

########################################
# Sample #
########################################

function draw_abridged(
    rng::Distributions.AbstractRNG,
    h::Float64,
    z::Float64,
    max_inner::Int64,
)
    # p.132 Windle(2013)
    if (h < 1 || h > 4)
        println("draw h = $h must be in [1,4]")
        return 0
    end

    # Change the parameter.
    z = abs(z) * 0.5

    trunc = trunc_schedule[Int64(floor((h - 1.0) * 100))]

    # Now sample 0.25 * J^*(1, z := z/2). 

    lambda_z = (PISQ + 4 * (z^2)) / 8
    weight_left = w_left(trunc, h, z)
    weight_right = w_right(trunc, h, z)
    prob_right = weight_right / (weight_right + weight_left)

    # printf("prob_right: %g\n", prob_right)

    coef1_h = _exp_c(h * _log_c(2.0) - 0.5 * _log_c(2.0 * __PI))
    # double gamma_nh_over_n = RNG::Gamma(h)
    gnh_over_gn1_gh = 1.0 # Will fill in value on first call to a_coef_recursive.

    num_trials = 0
    total_iter = 0

    while (num_trials < 10000)
        num_trials += 1

        X = 0.0
        Y = 0.0

        # if (Distributions.rand(rng) < p/(p+q))
        uu = Distributions.rand(rng) #runif(1)
        if (uu < prob_right)
            X = Distributions.rand(rng, TruncatedGamma(h, 1 / lambda_z, trunc, Inf))
        else
            X = Distributions.rand(rng, TruncatedInverseGaussian(h / z, h^2, 0.0, trunc))
        end
        # double S  = a_coef(0, X, h)
        S = a_coef_recursive(0.0, X, h, coef1_h, gnh_over_gn1_gh)
        a_n = copy(S)
        gt = g_tilde(X, h, trunc)
        Y = Distributions.rand(rng) * gt
        decreasing = false

        n = 0
        go = true
        # Cap the number of iterations?
        while (go && n <= max_inner)
            total_iter += 1
            n += 1
            # Break infinite loop.  Put first so it always checks n==0.
            prev = copy(a_n)
            # a_n  = a_coef(n, X, h)
            a_n = a_coef_recursive(Float64(n), X, h, coef1_h, gnh_over_gn1_gh)
            # printf("a_n=%g, a_n2=%g\n", a_n, a_n2)
            decreasing = a_n <= prev

            if ((n % 2) == 1)
                S = S - a_n
                if (Y <= S && decreasing)
                    return X
                end
            else
                S = S + a_n
            end
            if (Y > S && decreasing)
                go = false
            end
        end
        # Need Y <= S in event that Y = S, e.g. when X = 0.
    end
    # We should never get here.
    return -1.0
end # draw

function Distributions.rand(rng::Distributions.AbstractRNG, s::PolyaGammaAltSampler)
    if (s.h < 1)
        println("draw h = ", s.h, " must be >= 1")
        return 0
    end
    n = floor((s.h - 1.0) / 4.0)
    remain = s.h - 4.0 * n
    x = 0.0
    for i = 1:n
        x += draw_abridged(rng, 4.0, s.z, s.maxinner)
        if (remain > 4.0)
            x += (
                draw_abridged(rng, 0.5 * remain, s.z, s.maxinner) +
                draw_abridged(rng, 0.5 * remain, s.z, s.maxinner)
            )
        else
            x += draw_abridged(rng, remain, s.z, s.maxinner)
        end
    end
    return x
end

########################################
# APPENDIX #
########################################

# We should only have to calculate Gamma(h) once.  We can then get Gamma(n+h)
# from the recursion Gamma(z+1) = z Gamma(z).  Not sure how that is in terms of
# stability, but that should save us quite a few computations.  This affects
# a_coef and g_tilde.
