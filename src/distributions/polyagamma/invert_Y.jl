# Adaptation of BayesLogit C code by Nicholas Polson, James Scott, Jesse Windle, 2012-2019

#------------------------------------------------------------------------------
# Constants
const _TRUNC = 0.64
const tol = 1e-8
const grid_size = 81
const ygrid = [
    0.0625,
    0.06698584,
    0.07179365,
    0.07694653,
    0.08246924,
    0.08838835,
    0.09473229,
    0.1015315,
    0.1088188,
    0.1166291,
    0.125,
    0.1339717,
    0.1435873,
    0.1538931,
    0.1649385,
    0.1767767,
    0.1894646,
    0.2030631,
    0.2176376,
    0.2332582,
    0.25,
    0.2679434,
    0.2871746,
    0.3077861,
    0.329877,
    0.3535534,
    0.3789291,
    0.4061262,
    0.4352753,
    0.4665165,
    0.5,
    0.5358867,
    0.5743492,
    0.6155722,
    0.659754,
    0.7071068,
    0.7578583,
    0.8122524,
    0.8705506,
    0.933033,
    1,
    1.071773,
    1.148698,
    1.231144,
    1.319508,
    1.414214,
    1.515717,
    1.624505,
    1.741101,
    1.866066,
    2,
    2.143547,
    2.297397,
    2.462289,
    2.639016,
    2.828427,
    3.031433,
    3.24901,
    3.482202,
    3.732132,
    4,
    4.287094,
    4.594793,
    4.924578,
    5.278032,
    5.656854,
    6.062866,
    6.498019,
    6.964405,
    7.464264,
    8,
    8.574188,
    9.189587,
    9.849155,
    10.55606,
    11.31371,
    12.12573,
    12.99604,
    13.92881,
    14.92853,
    16,
]

const vgrid = [
    -256,
    -222.8609,
    -194.0117,
    -168.897,
    -147.0334,
    -128,
    -111.4305,
    -97.00586,
    -84.4485,
    -73.51668,
    -63.99997,
    -55.71516,
    -48.50276,
    -42.22387,
    -36.75755,
    -31.99844,
    -27.85472,
    -24.24634,
    -21.10349,
    -18.36524,
    -15.97843,
    -13.89663,
    -12.07937,
    -10.49137,
    -9.101928,
    -7.884369,
    -6.815582,
    -5.875571,
    -5.047078,
    -4.315237,
    -3.667256,
    -3.092143,
    -2.580459,
    -2.124095,
    -1.716085,
    -1.350442,
    -1.022007,
    -0.7263359,
    -0.4595871,
    -0.2184366,
    0,
    0.1982309,
    0.3784427,
    0.5425468,
    0.6922181,
    0.828928,
    0.953973,
    1.068498,
    1.173516,
    1.269928,
    1.358533,
    1.440046,
    1.515105,
    1.584282,
    1.64809,
    1.706991,
    1.761401,
    1.811697,
    1.858218,
    1.901274,
    1.941143,
    1.978081,
    2.012318,
    2.044068,
    2.073521,
    2.100856,
    2.126234,
    2.149802,
    2.171696,
    2.192042,
    2.210954,
    2.228537,
    2.244889,
    2.260099,
    2.274249,
    2.287418,
    2.299673,
    2.311082,
    2.321703,
    2.331593,
    2.340804,
]

function y_eval(v::Float64)
    tol = 1e-6
    r = sqrt(abs(v))
    if (v > tol)
        return tan(r) / r
    elseif (v < -tol)
        return tanh(r) / r
    else
        #y = 1 + (1 / 3) * r^2 + (2 / 15) * r^4 + (17 / 315) * r^6
        return 1 + (1 / 3) * v + (2 / 15) * (v^2) + (17 / 315) * (v^3)
    end
end

function ydy_eval(v::Float64) #! put yp from second to first
    yp2 = y_eval(v)
    if (abs(v) >= tol)
        dyp2 = 0.5 * (yp2^2 + (1 - yp2) / v)
    else
        dyp2 = 0.5 * (yp2^2 - (1 / 3) - ((2 / 15) * v))
    end
    return yp2, dyp2
end

function f_eval(v::Float64, params::Float64)
    return y_eval(v) - params
end

function fdf_eval(v::Float64, params::Float64)
    fp2, dfp2 = ydy_eval(v)
    fp2 = fp2 - params
    return fp2, dfp2
end

function df_eval(v::Float64)
    f, df = ydy_eval(v)
    return df
end

function v_eval(y::Float64, tol::Float64, max_iter::Int64)
    ylower = ygrid[1]
    yupper = ygrid[grid_size]

    if (y < ylower)
        return -(1 / (y^2))
    elseif (y > yupper)
        v = atan(0.5 * y * __PI)
        return v^2
    elseif (y == 1)
        return 0.0
    end

    id = (_log_c(y) / _log_c(2.0) + 4.0) / 0.1

    idlow = Int64(floor(id) + 1)
    idhigh = Int64(min(idlow + 1, grid_size))
    vl = vgrid[idlow]  # lower bound
    vh = vgrid[idhigh] # upper bound

    iter = 0
    diff = tol + 1.0
    vnew = copy(vl)
    vold = copy(vl)

    while (diff > tol && iter <= max_iter)
        iter += 1
        vold = copy(vnew)
        f0, f1 = fdf_eval(vold, y)
        vnew = vold - f0 / f1
        vnew = vnew > vh ? vh : vnew
        vnew = vnew < vl ? vl : vnew
        diff = abs(vnew - vold)
    end

    if (iter > max_iter)
        println("v_eval: reached max_iter: ", iter)
    end

    return vnew
end
