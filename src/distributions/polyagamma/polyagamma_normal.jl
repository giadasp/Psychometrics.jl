
function jj_m1(b::Float64, z::Float64)
    z = abs(z)
    m1 = 0.0
    if (z > 1e-12)
        m1 = b * tanh(z) / z
    else
        m1 = b * (1 - (1.0 / 3) * z^2 + (2.0 / 15) * z^4 - (17.0 / 315) * z^6)
    end
    return m1::Float64
end

function jj_m2(b::Float64, z::Float64)
    z = abs(z)
    m2 = 0.0
    if (z > 1e-12)
        m2 = (b + 1) * b * (tanh(z) / z)^2 + b * ((tanh(z) - z) / (z^3))
    else
        m2 =
            (b + 1) * b * (1 - (1.0 / 3) * z^2 + (2.0 / 15) * z^4 - (17.0 / 315) * z^6)^2 +
            b * ((-1.0 / 3) + (2.0 / 15) * z^2 - (17.0 / 315) * z^4)
    end
    return m2
end

function pg_m1(b::Float64, z::Float64)
    return jj_m1(b, 0.5 * z) * 0.25
end

function pg_m2(b::Float64, z::Float64)
    return jj_m2(b, 0.5 * z) * 0.0625
end
