function my_rescale(
    X::Vector{Float64},
    Wk::Vector{Float64},
    observed::Vector{Float64},
)
    X = X .- observed[1]
    X = X ./ observed[2]
    X = vcat(X[1] - (X[2] - X[1]), X)
    X = vcat(X, X[end] + (X[2] - X[1]))
    Wk = vcat(zero(Float64), Wk)
    Wk = vcat(Wk, zero(Float64))
    Wk = Wk ./ sum(Wk)
    return X, Wk
end

function cubic_spline_int(X::Vector{Float64}, NewX::Vector{Float64}, Wk::Vector{Float64})
    bw = abs(NewX[2] - NewX[1])
    K = size(X, 1)
    ok = 0
    mainArealiminf = 1
    mainArealimsup = K + 2
    NewX = NewX[mainArealiminf]:bw:(NewX[mainArealiminf]+(bw*(size(Wk, 1)-1)))
    interp_cubic = Interpolations.extrapolate(
        Interpolations.scale(
            Interpolations.interpolate(
                Wk,
                Interpolations.BSpline(Interpolations.Cubic(Interpolations.Line(Interpolations.OnGrid()))),
            ),
            NewX,
        ),
        Interpolations.Line(),
    )

    Wk_cubic = zeros(K)
    for k = 1:K
        Wk_cubic[k] = interp_cubic(X[k])
    end

    Wk = clamp.(Wk_cubic, 1e-20, one(Float64))
    Wk = Wk ./ sum(Wk)
    return Wk::Vector{Float64}
end