function rescale!(
    dist::Distributions.DiscreteNonParametric,
    examinees::Vector{<:AbstractExaminee};
    metric::Vector{Float64} = [0.0, 1.0]
)
#check observed metric
    N = size(examinees, 1)
    Wk = mapreduce(e -> e.latent.posterior.p, +, examinees) ./ N
    Xk = dist.support
    observed =
        [LinearAlgebra.dot(Wk, Xk), sqrt(LinearAlgebra.dot(Wk, Xk .^ 2))]
    observed = [
        observed[1] - metric[1],
        observed[2] / metric[2],
    ]
    Xk2, Wk2 = my_rescale(Xk, Wk, observed)
    Wk = cubic_spline_int(Xk, Xk2, Wk2)
    dist = Distributions.DiscreteNonParametric(Xk2[2:(end-1)], Wk; check_args=false)
    return nothing
end
