
TruncatedInverseGaussian(μ, λ, l, u)
rand(PolyaGamma(1, 2))
using Plots
using StatsPlots

using Distributions
a = rand(LogNormal(0, 0.5), 500)
b = rand(Normal(0, 1), 500)
theta = rand(Normal(0, 1), 10_000)
x = abs.([a[i] * (theta[e] - b[i]) for e = 1:10_000, i = 1:500])
@benchmark [rand(PolyaGamma(1, x[e, i]), 100_000) for e = 1:1, i = 1:50]
