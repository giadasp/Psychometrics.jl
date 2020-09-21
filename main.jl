using Distributions
using Dates
include("utils.jl")
include("dist.jl")
include("latent.jl")
include("parameters.jl")
include("item.jl")
include("examinee.jl")
include("design.jl")
include("response.jl")
include("latent.jl")
include("probability.jl")
include("likelihood.jl")
include("information.jl")



# function posterior(f::function; prior = Normal(0,1), N = 1000)
#     return (1/N)*mapreduce(x -> f(rand(prior)),+,[1:N])
# end

## Speed tests

using BenchmarkTools
I = 200
N = 20_000

items = [Item2PL(i, string("item_",i), ["math"], Parameters2PL()) for i = 1 : I];
examinees = [Examinee1D(n, string("examinee_",n), Latent1D()) for n = 1 : N]; 

responses = map( (e, i) -> generate_response(e, i), examinees, items)

log_likelihood_item_g = zeros(2)
log_likelihood_latent_g = zeros(1)
l_sum = log_likelihood(responses, log_likelihood_item_g, log_likelihood_latent_g)
# @benchmark gradients_1 = [ probability_item_g(examinees[n].val, items[i]) for n = 1 : 10_000, i = 1 : 200]
# @benchmark gradients_2 = [ probability_item_g_2(examinees[n].val, items[i]) for n = 1 : 10_000, i = 1 : 200] #faster than 1

information = [expected_information_item(examinees[n].val, items[i]) for n = 1 : 20_000, i = 1 : 200];


@show information



using StatsBase
StatsBase.cov(hcat([rand(Normal(0,1),2) for n=1:1000]...), dims=2)

#StatsBase.cov(rand(MvNormal([0.0,0.0],[1.0 0.0 ; 0.0 1.0]),10000),dims=2)