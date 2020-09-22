using Pkg
Pkg.activate(".")
using Psychometrics
using Distributions
using Dates



# function posterior(f::function; prior = Normal(0,1), N = 1000)
#     return (1/N)*mapreduce(x -> f(rand(prior)),+,[1:N])
# end

## Speed tests

I = 200
N = 10_000

items = [Item2PL(i, string("item_",i), ["math"], Parameters2PL()) for i = 1 : I];
examinees = [Examinee1D(n, string("examinee_",n), Latent1D()) for n = 1 : N]; 

responses = vcat(map( e -> map( i -> generate_response(e, i), items), examinees)...);

g_item = zeros(2)
g_latent = zeros(1)
l_sum = log_likelihood(responses, g_item, g_latent);
# @benchmark gradients_1 = [ probability_item_g(examinees[n].val, items[i]) for n = 1 : 10_000, i = 1 : 200]
# @benchmark gradients_2 = [ probability_item_g_2(examinees[n].val, items[i]) for n = 1 : 10_000, i = 1 : 200] #faster than 1

expected_information_item_val = map( r -> expected_information_item(r.examinee.latent, r.item.parameters), responses);
#only for 3PL
#observed_information_item_val = map( r -> observed_information_item(r), responses);
information_latent_val = map( r -> information_latent(r.examinee.latent, r.item.parameters), responses);

