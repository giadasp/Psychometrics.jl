using Pkg
Pkg.activate(".")
using Psychometrics
using Distributions
using LinearAlgebra
using Dates



# function posterior(f::function; prior = Normal(0,1), N = 1000)
#     return (1/N)*mapreduce(x -> f(rand(prior)),+,[1:N])
# end

## Speed tests

    I = 200
    N = 10_000

    items_cores = [Item2PL(string("item_",i), ["math"], Parameters2PL()) for i = 1 : I];
    examinees_cores = [Examinee1D(string("examinee_",n), Latent1D()) for n = 1 : N]; 


examinees_dict = Dict(n => examinees_cores[n] for n=1:N)
items_dict = Dict(n => items_cores[n] for n=1:I)

responses = generate_response(examinees_dict, items_dict);
responses_per_item = Dict( key_i => get_item_responses(key_i, responses)  for key_i in keys(items_dict))

# Total log-likelihood
l_sum = log_likelihood(responses, examinees_dict, items_dict);

# Log-likelihood per item
l_items = [log_likelihood(filter(r -> r.item_idx == key_i, responses), examinees_dict, Dict(key_i => i)) for (key_i, i) in items_dict];

# Total log-likelihood

L_sum = [ map( (x,w) -> mapreduce(r -> likelihood(r.val, x, i.parameters) , * , responses_per_item[key_i])*w , X, W) for (key_i, i) in items_dict]

# Likelihood per item
X = range(-6, stop=6, length=61)
W = fill(1/61,61)
L_per_item = [ map( (x,w) -> mapreduce(r -> likelihood(r.val, x, i.parameters) , * , responses_per_item[key_i])*w , X, W) for (key_i, i) in items_dict]
sum_l_per_item = map( l -> sum(l), l_per_item)

# with gradients
#g_item = zeros(2)
#g_latent = zeros(N)
#l_sum = log_likelihood(responses, examinees_dict, items_dict, g_item, g_latent);

expected_information_item_val = expected_information_item(examinees_dict, items_dict);
#only for 3PL
#observed_information_item_val = map( r -> observed_information_item(r), responses);
information_latent_val = information_latent(examinees_dict, items_dict);

## Estimation

items_cores_est = [Item2PL(string("item_",i), ["math"], Parameters2PL(1.0, [1e-5,5.0], 0.0, [-6.0, 6.0], Product([LogNormal(0, 1), Normal(0,1)]),  Product([LogNormal(0, 1), Normal(0,1)]), Vector{Vector{Float64}}(undef,2), LinearAlgebra.I(2) )) for i = 1 : I];
examinees_cores_est = [Examinee1D(string("examinee_",n), Latent1D(0.0, [-6.0, 6.0], Normal(0,1), Normal(0,1), zeros(0), 1.0)) for n = 1 : N]; 

map responses