using Pkg
Pkg.activate(".")
using Psychometrics
using Distributions
using LinearAlgebra
using Dates

const I = 100
const N = 1_000


# ITEM PARAMETERS AND LATENTS 

items = [Item2PL(i, string("item_",i), ["math"], Parameters2PL()) for i = 1 : I];
examinees = [Examinee1D(e, string("examinee_",e), Latent1D()) for e = 1 : N];   

# Put parameters and latents in matrix  
parameters_matrix = get_parameters(items)
latents_matrix = get_latents(examinees)                                                                                                                                                      


# RESPONSES

responses = reduce(vcat, [generate_response([e], items[sort!(sample(collect(1:I), 40, replace = false))]) for e in examinees]);
responses_per_item = map(i -> get_responses_by_item_id(i.id, responses), items);  # slow

# Get design matrix
design_matrix = get_design(responses, I, N)

# Get response matrix (zeros are for wrong answers and for missing answers)
response_matrix = get_response_matrix(responses, I, N)

# Go back to a vector of `Response`s (starting and ending time of response is Dates.now())
responses2 = get_responses(response_matrix, design_matrix, items, examinees)


# ITEM CHARACTERISTIC FUNCTION (ICF)

## compute probabilities (ICF) by matrix (1PL or 2PL, latent 1D or latend ND) using aθ - b parametrization
p = probability(parameters_matrix, latents_matrix)

## compute probabilities (ICF) on vectors of examinees and items using a(θ - b) parametrization
p_2 = probability(items, examinees)


# ITEM INFORMATION FUNCTION (IIF)

## compute informations (IIF) by matrix (1PL or 2PL, latent 1D) using aθ - b parametrization
information = information_latent(latents_matrix, parameters_matrix)

## compute informations (IIF) on vectors of examinees and items using a(θ - b) parametrization
information_2 = information_latent(examinees, items)


# LIKELIHOOD AND LOG-LIKELIHOOD

# Total Likelihood (only for a(θ - b) parametrization)
L_tot = prod(likelihood(responses, examinees, items));

# Total log-likelihood (only for a(θ - b) parametrization)
l_tot = sum(log_likelihood(responses, examinees, items));
l_tot_per_item = [sum(log_likelihood(responses_per_item[i], examinees, items)) for i in 1:I];

# with gradients
#g_item = zeros(2)
#g_latent = zeros(N)
#l_sum = log_likelihood(responses, examinees, items, g_item, g_latent);

expected_information_item_val = expected_information_item(examinees, items);
# only for 3PL the observed information
#observed_information_item_val = map( r -> observed_information_item(r), responses);

## Estimation

items_est = [Item2PL(string("item_",i), ["math"], Parameters2PL(1.0, [1e-5,5.0], 0.0, [-6.0, 6.0], Product([LogNormal(0, 0.5), Normal(0,1)]),  Product([LogNormal(0, 0.5), Normal(0,1)]), Vector{Vector{Float64}}(undef,2), LinearAlgebra.I(2) )) for i = 1 : I];
examinees_est = [Examinee1D(string("examinee_",e), Latent1D(0.0, [-6.0, 6.0], Normal(0,1), Normal(0,1), zeros(0), 1.0)) for e = 1 : N]; 

map responses