using Pkg
Pkg.activate(".")
using Psychometrics
using Distributions
using LinearAlgebra
using Dates
using Random

const I = 20
const N = 500

# ITEM PARAMETERS AND LATENTS 

items = [
    Item(
        i,
        string("item_", i),
        ["math"],
        Parameters2PL(
            Product([LogNormal(0.0, 0.25), Normal(0, 1)]),
            [1e-5, Inf],
            [-Inf, Inf],
        ),
    ) for i = 1:I
];
examinees = [Examinee(e, string("examinee_", e), Latent1D()) for e = 1:N];

# RESPONSES

responses = generate_response(examinees, items);
# set Seeds for Random Generation
##################################################################################################
###########################      PolyGamma MCMC sampler   ########################################
##################################################################################################


##################################################################################################
###########################     Estimation Starts from here#######################################
##################################################################################################

#Initial Values, need to make sure that all Variance needs to be positive
items_est = [
    Item(
        i,
        string("item_", i),
        ["math"],
        Parameters2PL(
            Product([TruncatedNormal(1, 5, 0.0, Inf), Normal(0, 5)]),
            [1e-5, 5.0],
            [-6.0, 6.0],
        ),
    ) for i = 1:I
];
map(i -> i.parameters.calibrated = false, items_est);
map(i -> i.parameters.a = 1.0, items_est);
map(i -> i.parameters.b = 0.0, items_est);


examinees_est =
    [Examinee(e, string("examinee_", e), Latent1D(Normal(0, 1), [-6.0, 6.0])) for e = 1:N];
for n = 1:N
    # examinees_est[n].latent.val = examinees[n].latent.val + rand(Normal(0.0,0.3))
    examinees_est[n].latent.val = 0.0
    #examinees_est[n].latent.prior = Normal(examinees[n].latent.val, 0.3)
end

Iter = 4_000

responses_per_examinee = map(
    e -> sort(get_responses_by_examinee_id(e.id, responses), by = x -> x.item_idx),
    examinees,
);
items_idx_per_examinee = map(
    e -> sort(map(r -> items_est[r.item_idx].idx, responses_per_examinee[e.idx])),
    examinees_est,
);

responses_per_item = map(
    i -> sort(get_responses_by_item_id(i.id, responses), by = x -> x.examinee_idx),
    items,
);
examinees_idx_per_item = map(
    i -> sort(map(r -> examinees_est[r.examinee_idx].idx, responses_per_item[i.idx])),
    items_est,
);

# map( i -> chain_append!(i), items_est)
# map( e -> chain_append!(e), examinees_est)
# map( i -> set_val_from_chain!(i), items_est)
# map( e -> set_value_from_chain!(e), examinees_est)
map(
    i -> begin
        i.parameters.chain = [[i.parameters.a, i.parameters.b] for j = 1:1000]
    end,
    items_est,
);
#map( i -> calibrate_item!(i,  responses_per_item[i.idx], examinees_est[examinees_idx_per_item[i.idx]]), items_est)

for iter = 1:Iter
    if mod(iter, 100) == 0
        println(iter)
    end

    W = generate_w(
        items_est,
        map(i -> examinees_est[examinees_idx_per_item[i.idx]], items_est),
    )
    #    map( i -> begin
    #      i.parameters.chain[sample(1:1000)] = rand(posterior(i, examinees_est[examinees_idx_per_item[i.idx]],  responses_per_item[i.idx], map(w -> w.val, W)))
    #    end, items_est)
    map(
        i -> mcmc_iter!(
            i,
            examinees_est[examinees_idx_per_item[i.idx]],
            responses_per_item[i.idx],
            map(y -> y.val, sort(filter(w -> w.i_idx == i.idx, W), by = x -> x.e_idx));
            sampling = false,
        ),
        items_est,
    )
    map(
        e -> mcmc_iter!(
            e,
            items_est[items_idx_per_examinee[e.idx]],
            responses_per_examinee[e.idx],
            map(y -> y.val, sort(filter(w -> w.e_idx == e.idx, W), by = x -> x.i_idx));
            sampling = false,
        ),
        examinees_est,
    )

end

map(i -> update_estimate!(i), items_est);
map(e -> update_estimate!(e), examinees_est);


mean_a = map(i -> i.parameters.a, items_est);
mean_b = map(i -> i.parameters.b, items_est);
mean_theta = map(e -> e.latent.val, examinees_est);

# RMSEs
println(sqrt(sum((map(i -> i.parameters.a, items) .- mean_a) .^ 2) / I))
println(sqrt(sum((map(i -> i.parameters.b, items) .- mean_b) .^ 2) / I))
println(sqrt(sum((map(e -> e.latent.val, examinees) .- mean_theta) .^ 2) / N))
