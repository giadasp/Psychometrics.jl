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

items = [Item2PL(i, string("item_", i), ["math"], Parameters2PL(Product([LogNormal(0.2,0.3), Normal(0,1)]), [1e-5,Inf], [-Inf, Inf])) for i = 1:I];
examinees = [Examinee1D(e, string("examinee_", e), Latent1D()) for e = 1:N];

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
items_est = [Item2PL(i, string("item_",i), ["math"], Parameters2PL(Product([TruncatedNormal(1.0, 1.0, 0.0, Inf), Normal(0,1)]), [1e-5,5.0], [-6.0, 6.0])) for i = 1 : I];
examinees_est = [Examinee1D(e, string("examinee_",e), Latent1D(Normal(0,1), [-6.0, 6.0])) for e = 1 : N]; 

Iter = 2_000

responses_per_examinee = map( e -> sort(get_responses_by_examinee_id(e.id, responses), by = x -> x.item_idx), examinees);
items_per_examinee = map( e -> sort(items[get_items_idx_answered_by_examinee(examinees[e[1].examinee_idx], e)], by = x -> x.idx), responses_per_examinee);

responses_per_item = map( i -> sort(get_responses_by_item_id(i.id, responses), by= x-> x.examinee_idx), items);
examinees_per_item = map( i -> sort(examinees[get_examinees_idx_who_answered_item(items[i[1].item_idx], i)], by= x-> x.idx), responses_per_item);

map( i -> chain_append!(i), items_est)
map( e -> chain_append!(e), examinees_est)
map( i -> set_value_from_chain!(i), items_est)
map( e -> set_value_from_chain!(e), examinees_est)


for iter = 1:Iter
    if mod(iter,100)==0 
        println(iter)
    end
    W = generate_w(items, examinees_per_item)
    map( i -> mcmc_iter!(i, examinees_per_item[i.idx], responses_per_item[i.idx], map( y -> y.val, sort(filter(w -> w.i_idx == i.idx, W), by= x -> x.e_idx))), items_est)
    map( e -> mcmc_iter!(e, items_per_examinee[e.idx], responses_per_examinee[e.idx], map( y -> y.val, sort(filter(w -> w.e_idx == e.idx, W), by= x -> x.i_idx))), examinees_est)
end


mean_a = map(i -> mean(hcat(i.parameters.chain[Iter-1000:Iter]...)[1,:]), items_est);
mean_b= map(i -> mean(hcat(i.parameters.chain[Iter-1000:Iter]...)[2,:]), items_est);
mean_theta = map(e -> mean(e.latent.chain[Iter-1000:Iter]), examinees_est);
# RMSEs
println(sqrt(sum((map(i -> i.parameters.a, items) .- mean_a).^2)/I))
println(sqrt(sum((map(i -> i.parameters.b, items) .- mean_b).^2)/I))
println(sqrt(sum((map(e -> e.latent.val, examinees) .- mean_theta).^2)/N))

RMSE_a_100 = zeros(Iter-500)
RMSE_b_100 = zeros(Iter-500)
RMSE_t_100 = zeros(Iter-500)
for j = 1:Iter-100
    println(j)
    mean_a = [mean(hcat(i.parameters.chain[j:(j+499)]...)[1,:]) for i in items_est];
    mean_b = [mean(hcat(i.parameters.chain[j:(j+499)]...)[2,:]) for i in items_est];
    mean_theta = [mean(e.latent.chain[j:(j+499)]) for e in examinees_est];
 # RMSEs
RMSE_a_100[j]=sqrt(sum((map(i -> i.parameters.a, items) .- mean_a).^2)/I)
RMSE_b_100[j]=sqrt(sum((map(i -> i.parameters.b, items) .- mean_b).^2)/I)
RMSE_t_100[j]=sqrt(sum((map(e -> e.latent.val, examinees) .- mean_theta).^2)/N)
end
plot(hcat(RMSE_a_100, RMSE_b_100, RMSE_t_100))




mean_a = [mean(hcat(i.parameters.chain[1500:2000]...)[1,:]) for i in items_est]
mean_b = [mean(hcat(i.parameters.chain[1500:2000]...)[2,:]) for i in items_est]
mean_theta = [mean(e.latent.chain[1500:2000]) for e in examinees_est];

# hcat(map(i -> i.parameters.a, items), mean_a)
# hcat(map(i -> i.parameters.b, items), mean_b)


# RMSEs
sqrt(sum((map(i -> i.parameters.a, items) .- mean_a).^2)/I)
sqrt(sum((map(i -> i.parameters.b, items) .- mean_b).^2)/I)
sqrt(sum((map(e -> e.latent.val, examinees) .- mean_theta).^2)/N)