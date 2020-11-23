    @everywhere using Pkg
    @everywhere Pkg.instantiate()
    @everywhere Pkg.activate(".")
    @everywhere using Psychometrics
    @everywhere using Distributions
    @everywhere using LinearAlgebra
    @everywhere using Dates
    @everywhere using Random
    @everywhere using SharedArrays



import Base.copy

function copy(examinee::Examinee1D) 
    e = Examinee1D(examinee.idx, examinee.id, examinee.latent)
    return e::Examinee1D
end

I = 100
I_to_calibrate = 30
N = 20_000
test_length = 30
field_test_items = 3
true_items  = test_length - field_test_items
iter_mcmc_latent = 2_000
iter_mcmc_item = 4_000
#after how many responses update item parameter estimates
required_responses = 100
maximum_required_responses = 600

# TRUE VALUES

## ITEMS

a_dist = LogNormal(0.3,0.2);
a_bounds = [1e-5,Inf];
b_dist = Normal(0, 1);
b_bounds = [-6, 6];
items = [Item2PL(i, string("item_", i), ["math"], Parameters2PL(Product([a_dist, b_dist]), a_bounds, b_bounds)) for i = 1:I];

## EXAMINEES

latent_dist = Normal(0, 1)
latent_bounds = [-Inf, Inf]
examinees = Examinee1D[]
global n=1
for n in 1:N
    push!(examinees,  Examinee1D(n, string("examinee_", n), Latent1D(latent_dist, latent_bounds))) 
end

# RESPONSES

responses_per_examinee = [generate_response(e, items) for e in examinees];

# INITIAL VALUES

## ITEMS

a_est_prior = TruncatedNormal(1.0, 5, 0.0, Inf);
a_est_bounds = [1e-5, 5.0];
b_est_prior = Normal(0, 5);
b_est_bounds = [-6.0, 6.0];
# first I-I_to_calibrate items have true values of parameters (very well estimated)
items_est_calibrated = items[1:(I-I_to_calibrate)];
items_est_not_calibrated = [Item2PL(i+(I-I_to_calibrate), string("item_", i+(I-I_to_calibrate)), ["math"], Parameters2PL(Product([a_est_prior, b_est_prior]), a_est_bounds, b_est_bounds)) for i = 1 : I_to_calibrate];
map( i -> i.parameters.calibrated = false, items_est_not_calibrated);
# starting values
map( i -> i.parameters.a = 1.0, items_est_not_calibrated)
map( i -> i.parameters.b = 0.0, items_est_not_calibrated)
# starting expected information matrices
map( i -> i.parameters.expected_information = expected_information_item(i.parameters, Latent1D(0.0)), items_est_not_calibrated)
# append not calibrated items 
items_est = vcat(items_est_calibrated, items_est_not_calibrated);
#responses vectors for not calibrated items
responses_not_calibrated = Response[]

## EXAMINEES

latent_est_prior = Normal(0, 3);
latent_est_bounds = [-6.0, 6.0];
examinees_est = [Examinee1D(e, string("examinee_",e), Latent1D(latent_est_prior, latent_est_bounds)) for e = 1 : N]; 

#set starting value taking a random value in the interval -0.5:0.5
map(e -> begin e.latent.val = Random.rand(-0.3:0.3) end, examinees_est)
items_idx_per_examinee = Vector{Vector{Int64}}(undef, N);
examinees_est_theta = [[zero(Float64) for i=1:test_length] for n=1:N];

# START ONLINE-CALIBRATION

global n=1
while size(filter(i -> i.parameters.calibrated == false, items_est),1) > 0   
    examinees_n = copy(examinees_est[n])
    idx_n = examinees_n.idx
    items_idx_n = Int64[]
    responses_n = responses_per_examinee[n]
    println("n: ", n)
    #println("true: ", examinees[n].latent.val)
    available_items_idx = map(i -> i.idx, filter(i2 -> i2.parameters.calibrated, items_est))
    for i in 1:true_items
        next_item_idx = find_best_item(examinees_n, items_est[available_items_idx]);
        push!(items_idx_n, next_item_idx);
        available_items_idx = setdiff(available_items_idx, next_item_idx)
        sort!(items_idx_n)
        resp_n = responses_n[items_idx_n]
        items_est_n = items_est[items_idx_n]
        chain_n = SharedArray{Float64}(iter_mcmc_latent)
        # extract `iter_mcmc_latent` samples from the polyagamma and from theta conditional posterior
        @sync @distributed for iter in 1:iter_mcmc_latent
            W = generate_w(items_est[items_idx_n], examinees_n)
            chain_n[iter] = rand(posterior(examinees_n, items_est_n, resp_n, W))
        end
        # assign chain
        examinees_n.latent.chain = chain_n
        # update theta estimate
        update_estimate!(examinees_n)
        #println("est_BIAS: ", examinees_n.latent.val - examinees[n].latent.val)
        # store theta estimate
        examinees_est_theta[n][i] = examinees_n.latent.val
    end
    # find available items to calibrate
    available_items_idx = map(i -> i.idx, filter(i2 -> !i2.parameters.calibrated, items_est))
    for i in 1:field_test_items
        if size(available_items_idx,1) > 0
            # find best item to be calibrated
            next_item_idx = find_best_item(examinees_n, items_est[available_items_idx]; method = "D-gain");
            push!(items_idx_n, next_item_idx);
            available_items_idx = setdiff(available_items_idx, next_item_idx)
            sort!(items_idx_n)
            resp_n = responses_n[items_idx_n]
            items_est_n = items_est[items_idx_n]
            chain_n = SharedArray{Float64}(iter_mcmc_latent)
            # extract `iter_mcmc_latent` samples from the polyagamma and from theta conditional posterior
            @sync @distributed for iter in 1:iter_mcmc_latent
                W = generate_w(items_est[items_idx_n], examinees_n)
                chain_n[iter] = rand(posterior(examinees_n, items_est_n, resp_n, W))
            end
            # assign chain
            examinees_n.latent.chain = chain_n
            # update theta estimate
            update_estimate!(examinees_n)
            # set the theta prior as theta posterior
            examinees_n.latent.prior = examinees_n.latent.posterior
            examinees_est[n] = copy(examinees_n)
            #examinees_n.latent.chain=Float64[]
            # store theta estimate
            examinees_est_theta[n][i+(true_items)] = examinees_n.latent.val
            #println("est_BIAS: ", examinees_est[n].latent.val - examinees[n].latent.val)
            push!(responses_not_calibrated, responses_n[next_item_idx])
            #resp_item = sort(filter(r -> r.item_idx == next_item_idx, responses_not_calibrated), by = r -> r.examinee_idx)
            resp_item = filter(r -> r.item_idx == next_item_idx, responses_not_calibrated)
            if mod(size(resp_item, 1), required_responses) == 0
                println("item idx: ", next_item_idx)
                println("# responses: ", size(resp_item, 1))
                println("calibrate item ", next_item_idx)
                println("true pars ", items[next_item_idx].parameters.a," ", items[next_item_idx].parameters.b)
                examinees_est_item = examinees_est[map( r -> r.examinee_idx, resp_item)]
                item = items_est[next_item_idx]
                # chain = Vector{Vector{Float64}}(undef, iter_mcmc_item)
                # # extract `iter_mcmc_item` samples from the polyagamma and from item parameters conditional posteriors
                # for iter in 1:iter_mcmc_item
                #     W = generate_w(item, examinees_est_item)
                #     chain[iter] = rand(posterior(item, examinees_est_item, resp_item, W))
                # end
                # # assign chain
                # item.parameters.chain = copy(chain)
                # # or
                for iter = 1:iter_mcmc_item
                    W = generate_w(item, examinees_est_item)
                    mcmc_iter!(item, examinees_est_item, resp_item, W; sampling = true)
                end
                update_estimate!(item)
                #calibrate_item!(item, resp_item, examinees_est_item)
                println("est pars ", item.parameters.a," ",item.parameters.b)
                if size(resp_item, 1)>= maximum_required_responses
                    item.parameters.calibrated = true
                end
                items_est[next_item_idx] = item
            end
        end
    end
    println("est_BIAS: ", examinees_n.latent.val - examinees[n].latent.val)
    items_idx_per_examinee[n] = copy(items_idx_n)
    global n+=1
end

rmse_a =  sqrt(mean(map( (i, i_est)-> (i.parameters.a - i_est.parameters.a)^2, items[(I-I_to_calibrate):I], items_est[(I-I_to_calibrate):I]))) 
rmse_b = sqrt(mean(map( (i, i_est)-> (i.parameters.b - i_est.parameters.b)^2, items[(I-I_to_calibrate):I], items_est[(I-I_to_calibrate):I])))
rmse_theta = sqrt(mean(map( (e, e_est)-> (e.latent.val- e_est.latent.val)^2, examinees[1:(n-1)], examinees_est[1:(n-1)])))
println( "avg RMSE a = ", rmse_a)
println( "avg RMSE b = ", rmse_b)
println( "avg RMSE tehta = ", rmse_theta)

using JLD2
@save string("test/online-calibration/rep_", rep, "_results.jld2")

theta_sets = collect(-6.0:0.5:6.0)
rmse_theta = zeros(Float64,(size(theta_sets,1) ))
gap = theta_sets[2] - theta_sets[1]
for t in 1:size(theta_sets,1)
    println(t)
        e_t_idx = map( e -> e.idx, filter( e2-> (e2.latent.val >= theta_sets[t] - (gap/2)) && (e2.latent.val < theta_sets[t] +(gap/2)), examinees[1:n-1]))
        if size(e_t_idx,1)>1
            rmse_theta[t] = sqrt(mean(map((e,e_est) -> (e.latent.val-e_est.latent.val)^2, examinees[e_t_idx], examinees_est[e_t_idx])))
        else
            rmse_theta[t] = 0.0
        end
end
plot(theta_sets, rmse_theta)