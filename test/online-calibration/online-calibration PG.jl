     @everywhere using Pkg
     #@everywhere Pkg.instantiate()
     #@everywhere Pkg.activate(".")

     @everywhere using Psychometrics
     @everywhere using Distributions
     @everywhere using LinearAlgebra
     @everywhere using Dates
     @everywhere using Random
     @everywhere using SharedArrays
     @everywhere using JLD2
     @everywhere using StatsBase
     @everywhere using DelimitedFiles
     @everywhere using FileIO



import Base.copy

function copy(examinee::Examinee1D) 
    e = Examinee1D(examinee.idx, examinee.id, examinee.latent)
    return e::Examinee1D
end



@sync @distributed for rep = 1:20

    
I_operational = 1000
I_field = 15
I_total = I_field + I_operational
N = 2_000

test_operational  = 30
test_field = 3
test_length = test_field + test_operational

iter_mcmc_latent = 2_000
iter_mcmc_item = 4_000
#after how many responses update item parameter estimates
batch_size = 5
N_T = 200

    # TRUE VALUES
    
    ## ITEMS

    ### operational items

    # log_a_dist = Normal(0.4, sqrt(0.1)); # mu= exp(0.4 + (0.5*0.1^2)), sigma^2= exp(0.4 + 0.1^2)*(exp(0.1^2) - 1)
    # a_bounds = [1e-5,Inf];
    # b_dist = Normal(0, 1);
    # b_bounds = [-6, 6];
    # items_operational = [Item2PL(i, string("item_", i), ["math"], Parameters2PL(Product([log_a_dist, b_dist]), a_bounds, b_bounds)) for i = 1:I_operational];
    # map(i ->
    #     begin 
    #         i.parameters.a = exp(i.parameters.a)
    #     end,
    # items_operational)

    # ### field items
    # quantiles_a = quantile(map(i -> i.parameters.a, items_operational), [0.25,0.5,0.75])
    # quantiles_b = quantile(map(i -> i.parameters.b, items_operational), [0.2, 0.35, 0.5, 0.65, 0.8] )
    # a_field = vcat([fill(q,5) for q in quantiles_a]...)
    # b_field = vcat([quantiles_b for q in quantiles_a]...)

    # items_field = [Item2PL(i, string("item_", i), ["math"], Parameters2PL(Product([log_a_dist, b_dist]), a_bounds, b_bounds)) for i = (I_operational + 1) : I_total];
    # map((i, a, b) ->
    #     begin
    #         i.parameters.a = a
    #         i.parameters.b = b
    #     end,
    #     items_field,
    #     a_field,
    #     b_field
    # )
    # items = vcat(items_operational, items_field)

    # @save "test/online-calibration/true values/true_items_1.jld2" items

    #@save "test/online-calibration/settings.jld2" I_total  I_field N test_length test_field test_operational iter_mcmc_latent iter_mcmc_item batch_size N_T
    items = load(string("test/online-calibration/true values/true_items_1.jld2"), "items")

    

    ## EXAMINEES
    
    # latent_dist = Normal(0, 1)
    # latent_bounds = [-Inf, Inf]
    # examinees = Examinee1D[]
    # global n=1
    # for n in 1:N
    #     push!(examinees,  Examinee1D(n, string("examinee_", n), Latent1D(latent_dist, latent_bounds))) 
    # end
    
    # @save "test/online-calibration/true_examinees.jld2" examinees
    examinees = load(string("test/online-calibration/true values/true_examinees_1.jld2"), "examinees")
retired_items = 0
retired_items_vector= fill(0, N)

# RESPONSES

responses_per_examinee = [ResponseBinary[] for e in examinees];

# INITIAL VALUES

## ITEMS

# first I_total-I_field items have true values of parameters (very well estimated)
items_est_operational = items[1:I_operational];
a_operational = map(i -> i.parameters.a, items_est_operational)
b_operational = map(i -> i.parameters.b, items_est_operational)
a_est_prior = TruncatedNormal(mean(a_operational), std(a_operational), 0,  Inf);
println(params(a_est_prior))
a_est_bounds = [1e-5, 5.0];
b_est_prior = Normal(mean(b_operational), std(b_operational));
b_est_bounds = [-6.0, 6.0];

items_est_field = [Item2PL(i+I_operational, string("item_", i+I_operational), ["math"], Parameters2PL(Product([a_est_prior, b_est_prior]), a_est_bounds, b_est_bounds)) for i = 1 : I_field];
map( i -> begin 
    i.parameters.calibrated = false
end, items_est_field);
# starting values
map( i -> begin
    i.parameters.a = mean(a_operational)
end, items_est_field);
map( i -> begin
    i.parameters.b = mean(b_operational)
end, items_est_field);
# starting expected information matrices
map( i -> begin
    i.parameters.expected_information = expected_information_item(i.parameters, Latent1D(0.0))
end, items_est_field)
# append not calibrated items 
items_est = vcat(items_est_operational, items_est_field);
#responses vectors for not calibrated items
responses_not_calibrated = ResponseBinary[]

## EXAMINEES

latent_est_prior = Normal(0, 3);
latent_est_bounds = [-6.0, 6.0];
examinees_est = [Examinee1D(e, string("examinee_",e), Latent1D(latent_est_prior, latent_est_bounds)) for e = 1 : N]; 

#set starting value taking a random value in the interval -0.5:0.5
map(e ->
        begin
            e.latent.val = Random.rand(-0.3:0.3)
        end,
examinees_est)
items_idx_per_examinee = Vector{Vector{Int64}}(undef, N);
examinees_est_theta = [[zero(Float64) for i=1:test_length] for n=1:N];
items_est_parameters = [[[0.0, 0.0] for n=1:ceil(N_T/batch_size)] for i=1:I_field]
# @save string("test/online-calibration/priors",myid(),".jld2") a_est_prior b_est_prior latent_est_prior 
# @save string("test/online-calibration/bounds_",myid(),".jld2") a_est_bounds b_est_bounds latent_est_bounds 

# START ONLINE-CALIBRATION

global n=1
while retired_items < I_field #size(filter(i -> i.parameters.calibrated == false, items_est),1) > 0   
    examinee = copy(examinees_est[n])
    examinee_true = copy(examinees[n])
    idx_n = examinee.idx
    items_idx_n = Int64[]
    responses_n = ResponseBinary[]
    items_est_n = AbstractItem[]
    println("n: ", n)
    #println("true: ", examinees[n].latent.val)
    available_items_idx = map(i -> i.idx, filter(i2 -> i2.parameters.calibrated, items_est))
    for i in 1:test_operational
        # find best item idx
        next_item_idx = find_best_item(examinee, items_est[available_items_idx]);
        push!(items_idx_n, next_item_idx);
        available_items_idx = setdiff(available_items_idx, next_item_idx)
        sort!(items_idx_n)
        items_est_n = items_est[items_idx_n]

        # answer to the item
        resp = answer(examinee_true, items[next_item_idx])
        push!(responses_n, resp)
        # sort responses by item_idx
        sort!(responses_n, by = r -> r.item_idx)

        estimate_ability!(examinee, items_est_n, responses_n; mcmc_iter = iter_mcmc_latent, sampling = false)
        #println("est_BIAS: ", examinee.latent.val - examinees[n].latent.val)
        # store theta estimate
        examinees_est_theta[n][i] = examinee.latent.val
    end
    # find available items to calibrate
    available_items_idx = map(i -> i.idx, filter(i2 -> !i2.parameters.calibrated, items_est))
    for i in 1:test_field
        if size(available_items_idx,1) > 0
            # find best item to be calibrated
            next_item_idx = find_best_item(examinee, items_est[available_items_idx]; method = "D-gain");
            push!(items_idx_n, next_item_idx);
            available_items_idx = setdiff(available_items_idx, next_item_idx)
            sort!(items_idx_n)
            items_est_n = items_est[items_idx_n]

            # answer to the item
            resp = answer(examinee_true, items[next_item_idx])
            push!(responses_n, resp)
            # sort responses by item_idx
            sort!(responses_n, by = r -> r.item_idx)
            estimate_ability!(examinee, items_est_n, responses_n; mcmc_iter = iter_mcmc_latent, sampling = false)

            # set the theta prior as theta posterior
            examinee.latent.prior = examinee.latent.posterior
            examinees_est[n] = copy(examinee)
            # store theta estimate
            #println("examinee.latent.val: ", examinee.latent.val)
            examinees_est_theta[n][i+(test_operational)] = examinee.latent.val
            #println("est_BIAS: ", examinees_est[n].latent.val - examinees[n].latent.val)
            push!(responses_not_calibrated, resp)
            item = copy(items_est[next_item_idx])
            #item.parameters.expected_information += expected_information_item(item.parameters, examinee.latent)
            resp_item = get_responses_by_item_idx(item.idx, responses_not_calibrated)
           if mod(size(resp_item, 1), batch_size) == 0
                #println("item idx: ", next_item_idx)
                #println("# responses: ", size(resp_item, 1))
                #println("calibrate item ", next_item_idx)
                #println("true pars ", items[next_item_idx].parameters.a," ", items[next_item_idx].parameters.b)
                # item = copy(items_est[next_item_idx])
                # empty_chain!(item)
                # for iter = 1:iter_mcmc_item
                #     W = generate_w(item, examinees_est_item)
                #     mcmc_iter!(item, examinees_est_item, resp_item, W; sampling = false)
                # end
                # update_estimate!(item; sampling = false)
                # if next_item_idx==1001
                # println("est_BIAS a 1: ", sqrt((item.parameters.a - items[next_item_idx].parameters.a)^2))
                # println("est_BIAS b 1: ", sqrt((item.parameters.b - items[next_item_idx].parameters.b)^2))
                # end
                examinees_est_item = examinees_est[map( r -> r.examinee_idx, resp_item)]
                calibrate_item!(item, examinees_est_item, resp_item; mcmc_iter = iter_mcmc_item, sampling = false)
                #save chain
                writedlm(string("test/online-calibration/rep_", rep, "_item_", item.idx, "_resp_", size(resp_item, 1), "_chain.csv"), item.parameters.chain)
                items_est_parameters[next_item_idx - I_operational][Int64(size(resp_item,1)/batch_size)] = [item.parameters.a, item.parameters.b]
                if next_item_idx==1001
                println("est_BIAS a 2: ", sqrt((item.parameters.a - items[next_item_idx].parameters.a)^2))
                println("est_BIAS b 2: ", sqrt((item.parameters.b - items[next_item_idx].parameters.b)^2))
                end
                if size(resp_item, 1) >= N_T
                    item.parameters.calibrated = true
                    retired_items +=1
                end
                #update expected information
                item.parameters.expected_information = mapreduce(e -> expected_information_item(item.parameters, e.latent), +, examinees_est_item)
            else
                #update expected information
                item.parameters.expected_information += expected_information_item(item.parameters, examinee.latent)
            end
            items_est[next_item_idx] = item
        end
    end
    examinees_est[n] = copy(examinee)
    #println("est_BIAS: ", examinee.latent.val - examinees[n].latent.val)
    responses_per_examinee[n] = copy(responses_n)
    items_idx_per_examinee[n] = copy(items_idx_n)
    retired_items_vector[n] = copy(retired_items)
    global n+=1
end
global n-=1
@save string("test/online-calibration/how_many_examinees_rep_",rep,".jld2") n
@save string("test/online-calibration/retired_items_vector_rep_",rep,".jld2") retired_items_vector
@save string("test/online-calibration/examinees_est_theta_rep_",rep,".jld2") examinees_est_theta
@save string("test/online-calibration/items_est_parameters_rep_",rep,".jld2") items_est_parameters

@save string("test/online-calibration/items_est_rep_",rep,".jld2") items_est
@save string("test/online-calibration/examinees_est_rep_",rep,".jld2") examinees_est

rmse_a =  sqrt(mean(map( (i, i_est)-> (i.parameters.a - i_est.parameters.a)^2, items[I_operational:I_total], items_est[I_operational:I_total]))) 
rmse_b = sqrt(mean(map( (i, i_est)-> (i.parameters.b - i_est.parameters.b)^2, items[I_operational:I_total], items_est[I_operational:I_total])))
rmse_theta = sqrt(mean(map( (e, e_est)-> (e.latent.val- e_est.latent.val)^2, examinees[1:(n-1)], examinees_est[1:(n-1)])))
println( "avg RMSE a = ", rmse_a)
println( "avg RMSE b = ", rmse_b)
println( "avg RMSE tehta = ", rmse_theta)

@save string("test/online-calibration/rep_", rep, "_items_est.jld2") items_est
@save string("test/online-calibration/rep_", rep, "_responses.jld2") responses_per_examinee

end
# theta_sets = collect(-6.0:0.5:6.0)
# rmse_theta = zeros(Float64,(size(theta_sets,1) ))
# gap = theta_sets[2] - theta_sets[1]
# for t in 1:size(theta_sets,1)
#     println(t)
#         e_t_idx = map( e -> e.idx, filter( e2-> (e2.latent.val >= theta_sets[t] - (gap/2)) && (e2.latent.val < theta_sets[t] +(gap/2)), examinees[1:n-1]))
#         if size(e_t_idx,1)>1
#             rmse_theta[t] = sqrt(mean(map((e,e_est) -> (e.latent.val-e_est.latent.val)^2, examinees[e_t_idx], examinees_est[e_t_idx])))
#         else
#             rmse_theta[t] = 0.0
#         end
# end
# plot(theta_sets, rmse_theta)


a_s = zeros(1020,rep)
b_s = zeros(1020,rep)
for rep = 1:rep
    @load string("test/online-calibration/rep_", rep, "_items_est.jld2") items_est
    a = map(i -> i.parameters.a, items_est)
    a_s[:, rep] = copy(a)
    b = map(i -> i.parameters.b, items_est)
    b_s[:, rep] = copy(b)
end

a_true = map(i -> i.parameters.a, items)
a_true = hcat([a_true for rep=1:rep]...)
b_true = map(i -> i.parameters.b, items)
b_true = hcat([b_true for rep=1:rep]...)
diff_a = (a_s[1001:1020,:] - a_true[1001:1020,:]).^2
diff_b = (b_s[1001:1020,:] - b_true[1001:1020,:]).^2

println(sqrt(sum(diff_a)/20/rep))
println(sqrt(sum(diff_b)/20/rep))