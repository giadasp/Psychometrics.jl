@everywhere using Pkg
@everywhere Pkg.instantiate()
@everywhere Pkg.activate(".")
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

function copy(examinee::Examinee) 
    e = Examinee(examinee.idx, examinee.id, examinee.latent)
    return e::Examinee
end


@sync @distributed for rep = 17:24

    I_operational = 1_000
    I_field = 25
    I_total = I_field + I_operational
    N = 3_000
    
    test_operational  = 25
    test_field = 5
    test_length = test_field + test_operational
    
    iter_mcmc_latent = 2_000
    iter_mcmc_item = 4_000
    #after how many responses update item parameter estimates
    batch_size = 5
    N_T = 400
    items = load(string("test/online-calibration/true values/true_items.jld2"), "items")
    examinees = load(string("test/online-calibration/true values/true_examinees.jld2"), "examinees")[1:N]
    
    retired_items = 0
    retired_items_vector= fill(0, N)

    # RESPONSES

    responses_per_examinee = [Response[] for e in examinees];

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

    items_est_field = [Item(i+I_operational, string("item_", i+I_operational), ["math"], Parameters2PL(Product([a_est_prior, b_est_prior]), a_est_bounds, b_est_bounds)) for i = 1 : I_field];
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
        i.parameters.expected_information = Psychometrics._expected_information_item(i.parameters, Latent1D(0.0))
    end, items_est_field)
    # append not calibrated items 
    items_est = vcat(items_est_operational, items_est_field);
    #responses vectors for not calibrated items
    responses_not_calibrated = Response[]

    ## EXAMINEES

    latent_est_prior = Normal(0, 3);
    latent_est_bounds = [-6.0, 6.0];
    examinees_est = [Examinee(e, string("examinee_",e), Latent1D(latent_est_prior, latent_est_bounds)) for e = 1 : N]; 

    #set starting value taking a random value in the interval -0.5:0.5
    map(e ->
            begin
                e.latent.val = Random.rand(-0.3:0.3)
            end,
    examinees_est)
    items_idx_per_examinee = Vector{Vector{Int64}}(undef, N);
    examinees_est_theta = [[zero(Float64) for i=1:test_length] for n=1:N];
    items_est_parameters = [[[0.0, 0.0] for n=1:ceil(N_T/batch_size)] for i=1:I_field]
    #@save string("test/online-calibration/true values/priors.jld2") a_est_prior b_est_prior latent_est_prior 
    #@save string("test/online-calibration/true values/bounds.jld2") a_est_bounds b_est_bounds latent_est_bounds 

    # START ONLINE-CALIBRATION

    global n=1
    while retired_items < I_field #size(filter(i -> i.parameters.calibrated == false, items_est),1) > 0   
        examinee = copy(examinees_est[n])
        examinee_true = copy(examinees[n])
        idx_n = examinee.idx
        items_idx_n = Int64[]
        responses_n = Response[]
        items_est_n = AbstractItem[]
        println("n: ", n)
        #println("true: ", examinees[n].latent.val)
        available_items_idx = map(i -> i.idx, filter(i2 -> i2.parameters.calibrated, items_est))
        for i in 1:test_operational
            # find best item idx
            next_item_idx = find_best_item(examinee, items_est[available_items_idx]);
            push!(items_idx_n, next_item_idx);
            available_items_idx = setdiff(available_items_idx, next_item_idx)
            # answer to the item
            resp = answer(examinee_true, items[next_item_idx])
            push!(responses_n, resp)
            estimate_ability!(examinee, items_est[items_idx_n], responses_n; mcmc_iter = iter_mcmc_latent, sampling = false, already_sorted = false)
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
                # answer to the item
                resp = answer(examinee_true, items[next_item_idx])
                push!(responses_n, resp)

                # set the theta prior as theta posterior
                #examinee.latent.prior = examinee.latent.posterior
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

                    examinees_est_item = examinees_est[map( r -> r.examinee_idx, resp_item)]
                    calibrate_item!(item, examinees_est_item, resp_item; mcmc_iter = iter_mcmc_item, sampling = false, already_sorted = true)
                    #save chain
                    if rep ==1
                        chain = item.parameters.chain
                        @save string("test/online-calibration/rep_", rep, "_item_", item.idx, "_resp_", size(resp_item, 1), "_chain.jld2") chain
                    end
                    items_est_parameters[next_item_idx - I_operational][Int64(size(resp_item,1)/batch_size)] = [item.parameters.a, item.parameters.b]
                    if next_item_idx==1001
                    println("est_BIAS a: ", sqrt((item.parameters.a - items[next_item_idx].parameters.a)^2))
                    println("est_BIAS b: ", sqrt((item.parameters.b - items[next_item_idx].parameters.b)^2))
                    end
                    if size(resp_item, 1) >= N_T
                        item.parameters.calibrated = true
                        retired_items +=1
                        #remove all responses from responses_not_calibrated
                        responses_not_calibrated = filter( r -> r.item_idx != item.idx, responses_not_calibrated)
                    end
                    #update expected information
                    item.parameters.expected_information = mapreduce(e -> Psychometrics._expected_information_item(item.parameters, e.latent), +, examinees_est_item)
                else
                    #update expected information
                    item.parameters.expected_information += Psychometrics._expected_information_item(item.parameters, examinee.latent)
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

    rmse_a =  sqrt(mean(map( (i, i_est)-> (i.parameters.a - i_est.parameters.a)^2, items[(I_operational+1):I_total], items_est[(I_operational+1):I_total]))) 
    rmse_b = sqrt(mean(map( (i, i_est)-> (i.parameters.b - i_est.parameters.b)^2, items[(I_operational+1):I_total], items_est[(I_operational+1):I_total])))
    rmse_theta = sqrt(mean(map( (e, e_est)-> (e.latent.val- e_est.latent.val)^2, examinees[1:(n-1)], examinees_est[1:(n-1)])))
    println( "avg RMSE a = ", rmse_a)
    println( "avg RMSE b = ", rmse_b)
    println( "avg RMSE tehta = ", rmse_theta)
    items_est_field = items_est[(I_operational+1):I_total]
    @save string("test/online-calibration/rep_", rep, "_items_est_field.jld2") items_est_field
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
rep=100

a_s = zeros(I_total,rep)
b_s = zeros(I_total,rep)
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