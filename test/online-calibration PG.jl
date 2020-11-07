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

const I = 100
I_to_calibrate = 20
const N = 2_000
test_length = 30
field_test_items = 3
true_items  = test_length - field_test_items
iter_mcmc_latent = 2_000
iter_mcmc_item = 2_000
#after how many responses update item parameter estimates
required_responses = 500
maximum_required_responses = 500

# ITEM PARAMETERS AND LATENTS 

a_prior = LogNormal(0.2,0.3);
a_bounds = [1e-5,Inf];
b_prior = Normal(0, 1);
b_bounds = [-Inf, Inf];
items = [Item2PL(i, string("item_", i), ["math"], Parameters2PL(Product([a_prior, b_prior]), a_bounds, b_bounds)) for i = 1:I];

latent_prior = Normal(0, 1)
latent_bounds = [-Inf, Inf]
examinees = vcat([[Examinee1D(e, string("examinee_", e), Latent1D(latent_prior, latent_bounds)) for theta in -4:1:4] for e = 1:Int(floor(N/9))]...);

# RESPONSES

responses_per_examinee = [generate_response([e], items) for e in examinees];

#INITIAL VALUES

a_est_prior = TruncatedNormal(1.0, 0.5, 0.0, Inf);
a_est_bounds = [1e-5, 5.0];
b_est_prior = Normal(0,1);
b_est_bounds = [-6.0, 6.0];
#first I-I_to_calibrate items have true values of parameters (very well estimated)
items_est_calibrated = items[1:(I-I_to_calibrate)];
items_est_not_calibrated = [Item2PL(i+(I-I_to_calibrate), string("item_", i+(I-I_to_calibrate)), ["math"], Parameters2PL(Product([a_est_prior, b_est_prior]), a_est_bounds, b_est_bounds)) for i = 1 : I_to_calibrate];
map( i -> i.parameters.calibrated = false, items_est_not_calibrated)
items_est = vcat(items_est_calibrated, items_est_not_calibrated)
#responses vectors for not calibrated items
responses_not_calibrated = Response[]

latent_est_prior = Normal(0, 1);
latent_est_bounds = [-6.0, 6.0];
    examinees_est = [Examinee1D(e, string("examinee_",e), Latent1D(latent_est_prior, latent_est_bounds)) for e = 1 : N]; 

#set starting value taking a random value in the interval -0.5:0.5
map(e -> begin e.latent.val = Random.rand(-0.5:0.5) end, examinees_est)
items_idx_per_examinee = Vector{Vector{Int64}}(undef, N);
examinees_est_theta = [[zero(Float64) for i=1:test_length] for n=1:N];
#START ONLINE-CALIBRATION

for n in 1:N
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
        @sync @distributed for iter in 1:iter_mcmc_latent
            W = generate_w(items_est[items_idx_n], examinees_n)
            chain_n[iter] = rand(posterior(examinees_n, items_est_n, resp_n, map(w -> w.val, W)))
            #mcmc_iter!(examinees_n, items_est[items_idx_n], resp_n, map(w -> w.val, W); sampling = false)
        end
        examinees_n.latent.chain = chain_n
        update_estimate!(examinees_n)
        #examinees_n.latent.chain=Float64[]
        examinees_est_theta[n][i] = examinees_n.latent.val
    end
    available_items_idx = map(i -> i.idx, filter(i2 -> !i2.parameters.calibrated, items_est))
    if size(available_items_idx,1) > 0
        for i in 1:field_test_items
            next_item_idx = find_best_item(examinees_n, items_est[available_items_idx]; method = "D-gain");
            push!(items_idx_n, next_item_idx);
            available_items_idx = setdiff(available_items_idx, next_item_idx)
            sort!(items_idx_n)
            resp_n = responses_n[items_idx_n]
            chain_n = SharedArray{Float64}(iter_mcmc_latent)
            items_est_n = items_est[items_idx_n]
            for iter in 1:iter_mcmc_latent
                W = generate_w(items_est[items_idx_n], examinees_n)
                chain_n[iter] = rand(posterior(examinees_n, items_est_n, resp_n, map(w -> w.val, W)))
                #mcmc_iter!(examinees_n, items_est_n, resp_n, map(w -> w.val, W); sampling = false)
            end
            examinees_n.latent.chain = chain_n
            update_estimate!(examinees_n)
            #examinees_n.latent.chain=Float64[]
            examinees_est_theta[n][i+(true_items)] = examinees_n.latent.val
            #println("est_BIAS: ", examinees_est[n].latent.val - examinees[n].latent.val)
            push!(responses_not_calibrated, responses_n[next_item_idx])
            resp_item = sort(filter(r -> r.item_idx == next_item_idx, responses_not_calibrated), by = r -> r.examinee_idx)
            if mod(size(resp_item, 1), required_responses) == 0
                println("item idx: ", next_item_idx)
                println("# responses: ", size(resp_item, 1))
                println("calibrate item ", next_item_idx)
                println("true pars ", items[next_item_idx].parameters.a," ", items[next_item_idx].parameters.b)
                examinees_est_item = examinees_est[map( r -> r.examinee_idx, resp_item)]
                item = items_est[next_item_idx]
                calibrate_item!(item, resp_item, examinees_est_item)
                println("est pars ", item.parameters.a," ",item.parameters.b)
            end
            if size(resp_item, 1)>= maximum_required_responses
                item.parameters.calibrated = true
            end
        end
    end
    println("est_BIAS: ", examinees_n.latent.val - examinees[n].latent.val)
    examinees_est[n] = copy(examinees_n)
    items_idx_per_examinee[n] = copy(items_idx_n)
end


println( "avg RMSE a = ", sqrt(mean(map( (i, i_est)-> (i.parameters.a - i_est.parameters.a)^2, items, items_est))))
println( "avg RMSE b = ", sqrt(mean(map( (i, i_est)-> (i.parameters.b - i_est.parameters.b)^2, items, items_est))))
println( "avg RMSE tehta = ", sqrt(mean(map( (e, e_est)-> (e.latent.val- e_est.latent.val)^2, examinees, examinees_est))))

rmse_theta = fill(Float64[],9)
for t in 1:9
    println(t)
    last_i =trunc(Int,N/9)*9
    rmse_theta[t] = map((e,e_est) -> abs(e.latent.val-e_est.latent.val), examinees[collect(t:9:last_i)], examinees_est[collect(t:9:last_i)])
end
plot(rmse_theta)