 using Pkg
 Pkg.instantiate()
 Pkg.activate(".")
 using Psychometrics
 using Distributions
 using JLD2
 using StatsBase
 using DelimitedFiles
 using FileIO


 function generate_true_values()
I_operational = 1000
I_field = 25
I_total = I_field + I_operational
N = 4_000

test_operational  = 25
test_field = 5
test_length = test_field + test_operational

iter_mcmc_latent = 2_000
iter_mcmc_item = 4_000
#after how many responses update item parameter estimates
batch_size = 5
N_T = 500

    # TRUE VALUES
    
    ## ITEMS

    ### operational items

    log_a_dist = Normal(0.4, sqrt(0.1)); # mu= exp(0.4 + (0.5*0.1^2)), sigma^2= exp(0.4 + 0.1^2)*(exp(0.1^2) - 1)
    a_bounds = [1e-5,Inf];
    b_dist = Normal(0, 1);
    b_bounds = [-6, 6];
    items_operational = [Item(i, string("item_", i), ["math"], Parameters2PL(Product([log_a_dist, b_dist]), a_bounds, b_bounds)) for i = 1:I_operational];
    map(i ->
        begin 
            i.parameters.a = exp(i.parameters.a)
        end,
    items_operational)

    ### field items
    quantiles_a = quantile(map(i -> i.parameters.a, items_operational), [0.2, 0.35, 0.5, 0.65, 0.8])
    quantiles_b = quantile(map(i -> i.parameters.b, items_operational), [0.2, 0.35, 0.5, 0.65, 0.8] )
    a_field = vcat([fill(q,5) for q in quantiles_a]...)
    b_field = vcat([quantiles_b for q in quantiles_a]...)

    items_field = [Item(i, string("item_", i), ["math"], Parameters2PL(Product([log_a_dist, b_dist]), a_bounds, b_bounds)) for i = (I_operational + 1) : I_total];
    map((i, a, b) ->
        begin
            i.parameters.a = a
            i.parameters.b = b
        end,
        items_field,
        a_field,
        b_field
    )
    items = vcat(items_operational, items_field)

    @save "examples/online-calibration/true values/true_items.jld2" items
    @save "examples/online-calibration/settings.jld2" I_total  I_field N test_length test_field test_operational iter_mcmc_latent iter_mcmc_item batch_size N_T

    # ## EXAMINEES
    
    latent_dist = Normal(0, 1)
    latent_bounds = [-Inf, Inf]
    examinees = Examinee[]
    global n=1
    for n in 1:N
        push!(examinees,  Examinee(n, string("examinee_", n), Latent1D(latent_dist, latent_bounds))) 
    end
    
    @save "examples/online-calibration/true values/true_examinees.jld2" examinees

end

generate_true_values()