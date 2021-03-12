@everywhere using Distributions
@everywhere using StatsBase

@everywhere using ATA
@everywhere using DataFrames
@everywhere using JuMP
@everywhere using Cbc
@everywhere using CSV

function bs()
    I_total = 250
    N = 3_000
    metric = [0.0, 1.0]

    #pre-test settings
    T = 6
    test_length = 50
    nDifficulty = 3 #levels of guessed difficulty (low, medium, high)
    alphas = [0.25, 0.75] #difficulty cuts

    ### generate true items

    a_dist = LogNormal(0, 0.25); # mu= exp(0.4 + (0.5*0.1^2)), sigma^2= exp(0.4 + 0.1^2)*(exp(0.1^2) - 1)
    a_bounds = [0.5, 5.0];
    b_dist = Normal(0, 1);
    b_bounds = [-4.0, 4.0];
    items = [Item(i, string("item_", i), ["math"], Parameters2PL(Product([a_dist, b_dist]), a_bounds, b_bounds)) for i = 1:I_total];

    ## generate true examinees
        
    latent_dist = Normal(0.0, 1.0)
    latent_bounds = [-4.0, 4.0]
    examinees = Psychometrics.Examinee[]
    for n in 1:N
        push!(examinees,  Examinee(n, string("examinee_", n), Latent1D(latent_dist, latent_bounds))) 
    end

    ## pre-test assembly  (needs ATA package)
    # generate guessed difficulties @everywhere using true difficulties and quantiles
    bQles = quantile(map(i -> i.parameters.b, items), alphas)
    guessed_difficulty = zeros(I_total)
    for i=1:I_total
        if items[i].parameters.b < bQles[1]
            guessed_difficulty[i]=1
        elseif items[i].parameters.b < bQles[2]
            guessed_difficulty[i]=2
        else
            guessed_difficulty[i]=3
        end
    end
    #save true bank
    item_bank = DataFrame(
        a = map(i -> i.parameters.a, items),
        b = map(i -> i.parameters.b, items),
        guessed_difficulty = string.(Int.(guessed_difficulty))
    )
    CSV.write("item_bank.csv", item_bank)

    #write constraints and save in csv
    constraints = DataFrame(
        group = [1, 1, 1],
        var = ["guessed_difficulty", "guessed_difficulty", "guessed_difficulty"],
        value = [1, 2, 3],
        min = [
            Int(floor(alphas[1]*test_length - 2)),
            Int(floor((alphas[2] - alphas[1])*test_length - 2)),
            Int(floor((1 - alphas[2])*test_length - 2))
            ],
        max = [
            Int(ceil(alphas[1]*test_length + 2)),
            Int(ceil((alphas[2] - alphas[1])*test_length + 2)),
            Int(ceil((1 - alphas[2])*test_length + 2))
            ],
        weight = [1.0, 1.0, 1.0]
    )
    CSV.write("constraints.csv", constraints)

    ata_model = compact_ata(
        settings_file = "pre_test_settings.jl",
        bank_file = "item_bank.csv",
        bank_delim = ",",
        add_friends = false,
        add_enemies = false,
        add_constraints = true,
        constraints_file = "constraints.csv",
        constraints_delim = ",",
        add_overlap = true,
        overlap_file = "op.csv",
        overlap_delim = ",",
        add_exp_score = false,
        group_by_friends = false,
        add_obj_fun = false,
        solver = "jumpATA", 
        print_it = true,
        print_folder = "RESULTS",
        plot_it = false,
        optimizer_constructor = "Cbc",
        optimizer_attributes = [("seconds", 500), ("logLevel", 1)]
    )
    
    ## generate responses, use ata_model.output.design to assign items to the examinees
    # 1:T examinees get the 1:T test forms, 
    responses = Response[]
    t = 1
    for n = 1:N
        for i in items[findall(ata_model.output.design[:, t] .> 0)]
            push!(responses, answer(examinees[n], i))
        end
        t += 1
        if t > T
            t = 1
        end
    end
    responses =  vcat(map( e -> answer(e, items), examinees)...)

    #start points and probs
    prob = Distributions.pdf(Normal(metric[1], metric[2]), collect(-6.0:0.3:6.0))
    prob = prob / sum(prob)
    dist = Distributions.DiscreteNonParametric(collect(-6.0:0.3:6.0), prob)

    # initialize examinees
    latent_est_prior = dist
    latent_est_bounds = [-4.0, 4.0]
    examinees_est = [Examinee(e, string("examinee_",e), Latent1D(latent_est_prior, latent_est_bounds)) for e = 1 : N]; 
    #set starting value taking a random value in the interval -0.5:0.5
    map(e ->
        begin
            e.latent.val = 0.0
        end,
    examinees_est)

    #initalize items estimates
    a_est_bounds = [1e-5, 5.0];
    b_est_bounds = [-4.0, 4.0];

    items_est = [Item(i, string("item_", i), ["math"], Parameters2PL(a_est_bounds, b_est_bounds)) for i = 1 : I_total];
    map( i -> begin 
        i.parameters.calibrated = false
    end, items_est);
    # starting values
    map( i -> begin
        i.parameters.a = 1.0
    end, items_est);
    map( i -> begin
        i.parameters.b = 0.0
    end, items_est);

    #start calibration
    Psychometrics.joint_estimate_mmle_quick!(
        items_est,
        examinees_est,
        responses;
        metric = metric,
        max_iter = 500,
        max_time = 500,
        x_tol_rel = 0.001,
        );

    items_est_bs = map(i -> i, items_est)
    examinees_est_bs = map(e -> e, examinees_est)

    bootstrap!(
        items_est_bs,
        examinees_est_bs,
        responses;
        method="mmle",
        metric = metric,
        max_iter = 500,
        x_tol_rel = 0.001,
        replications = 100,
        type = "parametric",
        sample_fraction = 1.0
        )

        println("a RMSE")
        println(sqrt(StatsBase.mean(map( (i, i_est) -> (i.parameters.a-i_est.parameters.a)^2, items, items_est))))
        println("b RMSE")
        println(sqrt(StatsBase.mean(map( (i, i_est) -> (i.parameters.b-i_est.parameters.b)^2, items, items_est))))
        println("latent RMSE")
        println(sqrt(StatsBase.mean(map( (i, i_est) -> (i.latent.val-i_est.latent.val)^2, examinees, examinees_est))))
    return examinees, examinees_est, items, items_est, responses, items_est_bs, examinees_est_bs
end
 examinees, examinees_est, items, items_est, responses, items_est_bs, examinees_est_bs = bs();
@everywhere using JLD2
@save items_est_bs "items.jld2s"