
using Distributions
using NLopt
using Interpolations
using Pkg;
Pkg.activate(".");
using Psychometrics

function est()
    I_total = 40
    N = 500
    test_length = 40
    metric = [0.0, 1.0]

    ### generate true items

    a_dist = LogNormal(0, 0.25) # mu= exp(0.4 + (0.5*0.1^2)), sigma^2= exp(0.4 + 0.1^2)*(exp(0.1^2) - 1)
    a_bounds = [1e-5, 5.0]
    b_dist = Normal(0, 1)
    b_bounds = [-4.0, 4.0]
    items = [
        Item(
            i,
            string("item_", i),
            ["math"],
            Parameters2PL(Product([a_dist, b_dist]), a_bounds, b_bounds),
        ) for i = 1:I_total
    ]

    ## generate true examinees

    latent_dist = Normal(0, 1)
    latent_bounds = [-4.0, 4.0]
    examinees = Psychometrics.Examinee[]
    for n = 1:N
        push!(
            examinees,
            Examinee(n, string("examinee_", n), Latent1D(latent_dist, latent_bounds)),
        )
    end

    ## generate responses
    responses = vcat(map(e -> answer(e, items), examinees)...)

    #start points and probs
    probs = Distributions.pdf(Normal(metric[1], metric[2]), collect(-6.0:0.3:6.0))
    probs = probs / sum(probs)
    dist = Distributions.DiscreteNonParametric(collect(-6.0:0.3:6.0), probs)

    # initialize examinees
    latent_est_prior = dist
    latent_est_bounds = [-4.0, 4.0]
    examinees_est = [
        Examinee(e, string("examinee_", e), Latent1D(latent_est_prior, latent_est_bounds)) for e = 1:N
    ]
    #set starting value taking a random value in the interval -0.5:0.5
    map(e -> begin
        e.latent.val = 0.0
    end, examinees_est)

    #initalize items estimates
    a_est_bounds = [1e-5, 5.0]
    b_est_bounds = [-4.0, 4.0]

    items_est = [
        Item(i, string("item_", i), ["math"], Parameters2PL(a_est_bounds, b_est_bounds)) for i = 1:I_total
    ]
    map(i -> begin
        i.parameters.calibrated = false
    end, items_est)
    # starting values
    map(i -> begin
        i.parameters.a = 1.0
    end, items_est)
    map(i -> begin
        i.parameters.b = 0.0
    end, items_est)

    #start calibration
    calibrate_mmle!(
        items_est,
        examinees_est,
        responses;
        metric = [0.0, 1.0],
        max_iter = 500,
        x_tol_rel = 0.001,
    )
    println("a RMSE")
    println(
        sqrt(
            mean(
                map(
                    (i, i_est) -> (i.parameters.a - i_est.parameters.a)^2,
                    items,
                    items_est,
                ),
            ),
        ),
    )
    println("b RMSE")
    println(
        sqrt(
            mean(
                map(
                    (i, i_est) -> (i.parameters.b - i_est.parameters.b)^2,
                    items,
                    items_est,
                ),
            ),
        ),
    )
    println("latent RMSE")
    println(
        sqrt(
            mean(
                map(
                    (i, i_est) -> (i.latent.val - i_est.latent.val)^2,
                    examinees,
                    examinees_est,
                ),
            ),
        ),
    )
    return examinees, examinees_est, items, items_est, responses
end
examinees, examinees_est, items, items_est, responses = est();
