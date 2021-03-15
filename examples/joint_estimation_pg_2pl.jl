using Dates
using Distributions
using Pkg

Pkg.activate(".")
using Psychometrics

function est_pg()
    I_total = 50
    N = 3000

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
        ) for i = 1:I_total
    ];
    examinees = [Examinee(e, string("examinee_", e), Latent1D(Normal(0.0, 1.0), [-6.0, 6.0])) for e = 1:N];
    # RESPONSES
    responses = answer(examinees, items);

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
        ) for i = 1:I_total
    ];
    map(i -> i.parameters.a = 1.0, items_est);
    map(i -> i.parameters.b = 0.0, items_est);


    examinees_est =
        [Examinee(e, string("examinee_", e), Latent1D(Normal(0, 1), [-6.0, 6.0])) for e = 1:N];
    for n = 1:N
        # examinees_est[n].latent.val = examinees[n].latent.val + rand(Normal(0.0,0.3))
        examinees_est[n].latent.val = 0.0
        #examinees_est[n].latent.prior = Normal(examinees[n].latent.val, 0.3)
    end

    joint_estimate!(
        items_est,
        examinees_est,
        responses;
        method = "pg",
        quick = true,
        mcmc_iter = 2000,
        max_time = 400,
        item_sampling = false,
        examinee_sampling = false
        );

    println("a RMSE")
    println(sqrt(mean(map( (i, i_est) -> (i.parameters.a-i_est.parameters.a)^2, items, items_est))))
    println("b RMSE")
    println(sqrt(mean(map( (i, i_est) -> (i.parameters.b-i_est.parameters.b)^2, items, items_est))))
    println("latent RMSE")
    println(sqrt(mean(map( (i, i_est) -> (i.latent.val-i_est.latent.val)^2, examinees, examinees_est))))
    return examinees, examinees_est, items, items_est, responses
end
examinees, examinees_est, items, items_est, responses = est_pg();

