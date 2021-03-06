using Pkg
Pkg.activate(".")
using Psychometrics
using Distributions
using LinearAlgebra
using Dates
using Random
#using StatsPlots 


function est()
    I_total = 40
    N = 500

    # ITEM PARAMETERS AND LATENTS 
    items = [Item(i, string("item_",i), ["math"], Parameters2PL(Product([LogNormal(0.3, 0.2), Normal(0,1)]), [1e-5,5.0], [-6.0, 6.0])) for i = 1 : I_total];
    examinees = [Examinee(e, string("examinee_",e), Latent1D(Normal(0,1), [-6.0, 6.0])) for e = 1 : N]; 

    # RESPONSES

    responses = answer(examinees, items);

    # Get response matrix (zeros are for wrong answers and for missing answers)
    response_matrix = get_response_matrix(responses, I_total, N);

    #Set Seeds for Random Generation
    #Random.seed!()

    # function tr_norm_gen(number, mu, v)
    #     temp_u = Distributions.rand(Distributions.Uniform(zero(Float64),one(Float64)), number)
    #     norm_dist = Distributions.Normal(mu, sqrt(v))
    #     norm_cdf = Distributions.cdf(norm_dist, zero(Float64))
    #     tr_norm_value = temp_u .- (temp_u .* norm_cdf) .+ norm_cdf
    #     return Distributions.quantile(norm_dist, tr_norm_value)
    # end

    # using StatsPlots
    # A short example of sampling truncated standard normal distribution 
    # histogram(tr_norm_gen(800,0,1))
    # #histogram(truncate_rand(Normal(0.0,1.0), [0.0, Inf], n=800))
    # histogram(rand(Distributions.TruncatedNormal(0.0,1.0, 0.0, Inf), 800))

    # Simulation code by Jiang and Templin 
    # Gibbs Samplers for Logistic Item Response Models via the Pólya–GammaDistribution: 
    # A Computationally Efficient Data-Augmentation StrategyArticleinPsychometrika 
    # October 2018 DOI: 10.1007/s11336-018-9641-xCITATIONS2READS692 

    ##################################################################################################
    ###########################      Data and Settings    ############################################
    ##################################################################################################

    #Specified by users:Response matrix X
    X = permutedims(response_matrix);
    ##################################################################################################
    ###########################      PolyGamma MCMC sampler   ########################################
    ##################################################################################################

    #K matrix for the coming PolyaGamma samplings
    K = X .- 0.5;

    #Initial Values, need to make sure that all Variance needs to be positive
    ini_a = ones(I_total);
    ini_b = zeros(I_total);
    ini_theta = zeros(N);
    ini_kernel = zeros(N, I_total);


    ini_w = [Distributions.rand(Psychometrics.PolyaGamma(1.0, ini_a[i] * (ini_theta[p] - ini_b[i]))) for p = 1:N, i = 1:I_total] ;

    function theta_sampler(
        vec_a, # I_total x 1
        vec_b, # I_total x 1
        mat_w_id, # N x 1
        mat_k_id, # N x 1
        theta_prior_mu, # 1 X 1
        theta_prior_var,# 1 X 1
    )
        omega_theta = LinearAlgebra.Diagonal(mat_w_id) # I_total x I_total
        theta_variance = 1 / ((vec_a' * omega_theta * vec_a) + (1 / theta_prior_var)) # 1 x 1
        z_theta = vec_a .* vec_b .* mat_w_id + mat_k_id # I_total x 1 # times w_e
        theta_mu = theta_variance * ((vec_a' * z_theta) + (theta_prior_mu / theta_prior_var))
        rand(Distributions.Normal(theta_mu, sqrt(theta_variance)))
    end
    function a_sampler(
        vec_theta, # N x 1
        vec_b_id, # I_total x 1
        mat_w_id, # N x I_total
        mat_k_id, # N x I_total
        a_prior_mu, # 1 x 1
        a_prior_var, # 1 x 1
    )
        omega_a = LinearAlgebra.Diagonal(mat_w_id) # N x N
        b_omega_a = vec_theta .- vec_b_id
        a_variance =
            1 / (
                (b_omega_a' * omega_a * b_omega_a) +
                (1 / a_prior_var)
            ) # 1 x 1 
        #z_a = mat_k_id ./ mat_w_id # don't need it
        a_mu = a_variance * (b_omega_a' * mat_k_id + (a_prior_mu / a_prior_var)) # ok
        #return tr_norm_gen(1, a_mu, a_variance)[1]
        return  rand(Distributions.TruncatedNormal(a_mu, sqrt(a_variance), 0.0, Inf))
    end
    #  ( T   -   b   ) ^2 *  w
    # N x 1   I_total x 1        N x I_total

    function b_sampler(vec_theta, vec_a_id, mat_w_id, mat_k_id, b_prior_mu, b_prior_var, N)
        omega_b = LinearAlgebra.Diagonal(mat_w_id) # N x N
        vec_a_id_vec = fill(vec_a_id, N) #N
        b_variance = 1 / (vec_a_id_vec' * omega_b * vec_a_id_vec + (1 / b_prior_var))  # 1 x 1 ok, vec_a_id_vec can be positive because it is squared anyway
        z_b = mat_k_id .- (vec_a_id .* vec_theta .* mat_w_id) # N x 1 # times w_i
        b_mu = b_variance * (((.-vec_a_id_vec)' * z_b) + (b_prior_mu / b_prior_var)) # ok
        return rand(Distributions.Normal(b_mu, sqrt(b_variance)))
    end
    function i_sampler(vec_theta, vec_a_id, vec_b_id, mat_w_id, mat_k_id, a_prior_mu, a_prior_var, b_prior_mu, b_prior_var)
        sigma2 = mapreduce(
            (e, w) -> [(e - vec_b_id)^2, vec_a_id^2 ] .* w,
            +,
            vec_theta,
            mat_w_id,
        )
        sigma2 = 1 ./ (sigma2 + (1 ./ [a_prior_var, b_prior_var]))
        mu = mapreduce(
            (e, w, r) -> [
                (e - vec_b_id) *
                (r),
                -vec_a_id * (
                    #(get_responses_by_examinee_id(e.id, responses)[1].val - 0.5) -
                    (r) -
                    (vec_a_id * e * w)
                ),
            ],
            +,
            vec_theta,
            mat_w_id,
            mat_k_id
        )
        mu =
            sigma2 .* (
                mu + (
                    [a_prior_mu, b_prior_mu] ./
                    [a_prior_var, b_prior_var]
                )
            )
        return rand(Distributions.Product([
            Distributions.TruncatedNormal(mu[1], sqrt(sigma2[1]), 0.0, Inf),
            Distributions.Normal(mu[2], sqrt(sigma2[2])),
        ]))
    end
    # using RCall
    # R""" 
    # rpg_sp <- function(x){
    # BayesLogit::rpg(1,1,x)
    # }
    # """
    function w_sampler(vec_theta, vec_a, vec_b, N, I_total) 
        #w_temp = [vec_a[i] * (vec_theta[p] - vec_b[i]) for p = 1:N, i = 1:I_total]
        #pg = rcopy(R"apply($w_temp, c(1,2), rpg_sp)")
        pg =[rand(PolyaGamma(1.0, vec_a[i] * (vec_theta[p] - vec_b[i]))) for p = 1:N, i = 1:I_total]
        return pg
    end

    ##################################################################################################
    ###########################   Estimation Starts from here  #######################################
    ##################################################################################################

    #Specified by users:Construct 2PL IRT Estimator

    Iter = 4000
    theta_prior_mu = 0
    theta_prior_var = 1
    a_prior_mu = 1
    a_prior_var = 5
    b_prior_mu = 0
    b_prior_var = 5

    sav_a = zeros(Float64, Iter, I_total);
    sav_b = zeros(Float64, Iter, I_total);
    sav_theta = zeros(Float64, Iter, N);

    sav_a[1, :] = ini_a
    sav_b[1, :] = ini_b
    sav_theta[1, :] = ini_theta
    #sav.iter.hist<-matrix(0,N,I_total)

    for iter = 2:Iter
        sav_w = w_sampler(sav_theta[iter-1, :], sav_a[iter-1, :], sav_b[iter-1, :], N, I_total)
        if mod(iter,100)==0 
            println(iter)
        end
        sav_theta_iter = map(
            p -> begin
                no_missing_i = findall(.!ismissing.(K[p, :]))
                theta_sampler(
                sav_a[iter-1, no_missing_i],
                sav_b[iter-1, no_missing_i],
                sav_w[p, no_missing_i],
                K[p, no_missing_i],
                theta_prior_mu,
                theta_prior_var,
            )
        end
        ,
        1:N
        )
        res_iter = map(
            i -> begin
                no_missing_e = findall(.!ismissing.(K[:, i]))
                i_sampler(
                    sav_theta_iter,
                    sav_a[iter-1, i],
                    sav_b[iter-1, i],
                    sav_w[no_missing_e, i],
                    K[no_missing_e, i],
                    a_prior_mu,
                    a_prior_var,
                    b_prior_mu,
                    b_prior_var
                )
            end,
            1:I_total,
        )
        sav_a_iter = map( r -> r[1], res_iter)
        sav_b_iter = map( r -> r[2], res_iter)

        # sav_a_iter = map(
        #     i -> a_sampler(
        #         sav_theta_iter,
        #         sav_b[iter-1, i],
        #         sav_w[:, i],
        #         K[:, i],
        #         a_prior_mu,
        #         a_prior_var,
        #     ),
        #     1:I_total,
        # )
        # sav_b_iter = map(
        #     i -> b_sampler(
        #         sav_theta_iter,
        #         sav_a_iter[i],
        #         sav_w[:, i],
        #         K[:, i],
        #         b_prior_mu,
        #         b_prior_var,
        #         N,
        #     ),
        #     1:I_total,
        # )
        sav_theta[iter, :] .= copy(sav_theta_iter)
        sav_a[iter, :] .= copy(sav_a_iter)
        sav_b[iter, :] .= copy(sav_b_iter)
    end

    sampled = (mcmc_iter-1000):mcmc_iter
    mean_a = [mean(sav_a[sampled, i]) for i = 1:I_total];
    mean_b = [mean(sav_b[sampled, i]) for i = 1:I_total];
    mean_theta = [mean(sav_theta[sampled, i]) for i = 1:N];

    # hcat(map(i -> i.parameters.a, items), mean_a)
    # hcat(map(i -> i.parameters.b, items), mean_b)


    # RMSEs
    println(sqrt(sum((map(i -> i.parameters.a, items) .- mean_a).^2)/I_total))
    println(sqrt(sum((map(i -> i.parameters.b, items) .- mean_b).^2)/I_total))
    println(sqrt(sum((map(e -> e.latent.val, examinees) .- mean_theta).^2)/N))

    return examinees, examinees_est, items, items_est, responses
end
examinees, examinees_est, items, items_est, responses = est();


# using StatsPlots 
# plot(sav_a[:,1])


# RMSE_a_500 = zeros(Iter-500);
# RMSE_b_500 = zeros(Iter-500);
# RMSE_t_500 = zeros(Iter-500);
# for j in 1:(Iter-500)
# mean_a = [mean(sav_a[j:(j+499), i]) for i = 1:I_total];
# mean_b = [mean(sav_b[j:(j+499), i]) for i = 1:I_total];
# mean_theta = [mean(sav_theta[j:(j+499), i]) for i = 1:N];
# # RMSEs
# RMSE_a_500[j]=sqrt(sum((map(i -> i.parameters.a, items) .- mean_a).^2)/I_total)
# RMSE_b_500[j]=sqrt(sum((map(i -> i.parameters.b, items) .- mean_b).^2)/I_total)
# RMSE_t_500[j]=sqrt(sum((map(e -> e.latent.val, examinees) .- mean_theta).^2)/N)
# end

# plot(hcat(RMSE_a_500, RMSE_b_500, RMSE_t_500))

# RMSE_a_100 = zeros(Iter-100);
# RMSE_b_100 = zeros(Iter-100);
# RMSE_t_100 = zeros(Iter-100);
# for j in 1:(Iter-100)
# mean_a = [mean(sav_a[j:(j+99), i]) for i = 1:I_total];
# mean_b = [mean(sav_b[j:(j+99), i]) for i = 1:I_total];
# mean_theta = [mean(sav_theta[j:(j+99), i]) for i = 1:N];
# # RMSEs
# RMSE_a_100[j]=sqrt(sum((map(i -> i.parameters.a, items) .- mean_a).^2)/I_total)
# RMSE_b_100[j]=sqrt(sum((map(i -> i.parameters.b, items) .- mean_b).^2)/I_total)
# RMSE_t_100[j]=sqrt(sum((map(e -> e.latent.val, examinees) .- mean_theta).^2)/N)
# end

# plot(hcat(RMSE_a_100, RMSE_b_100, RMSE_t_100))