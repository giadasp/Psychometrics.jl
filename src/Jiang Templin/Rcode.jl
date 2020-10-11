using Pkg
Pkg.activate(".")
using Psychometrics
using Distributions
using LinearAlgebra
using Dates
using Random

const I = 100
const N = 1_000


# ITEM PARAMETERS AND LATENTS 

items = [Item2PL(i, string("item_", i), ["math"], Parameters2PL()) for i = 1:I];
examinees = [Examinee1D(e, string("examinee_", e), Latent1D()) for e = 1:N];

# RESPONSES

responses = generate_response(examinees, items);

# Get response matrix (zeros are for wrong answers and for missing answers)
response_matrix = get_response_matrix(responses, I, N)

#Set Seeds for Random Generation
Random.seed!(09192016)

# function tr_norm_gen(number, mu, v)
#     temp_u = Distributions.rand(Distributions.Uniform(zero(Float64),one(Float64)), number)
#     norm_dist = Distributions.Normal(mu, sqrt(v))
#     norm_cdf = Distributions.cdf(norm_dist, zero(Float64))
#     tr_norm_value = temp_u .- (temp_u .* norm_cdf) .+ norm_cdf
#     return Distributions.quantile(norm_dist, tr_norm_value)
# end

# using StatsPlots
# #A short example of sampling truncated standard normal distribution 
# histogram(tr_norm_gen(800,0,1))
# #histogram(truncate_rand(Normal(0.0,1.0), [0.0, Inf], n=800))
# histogram(rand(Distributions.TruncatedNormal(0.0,1.0, 0.0, Inf), 800))

##################################################################################################
###########################      Data and Settings    ############################################
##################################################################################################

#Specified by users:Response matrix X
X = permutedims(response_matrix)
##################################################################################################
###########################      PolyGamma MCMC sampler   ########################################
##################################################################################################

#K matrix for the coming PolyaGamma samplings
K = X .- 0.5

#Initial Values, need to make sure that all Variance needs to be positive
ini_a = rand(Distributions.Uniform(0.01, 3), I)
ini_b = rand(Distributions.Uniform(-2.0, 2), I)
ini_theta = rand(Distributions.Uniform(-1.6, 1.98), N)
ini_kernel = zeros(N, I)

for p = 1:N
    ini_kernel[p, :] = ini_a .* (ini_theta[p] .- ini_b)
end
ini_w = zeros(N, I)

for p = 1:N
    for i = 1:I
        ini_w[p, i] = Distributions.rand(PolyaGamma(1, ini_kernel[p, i]))
    end
end

function theta_sampler(
    vec_a, # I x 1
    vec_b, # I x 1
    mat_w_id, # N x 1
    mat_k_id, # N x 1
    theta_prior_mu, # 1 X 1
    theta_prior_var,# 1 X 1
)
    omega_theta = LinearAlgebra.Diagonal(mat_w_id) # I x I
    theta_variance = 1 / ((vec_a' * omega_theta * vec_a) + (1 / theta_prior_var)) # 1 x 1
    z_theta = vec_a .* vec_b .* mat_w_id + mat_k_id # I x 1 # times w_e
    theta_mu = theta_variance * (vec_a' * z_theta) + (theta_prior_mu / theta_prior_var)
    rand(Distributions.Normal(theta_mu, sqrt(theta_variance)))
end

function a_sampler(
    vec_theta, # N x 1
    vec_b_id, # I x 1
    mat_w_id, # N x I
    mat_k_id, # N x I
    a_prior_mu, # 1 x 1
    a_prior_var, # 1 x 1
)
    omega_a = LinearAlgebra.Diagonal(mat_w_id) # N x N
    a_variance =
        1 / (
            (((vec_theta .- vec_b_id)' * omega_a) * (vec_theta .- vec_b_id)) +
            (1 / a_prior_var)
        ) # 1 x 1 
    #z_a = mat_k_id ./ mat_w_id # don't need it
    a_mu = a_variance * ((vec_theta .- vec_b_id)' * mat_k_id + (a_prior_mu / a_prior_var)) # ok
    #return tr_norm_gen(1, a_mu, a_variance)[1]
    return rand(Distributions.TruncatedNormal(a_mu, sqrt(a_variance), 0.0, Inf))
end

function b_sampler(vec_theta, vec_a_id, mat_w_id, mat_k_id, b_prior_mu, b_prior_var, N)
    omega_b = LinearAlgebra.Diagonal(mat_w_id) # N x N
    vec_a_id_vec = fill(vec_a_id, N) #N
    b_variance = 1 / (vec_a_id_vec' * omega_b * vec_a_id_vec + (1 / b_prior_var))  # 1 x 1 ok, vec_a_id_vec can be positive because it is squared anyway
    z_b = mat_k_id .- (vec_a_id .* vec_theta .* mat_w_id) # N x 1 # times w_i
    b_mu = b_variance * (((.-vec_a_id_vec)' * z_b) + (b_prior_mu / b_prior_var)) # ok
    return rand(Distributions.Normal(b_mu, sqrt(b_variance)))
end

function w_sampler(vec_theta, vec_a, vec_b, N, I)
    w_temp = zeros(Float64, N, I)
    W_gen = copy(w_temp)
    w_sav = zeros(Float64, N, I, 2)
    w_temp = [vec_a[i] * (vec_theta[p] - vec_b[i]) for p = 1:N, i = 1:I]
    w_sav[:, :, 1] = [rand(PolyaGamma(1.0, w_temp[p, i])) for p = 1:N, i = 1:I]
    # pg = zeros(N, I)
    #pg = rcopy(R"apply($w_temp, c(1,2), rpg_sp)")
    #w_sav[:,:,1] = copy(pg)
    return w_sav
end

##################################################################################################
###########################     Estimation Starts from here#######################################
##################################################################################################

#Specified by users:Construct 2PL IRT Estimator

Iter = 2_000
theta_prior_mu = 0
theta_prior_var = 1
a_prior_mu = 1
a_prior_var = 5
b_prior_mu = 0
b_prior_var = 5

sav_a = zeros(Float64, Iter, I)
sav_b = zeros(Float64, Iter, I)
sav_theta = zeros(Float64, Iter, N)

sav_a[1, :] = ini_a
sav_b[1, :] = ini_b
sav_theta[1, :] = ini_theta
#sav.iter.hist<-matrix(0,N,I)

for iter = 2:Iter
    sav_W = w_sampler(sav_theta[iter-1, :], sav_a[iter-1, :], sav_b[iter-1, :], N, I)
    sav_w = sav_W[:, :, 1]
    println(iter)
    sav_theta_iter = map(
        p -> theta_sampler(
            sav_a[iter-1, :],
            sav_b[iter-1, :],
            sav_w[p, :],
            K[p, :],
            theta_prior_mu,
            theta_prior_var,
        ),
        1:N,
    )
    sav_a_iter = map(
        i -> a_sampler(
            sav_theta_iter,
            sav_b[iter-1, i],
            sav_w[:, i],
            K[:, i],
            a_prior_mu,
            a_prior_var,
        ),
        1:I,
    )
    sav_a_iter[findall(sav_a_iter .== Inf)] .= 0.001
    sav_b_iter = map(
        i -> b_sampler(
            sav_theta_iter,
            sav_a_iter[i],
            sav_w[:, i],
            K[:, i],
            b_prior_mu,
            b_prior_var,
            N,
        ),
        1:I,
    )
    println(sav_a_iter[1],sav_b_iter[1],sav_theta_iter[1])
    sav_theta[iter, :] = copy(sav_theta_iter)
    sav_a[iter, :] = copy(sav_a_iter)
    sav_b[iter, :] = copy(sav_b_iter)
end


mean_a = [mean(sav_a[1500:2000, i]) for i = 1:I]
mean_b = [mean(sav_b[1500:2000, i]) for i = 1:I]
mean_theta = [mean(sav_theta[1500:2000, i]) for i = 1:N]

# hcat(map(i -> i.parameters.a, items), mean_a)
# hcat(map(i -> i.parameters.b, items), mean_b)


# RMSEs
sum((map(i -> i.parameters.a, items) .- mean_a).^2)/I
sum((map(i -> i.parameters.b, items) .- mean_b).^2)/I
sum((map(e -> e.latent.val, examinees) .- mean_theta).^2)/N

using StatsPlots
