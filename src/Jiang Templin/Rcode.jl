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

items = [Item2PL(i, string("item_",i), ["math"], Parameters2PL()) for i = 1 : I];
examinees = [Examinee1D(e, string("examinee_",e), Latent1D()) for e = 1 : N];   

# RESPONSES

responses = generate_response(examinees, items);

# Get response matrix (zeros are for wrong answers and for missing answers)
response_matrix = get_response_matrix(responses, I, N)

#Set Seeds for Random Generation
Random.seed!(09192016)

##################################################################################################
#######   Load package-BayesLogit-for polya-gamma distribution   #################################
##################################################################################################
#loading needed packages
# install_require_package = function(needed_packages){
#   for (i in 1:length(needed_packages)){
#     haspackage = require(needed_packages[i], character.only = TRUE)
#     if (haspackage==FALSE){
#       install.packages(needed_packages[i])
#       require(needed_packages[i], character.only = TRUE)
#     }
#   }
# }


# needed_packages = c("ROCR","stringr", "MASS","gtools","dplyr",'mirt')
# devtools::install_github('cran/BayesLogit')
# install_require_package(needed_packages = needed_packages)

##################################################################################################
##Funciton for sampling truncated/non-negative normal distribution:input is variance not SD#######
##################################################################################################
# tr_norm_gen<-function(number,mu,v){
#   temp_u<-runif(number,min=0,max=1)
#   tr_norm_value<-temp_u-temp_u*pnorm(0,mean=mu,sd=sqrt(v))+pnorm(0,mean=mu,sd=sqrt(v))
#   qnorm(tr_norm_value,mu,sqrt(v))
# }
function tr_norm_gen(number, mu, v)
    temp_u = Distributions.rand(Distributions.Uniform(zero(Float64),one(Float64)), number)
    norm_dist = Distributions.Normal(mu, sqrt(v))
    norm_cdf = Distributions.cdf(norm_dist, zero(Float64))
    tr_norm_value = temp_u .- (temp_u .* norm_cdf) .+ norm_cdf
    return Distributions.quantile(norm_dist, tr_norm_value)
end

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
Ni = size(X, 2)
Np = size(X, 1)
##################################################################################################
###########################      PolyGamma MCMC sampler   ########################################
##################################################################################################

#K matrix for the coming PolyaGamma samplings
K = X .- 0.5

#Initial Values, need to make sure that all Variance needs to be positive
#ini_a <- runif(Ni,0.01,3)
ini_a = rand(Distributions.Uniform(0.01,3), Ni)
#ini_b<-runif(Ni,-2,2)
ini_b = rand(Distributions.Uniform(-2.0, 2), Ni)
#ini_theta<-runif(Np,-1.6,1.98)
ini_theta = rand(Distributions.Uniform(-1.6, 1.98), Np)
#ini_kernel<- matrix(0,Np,Ni)
ini_kernel = zeros(Np, Ni)
# for (p in 1:Np){
#   ini_kernel[p,]<-ini_a*(ini_theta[p]-ini_b)	
# }

for p in 1:Np
    ini_kernel[p, :] = ini_a .* (ini_theta[p] .- ini_b)
end
# ini_w<-matrix(0,Np,Ni)
ini_w = zeros(Np, Ni)
# for (p in 1:Np){
#   ini_w[p,]<-rpg(Ni,rep(1,Ni),ini_kernel[p,])
# }
for p in 1:Np
    for i in 1:Ni
        ini_w[p,i] = Distributions.rand(PolyaGamma(1, ini_kernel[p, i]))
    end
end
#theta_sampler-sampling theta for a person(p) across items
# theta_sampler<-function(id,vec_a,vec_b,mat_w,mat_k,theta_prior_mu,theta_prior_var){
#   theta_w<-diag(mat_w[id,],Ni,Ni)
#   theta_V<-1/(t(vec_a)%*%theta_w%*%vec_a+(1/theta_prior_var))
#   if(theta_V<0){print(vec_a)}
#   theta_mean_multiplier<-vec_a*vec_b*mat_w[id,]+mat_k[id,]
#   theta_mu<-theta_V*( (t(vec_a)%*%t(theta_mean_multiplier))+(1/theta_prior_var)*theta_prior_mu)
#   rnorm(1,theta_mu,sqrt(theta_V))
# }

function theta_sampler(
id,
vec_a, # Ni x 1
vec_b, # Ni x 1
mat_w_id, # Np x 1
mat_k_id, # Np x 1
theta_prior_mu, # 1 X 1
theta_prior_var,# 1 X 1
Ni,
)
    omega_theta = LinearAlgebra.Diagonal(mat_w_id) # Ni x Ni
    theta_variance = 1 / ((vec_a' * omega_theta * vec_a) + (1 / theta_prior_var)) # 1 x 1
    z_theta = vec_a .* vec_b .* mat_w_id + mat_k_id # Ni x 1 # times w_e
    theta_mu = theta_variance * (vec_a' * z_theta) + (theta_prior_mu / theta_prior_var)
    rand(Distributions.Normal(theta_mu, sqrt(theta_variance)))
end

#a_sampler-sampling a for an item(i) across persons: use truncated generation funciton defined above
# a_sampler<-function(id,vec_theta,vec_b,mat_w,mat_k,a_prior_mu,a_prior_var){
#   a_w<-diag(mat_w[,id],Np,Np)
#   a_V<-1/(t(vec_theta-vec_b[id])%*%a_w%*%(vec_theta-vec_b[id])+ (1/a_prior_var))
#   #if(a_V<0){print(vec_theta)}
#   a_mu<-a_V*(t(vec_theta-vec_b[id])%*%mat_k[,id]+ (1/a_prior_var)*a_prior_mu)
#   #print(c(a_mu,a_V))
#   tr_norm_gen(1,a_mu,a_V)
# }

function a_sampler(
    id, 
    vec_theta, # Np x 1
    vec_b_id, # Ni x 1
    mat_w_id, # Np x Ni
    mat_k_id, # Np x Ni
    a_prior_mu, # 1 x 1
    a_prior_var, # 1 x 1
    Np, # 1 x 1
    )
    omega_a = LinearAlgebra.Diagonal(mat_w_id) # Np x Np
    a_variance = 1 / ((((vec_theta .- vec_b_id)' * omega_a) * (vec_theta .- vec_b_id)) + (1 / a_prior_var)) # 1 x 1 
    #z_a = mat_k_id ./ mat_w_id # don't need it
    a_mu = a_variance * ((vec_theta .- vec_b_id)' * mat_k_id + (a_prior_mu / a_prior_var)) # ok
    return tr_norm_gen(1, a_mu, a_variance)[1]
    #return rand(Distributions.TruncatedNormal(a_mu, sqrt(a_variance), 0.0, Inf))
end

#b_sampler-sampling b for an item(i) across persons
# b_sampler<-function(id,vec_theta,vec_a,mat_w,mat_k,b_prior_mu,b_prior_var){
#   b_w<-diag(mat_w[,id],Np,Np)
#   b_V<-1/(t(rep((-vec_a[id]),Np))%*% b_w %*% rep((-vec_a[id]),Np)+ (1/b_prior_var))
#   #if(b_V<0){print(-vec_a[id])}
#   #print(-vec_b[id])}
#   b_mean_multiplier<-mat_k[,id]-vec_a[id]*vec_theta*mat_w[,id]
#   ERROR not transpose vec_a_id
#   b_mu<-b_V*(rep((-vec_a[id]),Np)%*%b_mean_multiplier+(1/b_prior_var)*b_prior_mu)
#   rnorm(1,b_mu,sqrt(b_V))
# }

function b_sampler(id, vec_theta, vec_a_id, mat_w_id, mat_k_id, b_prior_mu, b_prior_var, Np)
    omega_b = LinearAlgebra.Diagonal(mat_w_id) # Np x Np
    vec_a_id_vec = fill(vec_a_id, Np) #Np
    b_variance = 1 / (vec_a_id_vec' * omega_b * vec_a_id_vec + (1 / b_prior_var))  # 1 x 1 ok, vec_a_id_vec can be positive because it is squared anyway
    z_b = mat_k_id .- (vec_a_id .* vec_theta .* mat_w_id) # Np x 1 # times w_i
    b_mu = b_variance * (((.-vec_a_id_vec)' * z_b) + (b_prior_mu / b_prior_var)) # ok
    return rand(Distributions.Normal(b_mu, sqrt(b_variance)))
end

#w_sampler-sampling latent variable w_ip across persons and items
# w_sampler<-function(vec_theta,vec_a,vec_b){
#   w_temp<-matrix(0,Np,Ni)
#   W_gen<-matrix(0,Np,Ni)
#   iter_hist<-matrix(0,Np,Ni)
#   w_sav<-array(rep(0,Np*Ni*2),c(Np,Ni,2))
#   for (p in 1:Np) {
#     for (i in 1:Ni){
#       w_temp[p,i] <- (vec_a[i]*(vec_theta[p]-vec_b[i]))	
#     }
#   }
#   for (p in 1:Np) {
#     for (i in 1:Ni){
#       random.w <- rpg.sp(1,1,w_temp[p,i],track.iter=T)	
#       # deprecated by BayesLogit
#       W_gen[p,i]<-random.w$samp
#       iter_hist[p,i]<-random.w$iter 
#     }
#   }
#   w_sav[,,2]<-iter_hist
#   w_sav[,,1]<-W_gen
#   w_sav
# }	
using RCall
R"library(BayesLogit)"
@rlibrary BayesLogit
function w_sampler(vec_theta, vec_a, vec_b, Np, Ni)
    w_temp = zeros(Float64, Np, Ni)
    W_gen = copy(w_temp)
    w_sav = zeros(Float64, Np, Ni, 2)
    w_temp = [vec_a[i] * (vec_theta[p] - vec_b[i]) for p in 1:Np, i in 1:Ni]
    #w_sav[:,:,1] = [rand(PolyaGamma(1.0, w_temp[p, i]))  for p in 1:Np, i in 1:Ni]
    pg = zeros(Np, Ni)
    for p in 1:Np, i in 1:Ni
        w_temp_pi = w_temp[p,i]
        pg[p,i] = rcopy(R"BayesLogit::rpg.sp(1, 1, $w_temp_pi)")
    end
    w_sav[:,:,1] = copy(pg)
    return w_sav
end

##################################################################################################
###########################     Estimation Starts from here#######################################
##################################################################################################

#Specified by users:Construct 2PL IRT Estimator

Iter = 2000
theta_prior_mu = 0
theta_prior_var= 1
a_prior_mu=1
a_prior_var=5
b_prior_mu=0
b_prior_var=5

sav_a = zeros(Float64, Iter, Ni)
sav_b = zeros(Float64, Iter, Ni)
sav_theta = zeros(Float64, Iter, Np)

sav_a[1,:] = ini_a
sav_b[1,:] = ini_b
sav_theta[1,:] = ini_theta
#sav.iter.hist<-matrix(0,Np,Ni)

# for(iter in 2:Iter){
  
#   sav.W<-w_sampler(sav.theta[iter-1,],sav.a[iter-1,],sav.b[iter-1,])
#   sav.w<-sav.W[,,1]
#   sav.iter.hist<-sav.iter.hist+sav.W[,,2]
#   print(iter) #(id,vec_a,vec_b,mat_w=sav.w,mat_k=K,theta_prior_mu=0,theta_prior_var=1)
#   for (p in 1:Np) {
#     sav.theta[iter,p]<-theta_sampler(p,sav.a[iter-1,],sav.b[iter-1,],sav.w,K,theta_prior_mu,theta_prior_var)
#   }
#   for( i in 1:Ni){
#     sav.a[iter,i]<-a_sampler(i,sav.theta[iter,],sav.b[iter-1,],sav.w,K,a_prior_mu,a_prior_var)
#   }
#   sav.a[sav.a==Inf]<-0.001
#   for( i in 1:Ni){
#     sav.b[iter,i]<-b_sampler(i,sav.theta[iter,],sav.a[iter,],sav.w,K,b_prior_mu,b_prior_var)
#   }
  
# }

for iter in 2:Iter
    sav_W = w_sampler(sav_theta[iter-1, :], sav_a[iter-1, :], sav_b[iter-1, :], Np, Ni)
    sav_w = sav_W[:, :, 1]
    println(iter)
    sav_theta_iter = map(p ->  theta_sampler(p, sav_a[iter-1, :], sav_b[iter-1, :], sav_w[p, :], K[p, :], theta_prior_mu, theta_prior_var, Ni), 1:Np)
    sav_a_iter = map(i -> a_sampler(i, sav_theta[iter, :], sav_b[iter-1, i], sav_w[:, i], K[:, i], a_prior_mu, a_prior_var, Np), 1:Ni)
    sav_a_iter[findall(sav_a_iter .== Inf)] .= 0.001
    sav_b_iter = map(i -> b_sampler(i, sav_theta[iter, :], sav_a[iter, i], sav_w[:, i], K[:, i], b_prior_mu, b_prior_var, Np), 1:Ni)
    sav_theta[iter, :] = copy(sav_theta_iter)
    sav_a[iter, :] = copy(sav_a_iter)
    sav_b[iter, :] = copy(sav_b_iter)
end




