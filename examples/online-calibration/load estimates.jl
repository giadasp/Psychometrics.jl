using Pkg
Pkg.instantiate()
Pkg.activate(".")
using Psychometrics
using Distributions
using LinearAlgebra
using Dates
using Random
using SharedArrays
using JLD2

I = 1020
I_to_calibrate = 20
N = 2_000
test_length = 35
field_items = 5
oper_items  = test_length - field_items
iter_mcmc_latent = 2_000
iter_mcmc_item = 4_000
#after how many responses update item parameter estimates
batch_size = 5
N_T = 200

@load "test/simulation/batch5    NT200 Dgain/true values/true_items_1.jld2" items
@load "test/simulation/batch5    NT200 Dgain/true values/true_examinees_1.jld2" examinees
a_s = zeros(1020,100)
b_s = zeros(1020,100)
for rep = 1:100
    @load string("test/simulation/batch5    NT200 Dgain/rep_", rep, "_items_est.jld2") items_est
    a = map(i -> i.parameters.a, items_est)
    a_s[:, rep] = copy(a)
    b = map(i -> i.parameters.b, items_est)
    b_s[:, rep] = copy(b)
end

a_true = map(i -> i.parameters.a, items)
a_true = hcat([a_true for rep=1:100]...)
b_true = map(i -> i.parameters.b, items)
b_true = hcat([b_true for rep=1:100]...)
diff_a = (a_s[1001:1020,:] - a_true[1001:1020,:]).^2
diff_b = (b_s[1001:1020,:] - b_true[1001:1020,:]).^2

println(sqrt(sum(diff_a)/20/100))
println(sqrt(sum(diff_b)/20/100))

#examinees


@load "test/simulation/batch5    NT200 Dgain/true values/true_items_1.jld2" items
@load "test/simulation/batch5    NT200 Dgain/true values/true_examinees_1.jld2" examinees
theta_s = zeros(2000,100)
for rep = 1:100
    @load string("test/simulation/batch5    NT200 Dgain/examinees_est_rep_", rep, ".jld2") examinees_est
    t = map(e -> e.latent.val, examinees_est)
    theta_s[:, rep] = copy(t)
end

t_true = map(e -> e.latent.val, examinees)
t_true = hcat([t_true for rep=1:100]...)
diff_t = (theta_s - t_true).^2

println(sqrt(sum(diff_t[1:800])/800/100))


@load "test/simulation/batch5    NT200 Dgain/examinees_est_theta_rep_1.jld2" examinees_est_theta

rmses=Matrix{Float64}(undef, test_length, N)
for n=1:N
    rmses[:,n] = sqrt.(((examinees_est_theta[n].-examinees[n].latent.val)).^2)
end

plot(1:test_length, rmses[:,1:20], legend=false)

rmses_k=Matrix{Float64}(undef, 35, size(collect(-3.0:1:3.0),1))
thetas = map(e -> e.latent.val, examinees)[1:700]
thetas_k = collect(-3.0:1:3.0)
for k=2:size(thetas_k, 1)
    rmses_k[:,k] = mean(rmses[:, findall((thetas .< thetas_k[k]) .& (thetas .>= thetas_k[max(1,k-1)]))], dims=2)[:,1]
end
