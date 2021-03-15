include("je_mmle_struct.jl")
include("je_mmle_matrix.jl")
include("je_mmle_quick.jl")
#too slow!
# function joint_estimate_mmle!(
#     items::Vector{<:AbstractItem},
#     examinees::Vector{<:AbstractExaminee},
#     responses::Vector{<:AbstractResponse};
#     dist::Distributions.DiscreteUnivariateDistribution = Distributions.DiscreteNonParametric([0.0,1.0], [0.5, 0.5]),
#     metric::Vector{Float64} = [0.0, 1.0],
#     max_time::Int64 = 100,
#     max_iter::Int64 = 10,
#     x_tol_rel::Float64 = 0.001,
#     f_tol_rel::Float64 = 0.00001,
#     kwargs...
#     )
#     #start points and probs
#     probs = Distributions.pdf(Distributions.Normal(metric[1], metric[2]), collect(-6.0:0.3:6.0))
#     probs = probs / sum(probs)
#     dist = Distributions.DiscreteNonParametric(collect(-6.0:0.3:6.0), probs)

#     #set priors 
#     set_prior!(examinees, dist);

#     #update posteriors
#     update_posterior!(examinees, items, responses; already_sorted = false);

#     #gr()
#     # starting values
#     before_time = time()
#     stop = false
#     old_likelihood = 0.0
#     old_pars = get_parameters_vals(items)
#     start_time = time()
#     response_matrix = get_response_matrix(responses, size(items,1), size(examinees,1));

#     #extract items per examinee and examinees per item indices
#     n_index = Vector{Vector{Int64}}(undef, size(items, 1))
#     i_index = Array{Array{Int64,1},1}(undef, size(examinees, 1))
#     for n = 1: size(examinees,1)
#         i_index[n] = findall(.!ismissing.(response_matrix[:, n]))
#         if n <= size(items, 1)
#             n_index[n] = findall(.!ismissing.(response_matrix[n, :]))
#         end
#     end #15ms
#     iter = 1
#     while !stop
#         #calibrate items
#         #calibrate_item_mmle!(items, examinees, responses);
#         Distributed.@sync Distributed.@distributed for i in 1 : size(items, 1)
#             calibrate_item_mmle!(items[i], examinees, responses);
#         end
#         #calibrate_item_mmle!(items, examinees, response_matrix);
#         #rescale dist
#         rescale!(
#             dist,
#             examinees; 
#             metric = [0.0, 1.0]
#         )
#         #println("dist")
#         #display(plot(dist.support, dist.p))
#         #update examinees' support
#         map( e -> e.latent.prior = dist, examinees)

#         #update posteriors
#         #update_posterior!(examinees, items, response_matrix; already_sorted = false);
#         Distributed.@sync Distributed.@distributed for n in 1 : size(examinees, 1)
#             update_posterior!(examinees[n], items, responses);
#         end
#         if any([
#             check_iter(iter; max_iter = max_iter),
#             check_time(start_time; max_time = max_time),
#             check_f_tol_rel!(
#                 examinees,
#                 old_likelihood;
#                 f_tol_rel = f_tol_rel
#                 ),
#             check_x_tol_rel!(
#                 items,
#                 old_pars;
#                 x_tol_rel = x_tol_rel
#             )]
#             )
#             stop = true
#         end
#         iter += 1

#     end
#     map( e -> e.latent.posterior = Distributions.DiscreteNonParametric(dist.support, e.latent.posterior.p), examinees);
#     map( e -> e.latent.val = e.latent.posterior.p' * e.latent.posterior.support, examinees);
    
#     return nothing
# end