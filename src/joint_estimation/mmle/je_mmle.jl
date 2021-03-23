include("je_mmle_struct.jl")
include("je_mmle_optimize.jl")
include("je_mmle_optimize_2pl_1d_super_fast.jl")


function joint_estimate_mmle!(
    items::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{<:AbstractResponse};
    dist::Distributions.DiscreteUnivariateDistribution = Distributions.DiscreteNonParametric([0.0,1.0], [0.5, 0.5]),
    metric::Vector{Float64} = [0.0, 1.0],
    max_iter::Int64 = 10,
    max_time::Int64 = 100,
    x_tol_rel::Float64 = 0.0001,
    f_tol_rel::Float64 = 0.000001,
    int_opt_max_time::Float64 = 100.0,
    int_opt_x_tol_rel::Float64 = 0.00001,
    int_opt_f_tol_rel::Float64 = 0.0000001,
    rescale_latent::Bool = true,
    super_fast_2pl_1d = false,
    kwargs...
    )
    I = size(items, 1)
    N = size(examinees, 1)

    #set priors 
    set_prior!(examinees, dist);

    #from now we work only on these, the algorithm is dependent on the type of latents and parameters.
    response_matrix = get_response_matrix(responses, size(items,1), size(examinees,1));
    parameters = map( i -> copy(get_parameters(i)), items)
    latents = map( e -> copy(get_latents(e)), examinees)

    #extract items per examinee and examinees per item indices
    n_index = Vector{Vector{Int64}}(undef, I)
    i_index = Array{Array{Int64,1},1}(undef, N)
    for n = 1 : N
        i_index[n] = findall(.!ismissing.(response_matrix[:, n]))
        if n <= I
            n_index[n] = findall(.!ismissing.(response_matrix[n, :]))
        end
    end 

    responses_per_item = [Vector{Float64}(response_matrix[i, n_index[i]]) for i = 1 : I]
    responses_per_examinee = [Vector{Float64}(response_matrix[i_index[n], n]) for n = 1 : N]

    je_mmle_model = JointEstimationMMLEModel(
        parameters,
        latents,
        responses_per_item,
        responses_per_examinee,
        n_index,
        i_index,
        copy(dist),
        metric,
        rescale_latent,
        [Float64(max_iter), Float64(max_time), x_tol_rel, f_tol_rel],
        [int_opt_max_time, int_opt_x_tol_rel, int_opt_f_tol_rel]
    )

    if !super_fast_2pl_1d
        parameters, latents, dist = optimize(je_mmle_model)
    else
        parameters, latents, dist = optimize_2pl_1d_super_fast(je_mmle_model)
    end

    for n in 1 : N
        l = copy(latents[n])
        l.prior = dist
        l.posterior = Distributions.DiscreteNonParametric(dist.support, l.posterior.p)
        l.val = l.posterior.p'*l.posterior.support
        examinees[n] = Examinee(examinees[n].idx, examinees[n].id, l)
        if n<=I
            p = copy(parameters[n])
            items[n] = Item(items[n].idx, items[n].id, p)
        end
    end
    return nothing
end
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