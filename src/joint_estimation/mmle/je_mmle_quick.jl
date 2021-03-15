
function joint_estimate_mmle_2pl_quick!(
    items::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{<:AbstractResponse};
    dist::Distributions.DiscreteUnivariateDistribution = Distributions.DiscreteNonParametric([0.0,1.0], [0.5, 0.5]),
    metric::Vector{Float64} = [0.0, 1.0],
    max_time::Int64 = 100,
    max_iter::Int64 = 10,
    x_tol_rel::Float64 = 0.00001,
    f_tol_rel::Float64 = 0.0001,
    int_opt_x_tol_rel::Float64 = 0.0001,
    int_opt_max_time::Float64 = 1000.0,
    int_opt_f_tol_rel::Float64 = 0.00001,
    rescale_latent = true,
    kwargs...
    )

    N = size(examinees, 1)
    I = size(items, 1)

    #start points and probs
    X = dist.support
    W = dist.p
    
    # starting values
    before_time = time()
    stop = false
    old_likelihood = 0.0
    old_pars = get_parameters_vals(items)
    start_time = time()
    response_matrix = get_response_matrix(responses, size(items,1), size(examinees,1));
    parameters = get_parameters(items)
    parameters_matrix = get_parameters_vals(items)
    parameters_vectors = [parameters_matrix[i,:] for i in 1:I]
    latents = get_latents(examinees)

    #latents = get_latents(examinees)
    bounds = map( i -> [i.parameters.bounds_a, i.parameters.bounds_b], items)

    #extract items idx per examinee and examinees idx per item
    n_index = Vector{Vector{Int64}}(undef, I)
    i_index = Array{Array{Int64,1},1}(undef, N)
    for n = 1: N
        i_index[n] = findall(.!ismissing.(response_matrix[:, n]))
        if n <= I
            n_index[n] = findall(.!ismissing.(response_matrix[n, :]))
        end
    end 
    responses_i = [ Vector{Float64}(response_matrix[i, n_index[i]]) for i = 1 : I]
    responses_n = [ Vector{Float64}(response_matrix[i_index[n], n]) for n = 1 : N]

    #update posterior
    likelihood = 0
    old_likelihood = Inf

    posterior = Vector{Vector{Float64}}(undef, N)
    likelihood = posterior_2pl_quick!(posterior, i_index, parameters_vectors, responses_n, X, W)

    #start
    iter = 1

    
    while !stop
        calibrate_item_mmle_2pl_quick!(
            parameters_vectors,
            n_index,
            bounds,
            posterior,
            responses_i,
            X;
            int_opt_x_tol_rel = int_opt_x_tol_rel,
            int_opt_max_time = int_opt_max_time,
            int_opt_f_tol_rel = int_opt_f_tol_rel,
        )
        #rescale dist
        if rescale_latent
            _rescale!(
                dist,
                latents; 
                metric = metric
            )
        end

        X = dist.support
        W = dist.p

        #update posteriors
        likelihood = posterior_2pl_quick!(posterior, i_index, parameters_vectors, responses_n, X, W)

        if any([
            check_iter(iter; max_iter = max_iter),
            check_time(start_time; max_time = max_time),
            check_f_tol_rel!(
                likelihood,
                old_likelihood;
                f_tol_rel = f_tol_rel
                )
            ])
            stop = true
        end
        iter += 1
    end
    for n = 1:N
        examinees[n].latent.posterior = Distributions.DiscreteNonParametric(dist.support, posterior[n]);
        examinees[n].latent.val = examinees[n].latent.posterior.p' * examinees[n].latent.posterior.support;
        if n <= I
            items[n].parameters.a = parameters_vectors[n][1]
            items[n].parameters.b = parameters_vectors[n][2]
        end
    end

    return nothing
end