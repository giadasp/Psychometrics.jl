
function joint_estimate_mmle_2pl_quick!(
    items::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{<:AbstractResponse};
    dist::Distributions.DiscreteUnivariateDistribution = Distributions.DiscreteNonParametric([0.0,1.0], [0.5, 0.5]),
    metric::Vector{Float64} = [0.0, 1.0],
    max_time::Int64 = 100,
    max_iter::Int64 = 10,
    x_tol_rel::Float64 = 0.001,
    f_tol_rel::Float64 = 0.0001,
    int_opt_x_tol_rel::Float64 = 0.001,
    int_opt_max_time::Float64 = 1000.0,
    int_opt_f_tol_rel::Float64 = 0.00001,
    rescale_latent = true,
    kwargs...
    )

    N = size(examinees, 1)
    I = size(items, 1)

    #start points and probs
    probs = Distributions.pdf(Distributions.Normal(metric[1], metric[2]), collect(-6.0:0.3:6.0))
    probs = probs / sum(probs)
    dist = Distributions.DiscreteNonParametric(collect(-6.0:0.3:6.0), probs)
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
    parameters_vectors = [parameters[i,:] for i in 1:I]
    latents = get_latents(examinees)
    bounds = map( i -> [i.parameters.bounds_a, i.parameters.bounds_b], items)

    #extract items idx per examinee and examinees idx per item
    n_index = Vector{Vector{Int64}}(undef, I)
    i_index = Array{Array{Int64,1},1}(undef, N)
    for n = 1: N
        i_index[n] = findall(.!ismissing.(response_matrix[:, n]))
        if n <= I
            n_index[n] = findall(.!ismissing.(response_matrix[n, :]))
        end
    end #15ms

    #update posterior
    likelihood = 0
    posterior = Vector{Vector{Float64}}(undef, N)
    for n = 1 : N
        p = posterior_2pl_quick(parameters_vectors[i_index[n]], response_matrix[i_index[n], n], X, W) 
        normalizer = sum(p)
        if normalizer > typemin(Float64)
            posterior[n] = p ./ normalizer
            likelihood += _log_c(normalizer) 
        else
            posterior[n] = copy(p) 
        end
    end
    old_likelihood = Inf

    #start
    iter = 1

    #nl solver
    opt = NLopt.Opt(:LD_SLSQP, 2)
    opt.xtol_rel = int_opt_x_tol_rel
    opt.maxtime = int_opt_max_time
    opt.ftol_rel = int_opt_f_tol_rel

    while !stop

        Distributed.@sync Distributed.@distributed for i in 1:I
            opt.lower_bounds = [bounds[i][1][1], bounds[i][2][1]]
            opt.upper_bounds = [bounds[i][1][2], bounds[i][2][2]]
            calibrate_item_mmle_2pl_quick!(
                parameters_vectors[i],
                posterior[n_index[i]],
                response_matrix[i, n_index[i]],
                X,
                opt
            )
            parameters[i].a = parameters_vectors[i][1]
            parameters[i].b = parameters_vectors[i][2]
        end

        #rescale dist
        if rescale_latent
            _rescale!(
                dist,
                latents; 
                metric = [0.0, 1.0]
            )
        end

        X = dist.support
        W = dist.p

        #update posteriors
        likelihood = 0
        posterior = Vector{Vector{Float64}}(undef, N)
        Distributed.@sync Distributed.@distributed for n = 1 : N
            p = posterior_2pl_quick(parameters_vectors[i_index[n]], response_matrix[i_index[n], n], X, W) 
            normalizer = sum(p)
            if normalizer > typemin(Float64)
                posterior[n] = p ./ normalizer
                likelihood += _log_c(normalizer)
            else
                posterior[n] = copy(p) 
            end
        end
        
        println("Likelihood: ", likelihood)
        if any([
            check_iter(iter; max_iter = max_iter),
            check_time(start_time; max_time = max_time),
            check_f_tol_rel!(
                likelihood,
                old_likelihood;
                f_tol_rel = f_tol_rel
                ),
            check_x_tol_rel!(
                items,
                old_pars;
                x_tol_rel = x_tol_rel
            )]
            )
            stop = true
        end
        iter += 1
    end
    for n = 1:N
        examinees[n].latent.posterior = Distributions.DiscreteNonParametric(dist.support, posterior[n]);
        examinees[n].latent.val = examinees[n].latent.posterior.p' * examinees[n].latent.posterior.support;
    end

    return nothing
end