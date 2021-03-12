
function joint_estimate_mmle_quick!(
    items::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{<:AbstractResponse};
    dist::Distributions.DiscreteUnivariateDistribution = Distributions.DiscreteNonParametric([0.0,1.0], [0.5, 0.5]),
    metric::Vector{Float64} = [0.0, 1.0],
    max_time::Int64 = 100,
    max_iter::Int64 = 10,
    x_tol_rel::Float64 = 0.001,
    f_tol_rel::Float64 = 0.0001,
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
    parameters = SharedArrays.SharedMatrix(get_parameters_vals(items))
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
    #update_posterior!(examinees, items, responses; already_sorted = false);
    likelihood = 0
    posterior = Vector{Vector{Float64}}(undef, N)
    Distributed.@sync Distributed.@distributed  for n = 1 : size(examinees,1)
        p = posterior_quick(parameters[i_index[n], :], response_matrix[:, n], X, W) #return a matrix N x K
        normalizer = sum(p)
        if normalizer > typemin(Float64)
            posterior[n] = p ./ normalizer
            likelihood += normalizer 
        else
            posterior[n] = copy(p) 
        end
    end

    #start
    iter = 1
    while !stop
        #calibrate items
        #calibrate_item_mmle!(items, examinees, responses);
        before_time = time()
        old_parameters = copy(parameters)
        Distributed.@sync Distributed.@distributed for i in 1:I
            parameters[i, :] .= calibrate_item_mmle_quick(
                old_parameters[i, :],
                bounds[i],
                posterior[n_index[i]],
                response_matrix[i, :],
                X;
                kwargs...
            )
        end
        Distributed.@sync Distributed.@distributed  for i in 1:I
            parameters_i = copy(parameters[i,:])
            items[i].parameters.a = parameters_i[1]
            items[i].parameters.b = parameters_i[2]
        end

        println("for cal items: ",time()-before_time)
        before_time = time()
        #rescale dist
        rescale!(
            dist,
            examinees; 
            metric = [0.0, 1.0]
        )
        println("for rescale: ",time()-before_time)
        before_time = time()

        X = dist.support
        W = dist.p

        #update posteriors
        likelihood = 0
        posterior = Vector{Vector{Float64}}(undef, N)
        Distributed.@sync Distributed.@distributed for n = 1 : N
            p = posterior_quick(parameters[i_index[n], :], response_matrix[:, n], X, W) #return a matrix N x K
            normalizer = sum(p)
            if normalizer > typemin(Float64)
                posterior[n] = p ./ normalizer
                likelihood += normalizer 
            else
                posterior[n] = copy(p) 
            end
        end
        println("for update post: ",time()-before_time)

        before_time = time()
        
        println("Likelihood: ", likelihood)
        if any([
            check_iter(iter; max_iter = max_iter),
            check_time(start_time; max_time = max_time),
            # check_f_tol_rel!(
            #     examinees,
            #     old_likelihood;
            #     f_tol_rel = f_tol_rel
            #     ),
            check_x_tol_rel!(
                items,
                old_pars;
                x_tol_rel = x_tol_rel
            )]
            )
            stop = true
        end
        println("for conv check: ", time()-before_time)
        before_time = time()
        iter += 1

    end
    map( e -> e.latent.posterior = Distributions.DiscreteNonParametric(dist.support, posterior[n]), examinees);
    map( e -> e.latent.val = e.latent.posterior.p' * e.latent.posterior.support, examinees);
    
    return nothing
end