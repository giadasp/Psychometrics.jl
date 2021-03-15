function optimize!(je_mmle_model::JointEstimationMMLEModel)
    parameters = je_mmle_model.parameters
    latents = je_mmle_model.latents

    I = size(parameters, 1)
    N = size(latents, 1)

    batch_size_N = Int(ceil(N/nprocs()))
    batch_size_I = Int(ceil(I/nprocs()))

    iter = 1

    # starting values
    before_time = time()
    stop = false
    old_likelihood = 0.0
    old_pars = hcat(_get_parameters_vals.(parameters)...)
    start_time = time()

    latents = pmap(
        (l, idx, r) -> 
            l.assessed || _update_posterior(l, parameters[idx], r),
        latents,
        je_mmle_model.i_index,
        je_mmle_model.responses_per_examinee,
        batch_size = batch_size_N
    )
    # Threads.@threads for n in 1:N
    #     latents[n] = latents[n].assessed || _update_posterior(latents[n], parameters[je_mmle_model.i_index[n]], je_mmle_model.responses_per_examinee[n])
    # end
    # Distributed.@sync Distributed.@distributed for n in 1:N
    #     latents[n] = latents[n].assessed || _update_posterior(latents[n], parameters[je_mmle_model.i_index[n]], je_mmle_model.responses_per_examinee[n])
    # end
    while !stop
        # calibrate items
        # Distributed.@sync Distributed.@distributed for i in 1:I
        #    parameters[i] = parameters[i].calibrated || _calibrate_item_mmle(parameters[i], latents[je_mmle_model.n_index[i]], je_mmle_model.responses_per_item[i], je_mmle_model.int_opt_settings)
        # end
        # Threads.@threads for i in 1:I
        #     parameters[i] = parameters[i].calibrated || _calibrate_item_mmle(parameters[i], latents[je_mmle_model.n_index[i]], je_mmle_model.responses_per_item[i], je_mmle_model.int_opt_settings)
        # end
        parameters = pmap(
             (p, idx, r) ->
                 p.calibrated || _calibrate_item_mmle(p, latents[idx], r, je_mmle_model.int_opt_settings),
             parameters,
             je_mmle_model.n_index,
             je_mmle_model.responses_per_item,
             batch_size = batch_size_I,
        )

        #rescale dist
        if je_mmle_model.rescale_latent
            _rescale!(
                je_mmle_model.dist,
                latents;
                metric = je_mmle_model.metric
            )
            for n in 1 : N
                latents[n].prior = je_mmle_model.dist
            end
        end

        # #update posteriors
        # Distributed.@sync Distributed.@distributed for n in 1:N
        #    latents[n] = latents[n].assessed || _update_posterior(latents[n], parameters[je_mmle_model.i_index[n]], je_mmle_model.responses_per_examinee[n])
        # end

        # Threads.@threads for n in 1:N
        #     latents[n] = latents[n].assessed || _update_posterior(latents[n], parameters[je_mmle_model.i_index[n]], je_mmle_model.responses_per_examinee[n])
        # end

        latents = pmap(
            (l, idx, r) -> 
                l.assessed || _update_posterior(l, parameters[idx], r),
            latents,
            je_mmle_model.i_index,
            je_mmle_model.responses_per_examinee,
            batch_size = batch_size_N
        )
        if any([
            check_iter(iter; max_iter = Int64(je_mmle_model.ext_opt_settings[1])),
            check_time(start_time; max_time = Int64(je_mmle_model.ext_opt_settings[2])),
            check_f_tol_rel!(
                latents,
                old_likelihood;
                f_tol_rel = je_mmle_model.ext_opt_settings[3]
                ),
            check_x_tol_rel!(
                parameters,
                old_pars;
                x_tol_rel = je_mmle_model.ext_opt_settings[4]
            )]
            )
            stop = true
        end
        iter += 1
    end
    
    return nothing
end

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
    int_opt_x_tol_rel::Float64 = 0.0001,
    int_opt_f_tol_rel::Float64 = 0.000001,
    rescale_latent::Bool = true,
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
    optimize!(je_mmle_model)

    for n in 1 : N
        l = copy(je_mmle_model.latents[n])
        l.prior = dist
        l.posterior = Distributions.DiscreteNonParametric(dist.support, l.posterior.p)
        l.val = l.posterior.p'*l.posterior.support
        examinees[n] = Examinee(examinees[n].idx, examinees[n].id, l)
        if n<=I
            p = copy(je_mmle_model.parameters[n])
            items[n] = Item(items[n].idx, items[n].id, p)
        end
    end
    return nothing
end
