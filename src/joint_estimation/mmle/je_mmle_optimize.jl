function optimize(je_mmle_model::JointEstimationMMLEModel)
    parameters = je_mmle_model.parameters
    latents = je_mmle_model.latents
    dist = je_mmle_model.dist

    I = size(parameters, 1)
    N = size(latents, 1)

    batch_den = max(1,(nprocs()-1))
    batch_size_N = Int(ceil(N/batch_den))
    batch_size_I = Int(ceil(I/batch_den))

    iter = 1

    # starting values
    before_time = time()
    stop = false
    old_likelihood = 0.0
    old_pars = hcat(_get_parameters_vals.(parameters)...)
    start_time = time()

    # latents = pmap(
    #     (l, idx, r) -> 
    #         l.assessed || _update_posterior(l, parameters[idx], r),
    #     latents,
    #     je_mmle_model.i_index,
    #     je_mmle_model.responses_per_examinee,
    #     batch_size = batch_size_N,
    #     distributed = true
    # )
    # Threads.@threads for n in 1:N
    #     latents[n] = latents[n].assessed || _update_posterior(latents[n], parameters[je_mmle_model.i_index[n]], je_mmle_model.responses_per_examinee[n])
    # end
    Distributed.@sync Distributed.@distributed for n in 1:N
        latents[n] = latents[n].assessed || _update_posterior(latents[n], parameters[je_mmle_model.i_index[n]], je_mmle_model.responses_per_examinee[n])
    end
    while !stop
        # calibrate items
        Distributed.@sync Distributed.@distributed for i in 1:I
           parameters[i] = parameters[i].calibrated || _calibrate_item_mmle(parameters[i], latents[je_mmle_model.n_index[i]], je_mmle_model.responses_per_item[i], je_mmle_model.int_opt_settings)
        end
        # Threads.@threads for i in 1:I
        #     parameters[i] = parameters[i].calibrated || _calibrate_item_mmle(parameters[i], latents[je_mmle_model.n_index[i]], je_mmle_model.responses_per_item[i], je_mmle_model.int_opt_settings)
        # end
        # parameters = pmap(
        #      (p, idx, r) ->
        #          p.calibrated || m_step(p, latents[idx], r, je_mmle_model.int_opt_settings),
        #      parameters,
        #      je_mmle_model.n_index,
        #      je_mmle_model.responses_per_item,
        #      batch_size = batch_size_I,
        #      distributed = true
        # )

        #rescale dist
        if je_mmle_model.rescale_latent
            dist = _rescale(
                dist,
                latents;
                metric = je_mmle_model.metric
            )
            for n = 1:N
                latents[n].prior = dist
            end
        end

        # #update posteriors
        Distributed.@sync Distributed.@distributed for n in 1:N
           latents[n] = latents[n].assessed || _update_posterior(latents[n], parameters[je_mmle_model.i_index[n]], je_mmle_model.responses_per_examinee[n])
        end

        # Threads.@threads for n in 1:N
        #     latents[n] = latents[n].assessed || _update_posterior(latents[n], parameters[je_mmle_model.i_index[n]], je_mmle_model.responses_per_examinee[n])
        # end

        # latents = pmap(
        #     (l, idx, r) -> 
        #         l.assessed || _update_posterior(l, parameters[idx], r),
        #     latents,
        #     je_mmle_model.i_index,
        #     je_mmle_model.responses_per_examinee,
        #     batch_size = batch_size_N,
        #     distributed = true
        # )
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
    
    return parameters::Vector{<:AbstractParameters}, latents::Vector{<:AbstractLatent}, dist::Distributions.DiscreteUnivariateDistribution
end
