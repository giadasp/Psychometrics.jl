function optimize(je_mmle_model::JointEstimationMMLEModel)
    local parameters = je_mmle_model.parameters
    local latents = je_mmle_model.latents
    local dist = je_mmle_model.dist
    local n_index = je_mmle_model.n_index
    local i_index = je_mmle_model.i_index
    local responses_per_item = je_mmle_model.responses_per_item
    local responses_per_examinee = je_mmle_model.responses_per_examinee


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

    latents = pmap(
        (l, idx, r) -> 
            l.assessed || _update_posterior(l, parameters[idx], r),
        latents,
        i_index,
        responses_per_examinee,
        batch_size = batch_size_N,
        distributed = true
    )
    # Threads.@threads for n in 1:N
    #     latents[n] = latents[n].assessed || _update_posterior(latents[n], parameters[i_index[n]], responses_per_examinee[n])
    # end
    # latents = @sync @distributed (vcat) for n in 1:N
    #     if latents[n].assessed 
    #         latents[n]
    #     else
    #         _update_posterior(latents[n], parameters[i_index[n]], responses_per_examinee[n])
    #     end
    # end
    while !stop
        # calibrate items

        # parameters = @sync @distributed (vcat) for i in 1:I
        #     if parameters[i].calibrated
        #         parameters[i]
        #     else
        #         m_step(parameters[i], latents[n_index[i]], responses_per_item[i], je_mmle_model.int_opt_settings)
        #     end
        # end
        # Threads.@threads for i in 1:I
        #     parameters[i] = parameters[i].calibrated || _calibrate_item_mmle(parameters[i], latents[n_index[i]], responses_per_item[i], je_mmle_model.int_opt_settings)
        # end
        parameters = pmap(
             (p, idx, r) ->
                 p.calibrated || m_step(p, latents[idx], r, je_mmle_model.int_opt_settings),
             parameters,
             n_index,
             responses_per_item,
             batch_size = batch_size_I,
             distributed = true
        )

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
 
        # latents = @sync @distributed (vcat) for n in 1:N
        #     if latents[n].assessed 
        #         latents[n]
        #     else
        #         _update_posterior(latents[n], parameters[i_index[n]], responses_per_examinee[n])
        #     end
        # end
        # Threads.@threads for n in 1:N
        #     latents[n] = latents[n].assessed || _update_posterior(latents[n], parameters[i_index[n]], responses_per_examinee[n])
        # end

        latents = pmap(
            (l, idx, r) -> 
                l.assessed || _update_posterior(l, parameters[idx], r),
            latents,
            i_index,
            responses_per_examinee,
            batch_size = batch_size_N,
            distributed = true
        )
        if any([
            check_iter(iter; max_iter = Int64(je_mmle_model.ext_opt_settings[1]), verbosity = Int(je_mmle_model.ext_opt_settings[5])),
            check_time(start_time; max_time = Int64(je_mmle_model.ext_opt_settings[2]), verbosity = Int(je_mmle_model.ext_opt_settings[5])),
            check_f_tol_rel!(
                latents,
                old_likelihood;
                f_tol_rel = je_mmle_model.ext_opt_settings[3],
                verbosity = Int(je_mmle_model.ext_opt_settings[5])
                ),
            check_x_tol_rel!(
                parameters,
                old_pars;
                x_tol_rel = je_mmle_model.ext_opt_settings[4],
                verbosity = Int(je_mmle_model.ext_opt_settings[5])
            )]
            )
            stop = true
        end
        iter += 1
    end
    
    return parameters::Vector{<:AbstractParameters}, latents::Vector{<:AbstractLatent}, dist::Distributions.DiscreteUnivariateDistribution
end
