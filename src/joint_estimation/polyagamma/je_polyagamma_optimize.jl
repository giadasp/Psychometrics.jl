function optimize(je_pg_model::JointEstimationPolyaGammaModel)
    local parameters = je_pg_model.parameters
    local latents = je_pg_model.latents
    local n_index = je_pg_model.n_index
    local i_index = je_pg_model.i_index
    local responses_per_item = je_pg_model.responses_per_item
    local responses_per_examinee = je_pg_model.responses_per_examinee

    I = size(parameters, 1)
    N = size(latents, 1)

    batch_den = max(1,(nprocs()-1))
    batch_size_N = Int(ceil(N/batch_den))
    batch_size_I = Int(ceil(I/batch_den))
    tuple_i_n = vcat([[[i, n] for n in n_index[i]] for i in 1:I]...)

    stop = false
    start_time = time()
    W = SharedMatrix{Float64}(I,N)
    iter = 1
    while !stop
        for n = 1:N
            local latent_n = latents[n]
            for i in i_index[n]
                W[i, n] = _generate_w(latent_n, parameters[i])
            end
        end
        #W_per_item = [W[i, n_index[i]] for i in 1:I]

        #W_per_examinee = [W[i_index[n], n] for n in 1:N]

        # parameters = pmap((p, idx, r, w) -> _mcmc_iter_pg(p, latents[idx], r, w[idx]; sampling = je_pg_model.item_sampling),
        #     parameters,
        #     n_index,
        #     je_pg_model.responses_per_item,
        #     eachrow(W), 
        #     batch_size = batch_size_I,
        # )
        # latents = pmap((l, idx, r, w) ->
        #             _mcmc_iter_pg(l, parameters[idx], r, w[idx]; sampling = je_pg_model.examinee_sampling),
        #     latents,
        #     i_index,
        #     je_pg_model.responses_per_examinee,
        #     eachcol(W), 
        #     batch_size = batch_size_N,
        # )
            for i in 1:I
                if !parameters[i].calibrated
                    parameters[i] = _mcmc_iter_pg(parameters[i], latents[n_index[i]], responses_per_item[i], W[i, n_index[i]]; sampling = je_pg_model.item_sampling)
               end
            end
            for n in 1:N
                if !latents[n].assessed
                   latents[n] = _mcmc_iter_pg(latents[n], parameters[i_index[n]], responses_per_examinee[n], W[i_index[n], n]; sampling = je_pg_model.examinee_sampling)
                end
            end
            if (iter % 200) == 0
                if any([
                    check_iter(iter; max_iter = Int(ext_opt_settings[1]), verbosity = Int(je_pg_model.ext_opt_settings[5])),
                    check_time(start_time; max_time = Int(ext_opt_settings[2]), verbosity = Int(je_pg_model.ext_opt_settings[5])),
                    # check_x_tol_rel!(
                    #     items,
                    #     old_pars;
                    #     x_tol_rel = x_tol_rel
                    # )
                    ])       
                    stop = true
                end
            end
        iter += 1
    end
    return parameters::Vector{<:AbstractParameters}, latents::Vector{<:AbstractLatent}
end
