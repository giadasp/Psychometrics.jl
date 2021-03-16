function optimize(je_pg_model::JointEstimationPolyaGammaModel)
    parameters = map(p -> copy(p), je_pg_model.parameters)
    latents =  map(p -> copy(p), je_pg_model.latents)

    I = size(parameters, 1)
    N = size(latents, 1)

    stop = false
    start_time = time()
    W = zeros(Float64, I, N)
    iter = 1
    while !stop
        for i in 1:I, n in je_pg_model.n_index[i]
            W[i, n] =  _generate_w(latents[n], parameters[i])
        end
            @sync @distributed for i in 1:I
                if !inparameters[i].calibrated 
                    parameters[i] = _mcmc_iter_pg(parameters[i], latents[je_pg_model.n_index[i]], je_pg_model.responses_per_item[i], W[i, je_pg_model.n_index[i]]; sampling = je_pg_model.item_sampling)
                end
            end
            @sync @distributed for n in 1:N
                if !latents[n].assessed
                    latents[n] = _mcmc_iter_pg(latents[n], parameters[je_pg_model.i_index[n]], je_pg_model.responses_per_examinee[n], W[je_pg_model.i_index[n], n]; sampling = je_pg_model.examinee_sampling)
                end
            end
            if (iter % 200) == 0
                if any([
                    check_iter(iter; max_iter = je_pg_model.mcmc_iterations),
                    check_time(start_time; max_time = je_pg_model.max_time),
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
    return parameters, latents
end
