function optimize!(je_pg_model::JointEstimationPolyaGammaModel)
    parameters = je_pg_model.parameters
    latents = je_pg_model.latents

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
            for i in 1:I
                parameters_i = parameters[i]
                if !parameters_i.calibrated
                    _mcmc_iter_pg!(parameters_i, latents[je_pg_model.n_index[i]], je_pg_model.responses_per_item[i], W[i, je_pg_model.n_index[i]]; sampling = je_pg_model.item_sampling)
                end
            end
            for n in 1:N
                latent_n = latents[n]
                if !latent_n.assessed
                    _mcmc_iter_pg!(latent_n, parameters[je_pg_model.i_index[n]], je_pg_model.responses_per_examinee[n], W[je_pg_model.i_index[n], n]; sampling = je_pg_model.examinee_sampling)
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
    return nothing
end

function joint_estimate_pg_quick!(
    items::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{<:AbstractResponse};
    max_time::Int64 = 100,
    mcmc_iter::Int64 = 10,
    x_tol_rel::Float64 = 0.001,
    item_sampling::Bool = false,
    examinee_sampling::Bool = false,
    kwargs...
    )
    I = size(items, 1)
    N = size(examinees, 1)

        #from now we work only on these, the algorithm is dependent on the type of latents and parameters.
        response_matrix = get_response_matrix(responses, I, N);
        parameters = map(i -> copy(get_parameters(i)), items)
        latents = map(e -> copy(get_latents(e)), examinees)

    
        #extract items per examinee and examinees per item indices
        n_index = Vector{Vector{Int64}}(undef, I)
        i_index = Array{Array{Int64,1},1}(undef, N)
        for n = 1 : N
            i_index[n] = findall(.!ismissing.(response_matrix[:, n]))
            if n <= I
                n_index[n] = findall(.!ismissing.(response_matrix[n, :]))
            end
        end #15ms
        responses_per_item = [ Vector{Float64}(response_matrix[i, n_index[i]]) for i = 1 : I]
        responses_per_examinee = [ Vector{Float64}(response_matrix[i_index[n], n]) for n = 1 : N]
        #set starting chain
        map(
            p -> begin
                p.chain = [[p.a, p.b] for j = 1:1000]
            end,
            parameters,
        );

    je_pg_model =  JointEstimationPolyaGammaModel(
        parameters,
        latents,
        responses_per_item,
        responses_per_examinee,
        n_index,
        i_index,
        mcmc_iter,
        max_time,
        item_sampling,
        examinee_sampling
        ) 
    optimize!(je_pg_model)
    Distributed.@sync Distributed.@distributed for n in 1 : N
        e = examinees[n]
        l = je_pg_model.latents[n]
        examinees[n] = Examinee(e.idx, e.id, l)
        if n<=I
            i = items[n]
            p = je_pg_model.parameters[n]
            items[n] = Item(i.idx, i.id, p)
        end
    end
    map(i -> update_estimate!(i), items);
    map(e -> update_estimate!(e), examinees);
    return nothing
end