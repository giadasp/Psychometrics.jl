
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

    stop = false
    old_pars = get_parameters_vals(items)
    start_time = time()
    iter = 1

        #from now we work only on these, the algorithm is dependent on the type of latents and parameters.
        response_matrix = get_response_matrix(responses, size(items,1), size(examinees,1));
        parameters = get_parameters(items)
        latents = get_latents(examinees)

    
        #extract items per examinee and examinees per item indices
        n_index = Vector{Vector{Int64}}(undef, I)
        i_index = Array{Array{Int64,1},1}(undef, N)
        for n = 1 : N
            i_index[n] = findall(.!ismissing.(response_matrix[:, n]))
            if n <= I
                n_index[n] = findall(.!ismissing.(response_matrix[n, :]))
            end
        end #15ms
        responses_i = [ Vector{Float64}(response_matrix[i, n_index[i]]) for i = 1 : I]
        responses_n = [ Vector{Float64}(response_matrix[i_index[n], n]) for n = 1 : N]
        #set starting chain
        map(
            p -> begin
                p.chain = [[p.a, p.b] for j = 1:1000]
            end,
            parameters,
        );
    W = zeros(Float64, I, N)
    while !stop
        for i in 1:I, n in n_index[i]
            W[i, n] =  _generate_w(latents[n], parameters[i])
        end
            Distributed.@sync Distributed.@distributed for i in 1:I
                parameters_i = parameters[i]
                if !parameters_i.calibrated
                    _mcmc_iter_pg!(parameters_i, latents[n_index[i]], response_i[i], W[i, n_index[i]]; sampling = item_sampling)
                end
            end
            Distributed.@sync Distributed.@distributed for n in 1:N
                latent_n = latents[n]
                if !latent_n.assessed
                    _mcmc_iter_pg!(latent_n, parameters[i_index[n]], responses_nx[n], W[i_index[n], n]; sampling = examinee_sampling)
                end
            end
            if (iter % 200) == 0
                #map(i -> update_estimate!(i), items);
                if any([
                    check_iter(iter; max_iter = mcmc_iter),
                    check_time(start_time; max_time = max_time),
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
    map(i -> update_estimate!(i), items);
    map(e -> update_estimate!(e), examinees);

    return nothing
end