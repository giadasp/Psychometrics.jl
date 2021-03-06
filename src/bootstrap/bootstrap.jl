include("sampling.jl")
function bootstrap!(
    items::Vector{Item},
    examinees::Vector{Examinee},
    responses::Vector{Response};
    method = "mmle", #"polyagamma"
    quick = false,
    replications = 100,
    type = "parametric",
    sample_fraction = 0.9,
    design = "incomplete", #"complete"
    kwargs...
)

    starting_pars = get_parameters_vals(items)
    I = size(starting_pars, 1)
    n_pars = size(starting_pars, 2)

    starting_latents = get_latents_vals(examinees)
    N = size(starting_latents, 2)
    n_latents = size(starting_latents, 1)

    #NB: save samples in chain
    if method == "mmle"
        #discrete ability distribution
        dist = examinees[1].latent.prior
        W = dist.p
        X = dist.support
        K = size(W, 1)

        #takes only the first Latent
        (bins, nothing) = cutR(
            starting_latents[1, :];
            start = minimum(X),
            stop = maximum(X),
            n_bins = K,
            return_breaks = false,
            return_mid_points = true,
        )
        for r = 1:replications
            println("Replication: ", r)
            if type == "parametric"
                n_sample = parametric_sample(N, sample_fraction, bins, dist)
                #loop until all items have at least a response
                examinees_sampled = examinees[n_sample]
                responses_sampled = vcat(map( e -> get_responses_by_examinee_id(e.id, responses), examinees_sampled)...)
                n_responses_sampled_per_item = map( i -> size(get_responses_by_item_id(i.id, responses_sampled), 1), items)
                while minimum(n_responses_sampled_per_item) < I
                    n_sample = parametric_sample(N, sample_fraction, bins, dist)
                end
                println("Number of examinees sample in replication ",r," : ", size(n_sample, 1))
            elseif type == "nonparametric"
                n_sample = non_parametric_sample(N, sample_fraction)
                #loop until all items have at least a response
                examinees_sampled = examinees[n_sample]
                responses_sampled = vcat(map( e -> get_responses_by_examinee_id(e.id, responses), examinees_sampled)...)
                n_responses_sampled_per_item = map( i -> size(get_responses_by_item_id(i.id, responses_sampled), 1), items)
                while minimum(n_responses_sampled_per_item) < I
                    n_sample = non_parametric_sample(N, sample_fraction)
                end
            else
                error("Bootstrap type can be \"parametric\" or \"nonparametric\".")
            end
            #must change examinees idx and examinee_idx ow get_response_matrix does not work well
            examinees_r = Examinee[]
            responses_r = Response[]
            for n = 1 : size(n_sample, 1) 
                n_main = n_sample[n]
                examinee_main = examinees[n_main]
                examinee = Examinee(n, examinee_main.id, examinee_main.latent)
                examinee.latent.assessed =  false
                push!(examinees_r, examinee)
                responses_e = get_responses_by_examinee_id(examinee_main.id, responses)
                for r in responses_e
                    push!(responses_r, Response(r.item_idx, n, r.item_id, r.examinee_id, r.val, r.start_time, r.end_time))
                end
            end

            items_r = map(i -> i, items)
            n_not_sampled = setdiff(collect(1:N), n_sample)
            if quick
                if items[1].parameters isa Parameters2PL
                    joint_estimate_mmle_2pl_quick!(items_r, examinees_r, responses_r; kwargs...)
                else
                    error("Only 2pl items are supported in quick mode")
                end
            else
                joint_estimate_mmle!(items_r, examinees_r, responses_r; kwargs...)
            end
            map( ( i_r , i ) -> push!(i.parameters.chain, get_parameters_vals(i_r)), items_r, items)
        end    
        elseif method == "polyagamma"
        
        if type == "parametric"

        elseif type == "nonparametric"
            
        else
            error("Bootstrap type can be \"parametric\" or \"nonparametric\".")
        end
    else
        error("Estimation method can be \"mmle\" or \"polyagamma\".")
    end
    return items, examinees
end
