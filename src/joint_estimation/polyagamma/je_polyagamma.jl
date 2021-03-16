include("je_polyagamma_struct.jl")
include("je_polyagamma_optimize.jl")

function joint_estimate_pg!(
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
    parameters, latents = optimize(je_pg_model)
    @sync @distributed for n in 1 : N
        e = examinees[n]
        l = latents[n]
        examinees[n] = Examinee(e.idx, e.id, l)
        if n<=I
            i = items[n]
            p = parameters[n]
            items[n] = Item(i.idx, i.id, p)
        end
    end
    map(i -> update_estimate!(i), items);
    map(e -> update_estimate!(e), examinees);
    return nothing
end
# function joint_estimate_pg!(
#     items::Vector{<:AbstractItem},
#     examinees::Vector{<:AbstractExaminee},
#     responses::Vector{<:AbstractResponse};
#     max_time::Int64 = 100,
#     mcmc_iter::Int64 = 10,
#     x_tol_rel::Float64 = 0.001,
#     item_sampling::Bool = false,
#     examinee_sampling::Bool = false,
#     kwargs...
#     )
    
#     responses_per_examinee = map(
#         e -> sort(get_responses_by_examinee_id(e.id, responses), by = x -> x.item_idx),
#         examinees,
#     );
#     items_idx_per_examinee = map(
#         e -> sort(map(r -> items[r.item_idx].idx, responses_per_examinee[e.idx])),
#         examinees,
#     );
#     responses_per_item = map(
#         i -> sort(get_responses_by_item_id(i.id, responses), by = x -> x.examinee_idx),
#         items,
#     );
#     examinees_idx_per_item = map(
#         i -> sort(map(r -> examinees[r.examinee_idx].idx, responses_per_item[i.idx])),
#         items,
#     );

#     map(
#         i -> begin
#             i.parameters.chain = [[i.parameters.a, i.parameters.b] for j = 1:1000]
#         end,
#         items,
#     );

#     stop = false
#     old_pars = get_parameters_vals(items)
#     start_time = time()
#     iter = 1

#     while !stop
#             W = generate_w(
#                 items,
#                 map(i -> examinees[examinees_idx_per_item[i.idx]], items),
#             )
#             map(
#                 i -> mcmc_iter_pg!(
#                     i,
#                     examinees[examinees_idx_per_item[i.idx]],
#                     responses_per_item[i.idx],
#                     filter(w -> w.i_idx == i.idx, W);
#                     sampling = item_sampling,
#                     already_sorted = true,
#                 ),
#                 items,
#             )
#             map(
#                 e -> mcmc_iter_pg!(
#                     e,
#                     items[items_idx_per_examinee[e.idx]],
#                     responses_per_examinee[e.idx],
#                     filter(w -> w.e_idx == e.idx, W);
#                     sampling = examinee_sampling,
#                     already_sorted = true,
#                 ),
#                 examinees,
#             )
#             if (iter % 200) == 0
#                 #map(i -> update_estimate!(i), items);
#                 if any([
#                     check_iter(iter; max_iter = mcmc_iter),
#                     check_time(start_time; max_time = max_time)
#                     #check_x_tol_rel!(
#                     #     items,
#                     #     old_pars;
#                     #     x_tol_rel = x_tol_rel)
#                     ])
                    
#                     stop = true
#                 end
#             end
#             iter += 1
#     end
#     map(i -> update_estimate!(i), items);
#     map(e -> update_estimate!(e), examinees);

#     return nothing
# end