include("je_polyagamma_struct.jl")
include("je_polyagamma_quick.jl")
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
    
    responses_per_examinee = map(
        e -> sort(get_responses_by_examinee_id(e.id, responses), by = x -> x.item_idx),
        examinees,
    );
    items_idx_per_examinee = map(
        e -> sort(map(r -> items[r.item_idx].idx, responses_per_examinee[e.idx])),
        examinees,
    );
    responses_per_item = map(
        i -> sort(get_responses_by_item_id(i.id, responses), by = x -> x.examinee_idx),
        items,
    );
    examinees_idx_per_item = map(
        i -> sort(map(r -> examinees[r.examinee_idx].idx, responses_per_item[i.idx])),
        items,
    );

    map(
        i -> begin
            i.parameters.chain = [[i.parameters.a, i.parameters.b] for j = 1:1000]
        end,
        items,
    );

    stop = false
    old_pars = get_parameters_vals(items)
    start_time = time()
    iter = 1

    while !stop
            W = generate_w(
                items,
                map(i -> examinees[examinees_idx_per_item[i.idx]], items),
            )
            map(
                i -> mcmc_iter_pg!(
                    i,
                    examinees[examinees_idx_per_item[i.idx]],
                    responses_per_item[i.idx],
                    filter(w -> w.i_idx == i.idx, W);
                    sampling = item_sampling,
                    already_sorted = true,
                ),
                items,
            )
            map(
                e -> mcmc_iter_pg!(
                    e,
                    items[items_idx_per_examinee[e.idx]],
                    responses_per_examinee[e.idx],
                    filter(w -> w.e_idx == e.idx, W);
                    sampling = examinee_sampling,
                    already_sorted = true,
                ),
                examinees,
            )
            if (iter % 200) == 0
                #map(i -> update_estimate!(i), items);
                if any([
                    check_iter(iter; max_iter = mcmc_iter),
                    check_time(start_time; max_time = max_time)
                    #check_x_tol_rel!(
                    #     items,
                    #     old_pars;
                    #     x_tol_rel = x_tol_rel)
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