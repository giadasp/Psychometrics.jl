include("m_step.jl")

function calibrate_item_mmle!(
    item::AbstractItem,
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{Response};
    clean = false,
    int_opt_x_tol_rel::Float64 = 0.0001,
    int_opt_max_time::Float64 = 10.0,
    int_opt_f_tol_rel::Float64 = 0.00001,
)
    if !clean
        responses = sort(filter(r -> r.item_idx == item.idx, responses), by = r -> r.examinee_idx)
        examinees_idx = map(r -> r.examinee_idx, responses)
        examinees = sort(filter( e -> e.idx in examinees_idx, examinees), by = e -> e.idx)
    end
    item.parameters = m_step(
        item.parameters,
        get_latents(examinees),
        map(r -> r.val, responses),
        [Float64(int_opt_max_time), int_opt_x_tol_rel, int_opt_f_tol_rel]
        )
    #pmap( i -> calibrate_item_mmle!(i, examinees, responses; int_opt_x_tol_rel =int_opt_x_tol_rel, int_opt_max_time = int_opt_max_time, int_opt_f_tol_rel = int_opt_f_tol_rel), items)
    return nothing
end

function calibrate_item_mmle!(
    items::Vector{AbstractItem},
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{Response};
    clean = false,
    int_opt_x_tol_rel::Float64 = 0.0001,
    int_opt_max_time::Float64 = 10.0,
    int_opt_f_tol_rel::Float64 = 0.00001,
)
    pmap( i -> calibrate_item_mmle!(
            i,
            examinees,
            responses;
            int_opt_max_time = int_opt_max_time,
            int_opt_x_tol_rel = int_opt_x_tol_rel,
            int_opt_f_tol_rel = int_opt_f_tol_rel,
            clean = clean,
        ),
        items
    )
    return nothing
end



