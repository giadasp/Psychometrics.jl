function _calibrate_item_mmle!(
    parameters::Parameters2PL,
    examinees::Vector{<:AbstractExaminee}, #only those who answered to item 
    responses::Vector{Union{Missing, Float64}};
    int_opt_x_tol_rel::Float64 = 0.00001,
    int_opt_time_limit::Float64 = 1000.0,
    int_opt_f_tol_rel::Float64 = 0.00001
)
    sumpk_i = mapreduce( e -> e.latent.posterior.p, +, examinees)
    if sum(responses)>0
        r1_i = mapreduce(e -> e.latent.posterior.p, +, examinees[responses .> 0])
    else
        r1_i = [0.0]
    end
    opt = NLopt.Opt(:LD_SLSQP, 2)
    opt.lower_bounds = [parameters.bounds_b[1], parameters.bounds_a[1]]
    opt.upper_bounds = [parameters.bounds_b[2], parameters.bounds_a[2]]
    opt.xtol_rel = int_opt_x_tol_rel
    opt.maxtime = int_opt_time_limit
    opt.ftol_rel = int_opt_f_tol_rel
    pars_i = max_i(examinees[1].latent.posterior.support, sumpk_i, r1_i, [parameters.b, parameters.a], opt)
    parameters.a = clamp(pars_i[2], parameters.bounds_a[1], parameters.bounds_a[2])
    parameters.b = clamp(pars_i[1], parameters.bounds_b[1], parameters.bounds_b[2])
    return nothing
end

function calibrate_item_mmle!(
    item::AbstractItem,
    examinees::Vector{<:AbstractExaminee},
    responses::Matrix{Union{Missing, Float64}};
    already_sorted::Bool = false,
    int_opt_x_tol_rel::Float64 = 0.0001,
    int_opt_time_limit::Float64 = 10.0,
    int_opt_f_tol_rel::Float64 = 0.00001,
)
    if !already_sorted 
        responses_filtered = responses[item.idx,:]
        examinees_filtered = examinees[.!ismissing.(responses_filtered)]
    else
        responses_filtered = responses
        examinees_filtered = examinees
    end

    _calibrate_item_mmle!(item.parameters,
        examinees_filtered,
        responses_filtered;
        int_opt_x_tol_rel = int_opt_x_tol_rel,
        int_opt_time_limit = int_opt_time_limit,
        int_opt_f_tol_rel = int_opt_f_tol_rel
    )
    return nothing
end

function calibrate_item_mmle!(
    item::AbstractItem,
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{Union{Missing, Float64}};
    int_opt_x_tol_rel::Float64 = 0.0001,
    int_opt_time_limit::Float64 = 10.0,
    int_opt_f_tol_rel::Float64 = 0.00001,
)
    _calibrate_item_mmle!(item.parameters, examinees, responses; int_opt_x_tol_rel = int_opt_x_tol_rel, int_opt_time_limit = int_opt_time_limit, int_opt_f_tol_rel = int_opt_f_tol_rel)
    return nothing
end

function calibrate_item_mmle!(
    items::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
    responses::Matrix{Union{Missing, Float64}};
    already_sorted::Bool = false,
    int_opt_x_tol_rel::Float64 = 0.0001,
    int_opt_time_limit::Float64 = 10.0,
    int_opt_f_tol_rel::Float64 = 0.00001,
)
    map( i -> calibrate_item_mmle!(i, examinees, responses; int_opt_x_tol_rel = int_opt_x_tol_rel, int_opt_time_limit = int_opt_time_limit, int_opt_f_tol_rel = int_opt_f_tol_rel), items)
    return nothing
end

