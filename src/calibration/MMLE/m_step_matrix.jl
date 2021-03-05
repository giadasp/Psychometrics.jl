function max_i(
    X::Vector{Float64},
    sumpk_i::Vector{Float64},
    r1_i::Vector{Float64},
    pars_i::Vector{Float64},
    opt::NLopt.Opt,
)
    new_pars_i = copy(pars_i)
    function myf(x::Vector, grad::Vector)
        phi = x[2] .* (X .- x[1])
        if size(grad, 1) > 0
            p = (r1_i - (sumpk_i .* _sig_c.(phi)))
            grad[1] = sum(-x[2] .* p)
            grad[2] = sum((X .- x[1]) .* p)
        end
        return sum(r1_i .* phi - (sumpk_i .* _log_c.( 1 .+ _exp_c.(phi))))
    end
    opt.max_objective = myf
    opt_f = Array{Cdouble}(undef, 1)
    ccall(
        (:nlopt_optimize, NLopt.libnlopt),
        NLopt.Result,
        (NLopt._Opt, Ptr{Cdouble}, Ptr{Cdouble}),
        opt,
        new_pars_i,
        opt_f,
    )
    #change parametrization from b + at to a(t-b) 
    return new_pars_i::Vector{Float64}
end


function _calibrate_item_mmle!(
    parameters::Parameters2PL,
    examinees::Vector{<:AbstractExaminee}, #only those who answered to item 
    responses::Vector{Union{Missing, Float64}};
    int_opt_x_tol_rel::Float64 = 0.00001,
    int_opt_time_limit::Float64 = 1000.0,
    int_opt_f_tol_rel::Float64 = 0.00001
)
    sumpk_i = mapreduce( e -> e.latent.posterior.p, +, examinees)
    r1_i =  mapreduce(e -> e.latent.posterior.p, +, examinees[responses .> 0])
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
        items_filtered = examinees[.!ismissing.(responses_filtered)]
    else
        responses_filtered = responses
        items_filtered = items
    end

    _calibrate_item_mmle!(item.parameters,
        items_filtered,
        responses_filtered;
        int_opt_x_tol_rel = int_opt_x_tol_rel,
        int_opt_time_limit = int_opt_time_limit,
        int_opt_f_tol_rel = int_opt_f_tol_rel
    )
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