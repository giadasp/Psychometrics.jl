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

function calibrate_item_mmle!(
    items::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{Response};
    int_opt_x_tol_rel::Float64 = 0.0001,
    int_opt_time_limit::Float64 = 10.0,
    int_opt_f_tol_rel::Float64 = 0.00001,
)
    Distributed.pmap( i -> calibrate_item_mmle!(i, examinees, responses; int_opt_x_tol_rel =int_opt_x_tol_rel, int_opt_time_limit = int_opt_time_limit, int_opt_f_tol_rel = int_opt_f_tol_rel), items)
    return nothing
end

function calibrate_item_mmle!(
    item::AbstractItem,
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{Response};
    int_opt_x_tol_rel::Float64 = 0.0001,
    int_opt_time_limit::Float64 = 10.0,
    int_opt_f_tol_rel::Float64 = 0.00001,
)
    responses_1 = filter(r -> r.item_idx == item.idx, responses)
    examinees_1 = examinees[map(r -> r.examinee_idx, responses_1)]

    _calibrate_item_mmle!(item.parameters,
        examinees_1,
        responses_1;
        int_opt_x_tol_rel = int_opt_x_tol_rel,
        int_opt_time_limit = int_opt_time_limit,
        int_opt_f_tol_rel = int_opt_f_tol_rel
    )
    return nothing
end

function _calibrate_item_mmle!(
    parameters::Parameters2PL,
    examinees::Vector{<:AbstractExaminee}, #only those who answered to item 
    responses::Vector{Response}; #sorted by examinee_idx
    int_opt_x_tol_rel::Float64 = 0.00001,
    int_opt_time_limit::Float64 = 1000.0,
    int_opt_f_tol_rel::Float64 = 0.00001
)
    sumpk_i = mapreduce( e -> e.latent.posterior.p, +, examinees)
    responses_1 = map( r2 -> r2.examinee_idx, filter(r -> r.val > 0.0, responses))
    r1_i =  mapreduce(e -> e.latent.posterior.p, +, filter(e -> e.idx in responses_1, examinees))
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