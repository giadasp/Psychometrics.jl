function max_i_quick(
    X::Vector{Float64},
    sumpk_i::Vector{Float64},
    r1_i::Vector{Float64},
    pars_i::Vector{Float64},
    opt::NLopt.Opt,
)
    new_pars_i = copy(pars_i)
    function myf(x::Vector, grad::Vector)
        phi = x[1] .* (X .- x[2])
        if size(grad, 1) > 0
            p = (r1_i - (sumpk_i .* _sig_c.(phi)))
            grad[2] = sum(-x[1] .* p)
            grad[1] = sum((X .- x[2]) .* p)
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
function calibrate_item_mmle_quick(
    parameters::Vector{Float64},
    bounds::Vector{Vector{Float64}},
    posterior::Vector{Vector{Float64}},
    responses::Vector{Union{Missing, Float64}},
    X::Vector{Float64};
    int_opt_x_tol_rel::Float64 = 0.0001,
    int_opt_time_limit::Float64 = 1000.0,
    int_opt_f_tol_rel::Float64 = 0.00001,
    kwargs...
)
    sumpk_i = sum(posterior)
    if sum(responses) > 0
        r1_i = sum(posterior[responses .> 0])
    else
        r1_i = [0.0]
    end
    opt = NLopt.Opt(:LD_SLSQP, 2)
    opt.lower_bounds = [bounds[1][1], bounds[2][1]]
    opt.upper_bounds = [bounds[1][2], bounds[2][2]]
    opt.xtol_rel = int_opt_x_tol_rel
    opt.maxtime = int_opt_time_limit
    opt.ftol_rel = int_opt_f_tol_rel
    new_pars_i = max_i_quick(X, sumpk_i, r1_i, parameters, opt)
    return new_pars_i
end
