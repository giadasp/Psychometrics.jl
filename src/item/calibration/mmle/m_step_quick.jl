function _calibrate_item_mmle_2pl_quick!(
    parameters::Vector{Float64},
    posterior::Vector{Vector{Float64}},
    responses::Vector{Float64},
    X::Vector{Float64},
    opt::NLopt.Opt
    )
    sumpk_i = reduce(+, posterior)
    if sum(responses) > 0
        r1_i = reduce(+, posterior[responses .> 0])
    else
        r1_i = zeros(Float64, size(X, 1))
    end 
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
        parameters,
        opt_f,
    )
    return nothing
end

function calibrate_item_mmle_2pl_quick!(
    parameters_vectors::Vector{Vector{Float64}},
    n_index::Vector{Vector{Int64}},
    bounds::Vector{Vector{Vector{Float64}}},
    posteriors::Vector{Vector{Float64}},
    responses::Vector{Vector{Float64}},
    X::Vector{Float64};
    int_opt_x_tol_rel::Float64 = 0.0001,
    int_opt_max_time::Float64 = 1000.0,
    int_opt_f_tol_rel::Float64 = 0.00001,
    )
    opt = NLopt.Opt(:LD_SLSQP, 2)
    opt.xtol_rel = int_opt_x_tol_rel
    opt.maxtime = int_opt_max_time
    opt.ftol_rel = int_opt_f_tol_rel

    for i in 1:size(parameters_vectors, 1)
        opt.lower_bounds = [bounds[i][1][1], bounds[i][2][1]]
        opt.upper_bounds = [bounds[i][1][2], bounds[i][2][2]]
        _calibrate_item_mmle_2pl_quick!(
            parameters_vectors[i],
            posteriors[n_index[i]],
            responses[i],
            X,
            opt
        )
    end
    return nothing
end
