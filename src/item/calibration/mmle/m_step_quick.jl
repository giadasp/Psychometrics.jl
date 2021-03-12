function calibrate_item_mmle_2pl_quick!(
    parameters::Vector{Float64},
    posterior::Vector{Vector{Float64}},
    responses::Vector{Union{Missing, Float64}},
    X::Vector{Float64},
    opt::NLopt.Opt
    )
    sumpk_i = sum(posterior)
    if sum(responses) > 0
        r1_i = sum(posterior[responses .> 0])
    else
        r1_i = [0.0]
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
