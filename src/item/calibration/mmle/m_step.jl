## for 2PL parameters and latents. Latents and responses are clean (at each examinee in vector corresponds a response)
function m_step(
    parameters::Parameters2PL,
    latents::Vector{<:AbstractLatent}, #only those who answered to item 
    responses::Vector{Float64},
    opt_settings::Vector{Float64}
)
    println(myid())
    opt = NLopt.Opt(:LD_SLSQP, 2)
    opt.maxtime = opt_settings[1]
    opt.xtol_rel = opt_settings[2]
    opt.ftol_rel = opt_settings[3]
    opt.lower_bounds = [parameters.bounds_a[1], parameters.bounds_b[1]]
    opt.upper_bounds = [parameters.bounds_a[2], parameters.bounds_b[2]]
    sumpk_i = mapreduce( l -> l.posterior.p, +, latents)
    X = latents[1].prior.support
    if sum(responses) > 0
        r1_i = mapreduce(l -> l.posterior.p, +, latents[responses .> 0])
    else
        r1_i = zeros(Float64, size(X, 1))
    end
    function myf(x::Vector, grad::Vector)
        phi = x[1] .* (X .- x[2])
        if size(grad, 1) > 0
            p = (r1_i - (sumpk_i .* _sig_c.(phi)))
            grad[1] = sum((X .- x[2]) .* p)
            grad[2] = sum(-x[1] .* p)
        end
        return sum(r1_i .* phi - (sumpk_i .* _log_c.( 1 .+ _exp_c.(phi))))
    end
    opt.max_objective = myf
    pars_i = [parameters.a, parameters.b]
    opt_f = Array{Cdouble}(undef, 1)
    ccall(
        (:nlopt_optimize, NLopt.libnlopt),
        NLopt.Result,
        (NLopt._Opt, Ptr{Cdouble}, Ptr{Cdouble}),
        opt,
        pars_i,
        opt_f,
    )
    println(ccall(
        (:nlopt_optimize, NLopt.libnlopt),
        NLopt.Result,
        (NLopt._Opt, Ptr{Cdouble}, Ptr{Cdouble}),
        opt,
        pars_i,
        opt_f,
    ))
    parameters.a = pars_i[1]
    parameters.b = pars_i[2]
    return parameters::Parameters2PL
end

