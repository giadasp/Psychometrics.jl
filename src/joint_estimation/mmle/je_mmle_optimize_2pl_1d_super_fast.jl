function likelihood!(
    likelihood::Matrix{Float64},
    N::Int64,
    K::Int64,
    i_index::Vector{Vector{Int64}},
    response_matrix::Matrix{Float64},
    phi::Matrix{Float64},
)
    ephizero = _sig_cplus.(phi)
    ephione = _sig_c.(phi)
    post_n = zeros(Float64, K)
    for n = 1:N
        for k = 1:K
            post_k = one(Float64)
            for i in i_index[n]
                if response_matrix[i, n] > 0
                    post_k *= ephione[k, i]
                else
                    post_k *= ephizero[k, i]
                end
            end
            likelihood[n, k] = copy(post_k)
        end
    end
    return nothing
end

function posterior_simplified!(
    post::Matrix{Float64},
    N::Int64,
    K::Int64,
    i_index::Vector{Vector{Int64}},
    r::Matrix{Float64},
    Wk::Vector{Float64},
    phi::Matrix{Float64},
)
    ephizero = _sig_cplus.(phi)
    ephione = 1 .- ephizero
    post_n = zeros(Float64, K)
    for n = 1:N
        for k = 1:K
            post_k = one(Float64)
            for i in i_index[n]
                if r[i, n] > 0
                    post_k *= ephione[k, i]
                else
                    post_k *= ephizero[k, i]
                end
            end
            post_n[k] = copy(post_k)
        end
        post_n = (post_n .* Wk) #modify with first_latent
        exp_cd = sum(post_n)
        if exp_cd > typemin(Float64)
            post_n = post_n ./ exp_cd
        end
        post[n, :] = copy(post_n)
    end
    return nothing
end

function max_i(
    X::Matrix{Float64},
    sumpk_i::Vector{Float64},
    r1_i::Vector{Float64},
    pars_i::Vector{Float64},
    bounds::Vector{Vector{Float64}},
    opt_settings::Vector{Float64}
)
    new_pars_i = copy(pars_i)
    opt = NLopt.Opt(:LD_SLSQP, 2)
    opt.maxtime = opt_settings[1]
    opt.xtol_rel = opt_settings[2]
    opt.ftol_rel = opt_settings[3]
    opt.lower_bounds = [bounds[1][1], bounds[2][1]]
    opt.upper_bounds = [bounds[1][2], bounds[2][2]]
    function myf(x::Vector, grad::Vector)
        n_par = size(x, 1)
        if n_par == 2
            y = X * x
        else
            y = x
        end
        if size(grad, 1) > 0
            p = r1_i - (sumpk_i .* _sig_c.(y))
            p = X' * p
            for i = 1:size(grad, 1)
                grad[i] = p[i]
            end
        end
        return sum(r1_i .* y - (sumpk_i .* _log_c.( 1 .+ _exp_c.(y))))
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
    return new_pars_i::Vector{Float64}
end

function max_LH_MMLE!(
    pars_start::Matrix{Float64},
    phi::Matrix{Float64},
    posterior::Matrix{Float64},
    i_index::Vector{Vector{Int64}},
    design::Matrix{Float64},
    X::Matrix{Float64},
    Wk::Vector{Float64},
    response_matrix::Matrix{Float64},
    opt_settings::Vector{Float64},
    bounds::Vector{Vector{Vector{Float64}}},
)
    n_items = size(pars_start, 2)
    N = size(i_index, 1)
    K = size(X, 1)
    sumpk = zeros(Float64, K, n_items)
    r1 = similar(sumpk)
    posterior_simplified!(posterior, N, K, i_index, response_matrix, Wk, phi)
    LinearAlgebra.BLAS.gemm!(
        'T',
        'T',
        one(Float64),
        posterior,
        design,
        zero(Float64),
        sumpk,
    )# sumpk KxI
    LinearAlgebra.BLAS.gemm!(
        'T',
        'T',
        one(Float64),
        posterior,
        response_matrix,
        zero(Float64),
        r1,
    )# r1 KxI

    #opt.maxeval=50
    pars_start = Distributed.@sync Distributed.@distributed (hcat) for i = 1:n_items
        max_i(X, sumpk[:, i], r1[:, i], pars_start[:, i], bounds[i], opt_settings)
    end
    LinearAlgebra.BLAS.gemm!('N', 'N', one(Float64), X, pars_start, zero(Float64), phi)# phi=New_pars*X1', if A'*B then 'T', 'N'
    return pars_start::Matrix{Float64}
end

function optimize_2pl_1d_super_fast(je_mmle_model::JointEstimationMMLEModel)
    local parameters = je_mmle_model.parameters
    local latents = je_mmle_model.latents
    local dist = je_mmle_model.dist
    local n_index = je_mmle_model.n_index
    local i_index = je_mmle_model.i_index
    local responses_per_item = je_mmle_model.responses_per_item
    local responses_per_examinee = je_mmle_model.responses_per_examinee

    I = size(parameters, 1)
    N = size(latents, 1)
    K = size(dist.support, 1)

    Xk = dist.support
    X = hcat(ones(K), Xk)
    Wk = dist.p

    iter = 1

    # starting values
    before_time = time()
    stop = false
    old_likelihood = -Inf
    old_pars = hcat(_get_parameters_vals.(parameters)...)
    #change parametrization form a(t-b) to b+at
    old_pars[2, :] = .-old_pars[2, :] .* old_pars[1, :]
    old_pars_2 = copy(old_pars)
    old_pars[1, :] = copy(old_pars_2[2, :])
    old_pars[2, :] = copy(old_pars_2[1, :])

    new_pars = copy(old_pars)

    items_bounds = map( p -> [[min(-p.bounds_b[1], -p.bounds_b[2]), max(-p.bounds_b[1], -p.bounds_b[2])], p.bounds_a], parameters)
    
    phi = X * new_pars


    design = zeros(Float64, I, N)
    for i in 1:I
        design[i, n_index[i]] .= 1.0
    end

    response_matrix = zeros(Float64, I, N)
    for i = 1:I
        response_matrix[i, n_index[i]] = responses_per_item[i]
    end

    posterior = zeros(Float64, N, K)
    likelihood_matrix = similar(posterior)
    oneoverN = fill(1 / N, N)

    posterior_simplified!(
                        posterior,
                        N,
                        K,
                        i_index,
                        response_matrix,
                        Wk,
                        phi,
                    )

    start_time = time()

    while !stop
        # calibrate items
        new_pars = max_LH_MMLE!(
            new_pars,
            phi,
            posterior,
            i_index,
            design,
            X,
            Wk,
            response_matrix,
            je_mmle_model.int_opt_settings,
            items_bounds,
        )
        likelihood!(likelihood_matrix, N, K, i_index, response_matrix, phi)
        new_likelihood = sum(
            _log_c.(
                LinearAlgebra.BLAS.gemv('N', one(Float64), likelihood_matrix, Wk),
            )
        )
        posterior_simplified!(
            posterior,
            N,
            K,
            i_index,
            response_matrix,
            Wk,
            phi,
        )
        Wk = LinearAlgebra.BLAS.gemv('T', one(Float64), posterior, oneoverN) #if Wk depends only on the likelihoods
        if je_mmle_model.rescale_latent
            observed = [LinearAlgebra.dot(Wk, Xk), sqrt(LinearAlgebra.dot(Wk, Xk .^ 2))]
            observed = [
                observed[1] - je_mmle_model.metric[1],
                observed[2] / je_mmle_model.metric[2],
            ]
            #check mean
            if  (abs(observed[1]) > 1e-4 ) || (abs(observed[2] - 1.0) > 1e-4)
                Xk2, Wk2 = my_rescale(Xk, Wk, observed)
                Wk = cubic_spline_int(Xk, Xk2, Wk2)
                X = hcat(ones(K), Xk2[2:(K+1)])
            end
        end
        if any([
            check_iter(iter; max_iter = Int64(je_mmle_model.ext_opt_settings[1]), verbosity = Int(je_mmle_model.ext_opt_settings[5])),
            check_time(start_time; max_time = Int64(je_mmle_model.ext_opt_settings[2]), verbosity = Int(je_mmle_model.ext_opt_settings[5])),
            check_f_tol_rel!(
                new_likelihood,
                old_likelihood;
                f_tol_rel = je_mmle_model.ext_opt_settings[3],
                verbosity = Int(je_mmle_model.ext_opt_settings[5])
                ),
            check_x_tol_rel!(
                new_pars,
                old_pars;
                x_tol_rel = je_mmle_model.ext_opt_settings[4],
                verbosity = Int(je_mmle_model.ext_opt_settings[5])
            )]
            )
            stop = true
        end
        iter += 1
    end
    #change parametrization form a(t-b) to b+at
    new_pars[1, :] = .-new_pars[1, :] ./ new_pars[2, :]
    map(i -> begin
            parameters[i].a = new_pars[2, i]
            parameters[i].b = new_pars[1, i]
        end
        ,1 : I)

    map(n -> begin
            latents[n].posterior = Distributions.DiscreteNonParametric(Xk, posterior[n, :])
        end
        ,1 : N)
    dist = Distributions.DiscreteNonParametric(Xk, Wk)
    return parameters::Vector{<:AbstractParameters}, latents::Vector{<:AbstractLatent}, dist::Distributions.DiscreteUnivariateDistribution
end
