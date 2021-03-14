
function joint_estimate_mmle!(
    items::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{<:AbstractResponse};
    dist::Distributions.DiscreteUnivariateDistribution = Distributions.DiscreteNonParametric([0.0,1.0], [0.5, 0.5]),
    metric::Vector{Float64} = [0.0, 1.0],
    max_time::Int64 = 100,
    max_iter::Int64 = 10,
    x_tol_rel::Float64 = 0.001,
    f_tol_rel::Float64 = 0.00001,
    int_opt_x_tol_rel::Float64 = 0.001,
    int_opt_max_time::Float64 = 100.0,
    int_opt_f_tol_rel::Float64 = 0.00001,
    rescale_latent = true,
    kwargs...
    )
    I = size(items, 1)
    N = size(examinees, 1)

    #start points and probs
    probs = Distributions.pdf(Distributions.Normal(metric[1], metric[2]), collect(-6.0:0.3:6.0))
    probs = probs / sum(probs)
    dist = Distributions.DiscreteNonParametric(collect(-6.0:0.3:6.0), probs)

    #set priors 
    set_prior!(examinees, dist);

    #update posteriors
    update_posterior!(examinees, items, responses; already_sorted = false);

    #gr()
    # starting values
    before_time = time()
    stop = false
    old_likelihood = 0.0
    old_pars = get_parameters_vals(items)
    start_time = time()

    #from now we work only on these, the algorithm is dependent on the type of latents and parameters.
    response_matrix = get_response_matrix(responses, size(items,1), size(examinees,1));
    parameters = get_parameters(items)
    latents = get_latents(examinees)

    #extract items per examinee and examinees per item indices
    n_index = Vector{Vector{Int64}}(undef, I)
    i_index = Array{Array{Int64,1},1}(undef, N)
    for n = 1 : N
        i_index[n] = findall(.!ismissing.(response_matrix[:, n]))
        if n <= I
            n_index[n] = findall(.!ismissing.(response_matrix[n, :]))
        end
    end 

    responses_i = [ Vector{Float64}(response_matrix[i, n_index[i]]) for i = 1 : I]
    responses_n = [ Vector{Float64}(response_matrix[i_index[n], n]) for n = 1 : N]

    iter = 1

    opt = NLopt.Opt(:LD_SLSQP, 2)
    opt.xtol_rel = int_opt_x_tol_rel
    opt.maxtime = int_opt_max_time
    opt.ftol_rel = int_opt_f_tol_rel
    while !stop
        #calibrate items
        #before_time = time()
        map( (p, idx, r) -> p.calibrated || _calibrate_item_mmle!(p, latents[idx], r), parameters, n_index, responses_i)

        #println("calibration took ", time() - before_time)
        #before_time = time()
        #rescale dist
        if rescale_latent
            _rescale!(
                dist,
                latents;
                metric = [0.0, 1.0]
            )
            Distributed.@sync Distributed.@distributed for n in 1 : N
                latents[n].prior = dist
            end
        end
        #println("rescale took ", time() - before_time)
        #before_time = time()
        #update posteriors
        map( (l, idx, r) -> l.assessed || _update_posterior!(l, parameters[idx], r), latents, i_index, responses_n)
        #println("post took ", time() - before_time)
        #before_time = time()
        if any([
            check_iter(iter; max_iter = max_iter),
            check_time(start_time; max_time = max_time),
            check_f_tol_rel!(
                latents,
                old_likelihood;
                f_tol_rel = f_tol_rel
                ),
            check_x_tol_rel!(
                items,
                old_pars;
                x_tol_rel = x_tol_rel
            )]
            )
            stop = true
        end
        #println("check took ", time() - before_time)
        #before_time = time()
        iter += 1

    end
    Distributed.@sync Distributed.@distributed for n in 1 : N
        e = examinees[n]
        l = latents[n]
        l.prior = dist
        l.posterior = Distributions.DiscreteNonParametric(dist.support, latents[n].posterior.p)
        l.val = l.posterior.p'*l.posterior.support
        examinees[n] = Examinee(e.idx, e.id, l)
        if n<=I
            i = items[n]
            p = parameters[n]
            items[n] = Item(i.idx, i.id, p)
        end
    end
    return nothing
end