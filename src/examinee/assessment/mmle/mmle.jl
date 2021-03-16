function assess_examinee_mmle!(
    examinee::AbstractExaminee,
    items::Vector{<:AbstractItem},
    responses::Vector{Response};
    clean = false,
    rescale_latent = false, 
    dist::Distributions.DiscreteUnivariateDistribution = Distributions.DiscreteNonParametric([0.0,1.0], [0.5, 0.5]),
    max_time::Int64 = 100,
    max_iter::Int64 = 10,
    x_tol_rel::Float64 = 0.001,
    f_tol_rel::Float64 = 0.00001,
    kwargs...
    )
    if !examinee.latent.assessed
        if !clean
            responses = sort(filter(r -> r.examinee_idx == examinee.idx, responses), by = r -> r.item_idx)
            items_idx = map(r -> r.item_idx, responses)
            items = sort(filter( i -> i.idx in items_idx, items), by = i -> i.idx)
        end

        parameters = get_parameters(items)
        latent = examinee.latent
        latent.prior = dist

        I = size(parameters, 1)
        responses_per_examinee = map( r -> r.val ,responses)
        iter = 1

        # starting values
        before_time = time()
        stop = false
        old_likelihood = 0.0
        old_latents = hcat(_get_latents.([latent])...)
        start_time = time()

        latent = _update_posterior(latent, parameters, responses_per_examinee),
        while !stop
            #rescale dist
            if rescale_latent
                dist = _rescale(
                    dist,
                    [latent];
                    metric = metric
                )
                latent.prior = dist
            end
            latent = _update_posterior(latent, parameters[i_index], responses_per_examinee),

            if any([
                check_iter(iter; max_iter = max_iter),
                check_time(start_time; max_time = max_time),
                check_f_tol_rel!(
                    [latent],
                    old_likelihood;
                    f_tol_rel = f_tol_rel
                    ),
                check_x_tol_rel!(
                    [latent],
                    old_latents;
                    x_tol_rel = x_tol_rel
                )]
                )
                stop = true
            end
            iter += 1
        end
        latent.posterior = Distributions.DiscreteNonParametric(dist.support, latent.posterior.p)
        latent.val = latent.posterior.p' * latent.posterior.support
        examinee = Examinee(examinee.idx, examinee.id, latent)
    end
    return nothing
end

function assess_examinee_mmle!(
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
    responses::Vector{Response};
    dist::Distributions.DiscreteUnivariateDistribution = Distributions.DiscreteNonParametric([0.0,1.0], [0.5, 0.5]),
    clean = false,
    max_time::Int64 = 100,
    max_iter::Int64 = 10,
    x_tol_rel::Float64 = 0.001,
    f_tol_rel::Float64 = 0.00001,
    kwargs...
    )
    pmap( e -> assess_examinee_mmle!(
            e,
            items,
            responses;
            clean = false, 
            max_time = max_time, 
            max_iter = max_iter, 
            x_tol_rel = x_tol_rel,
            f_tol_rel = f_tol_rel
            ),
            examinees,
            batch_size = Int(ceil(size(examinees, 1) / (nprocs()-1)))
            )
    return nothing
end
