include("polyagamma_sampler.jl")

function _mcmc_iter_pg(
    latent::AbstractLatent,
    parameters::Vector{<:AbstractParameters},
    responses_val::Vector{Float64},
    W_val::Vector{Float64};
    sampling = true
    )
        latent.posterior = __posterior(latent, parameters, responses_val, W_val) 
        vals = _chain_append!(latent; sampling = sampling)
        _set_val!(latent, vals)
    return latent::AbstractLatent
end

function mcmc_iter_pg!(
    examinee::AbstractExaminee,
    items::Vector{<:AbstractItem},
    responses::Vector{<:AbstractResponse},
    W::Vector{PolyaGammaSample};
    sampling = true,
    already_sorted = false,
)
    if !already_sorted
        sort!(items, by = i -> i.idx)
        sort!(responses, by = r -> r.examinee_idx)
    end
    update_posterior!(examinee, items, responses, W; already_sorted = true)
    set_val_from_posterior!(examinee; sampling = sampling)
end

function update_estimate!(item::AbstractItem; sampling = true)
    _update_estimate!(item.parameters; sampling = sampling)
end

function _assess_examinee_pg(
    latent::AbstractLatent,
    parameters::Vector{<:AbstractParameters},
    responses::Vector{Float64};
    mcmc_iter = 2_000,
    sampling = true
    )
    # same as, but without val update (faster)
    for iter = 1:mcmc_iter
        W = map( p -> _generate_w(latent, p), parameters)
        latent = _mcmc_iter_pg(latent, parameters, responses, W; sampling = sampling)
    end
    return latent::AbstractLatent
end

function assess_examinee_pg!(
    examinee::AbstractExaminee,
    items::Vector{<:AbstractItem},
    responses::Vector{Response};
    mcmc_iter = 2_000,
    sampling = true,
    already_sorted = false,
    kwargs...
)
    if !already_sorted
        sort!(responses, by = r -> r.item_idx)
        sort!(items, by = i -> i.idx)
    end
    latent = examinee.latent
    _empty_chain!(latent)
    # chain = SharedArrays.SharedArray{Float64}(mcmc_iter)
    # # extract `mcmc_iter` samples from the polyagamma and from theta conditional posterior
    # Distributed.@sync Distributed.@distributed for iter in 1:mcmc_iter
    #     W = generate_w(examinee, items)
    #     chain[iter] = rand(posterior(examinee, items, responses, W))
    # end
    # examinee.latent.chain = copy(chain)

    # same as, but without val update (faster)
    latent = _assess_examinee_pg(
        latent,
        get_parameters(items),
        map( r -> r.val, responses),
        mcmc_iter = mcmc_iter,
        sampling = sampling,
        )
    update_estimate!(examinee; sampling = sampling)
    return nothing
end