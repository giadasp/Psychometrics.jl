include("polyagamma_sampler.jl")
#update the posterior, append sample to chain and set the value as a sample from the posterior
function _mcmc_iter_pg!(
    parameters::AbstractParameters,
    latents::Vector{<:AbstractLatent},
    responses_val::Vector{Union{Missing,Float64}},
    W_val::Vector{Float64};
    sampling = true,
    )
    parameters.posterior = __posterior(parameters, latents, responses_val, W_val) 
    vals = _chain_append!(parameters; sampling = sampling)
    _set_val!(parameters, vals)
    return nothing
end

function mcmc_iter_pg!(
    item::AbstractItem,
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{<:AbstractResponse},
    W::Vector{PolyaGammaSample};
    sampling = true,
    already_sorted = false,
)
    if !already_sorted
        sort!(examinees, by = e -> e.idx)
        sort!(responses, by = r -> r.examinee_idx)
    end
    update_posterior!(item, examinees, responses, W; already_sorted = true)
    set_val_from_posterior!(item; sampling = sampling)
end

function calibrate_item_pg!(
    item::AbstractItem,
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{Response};
    mcmc_iter::Int64 = 4_000,
    sampling::Bool = true,
    already_sorted::Bool = false
)

    if !already_sorted
        sort!(examinees, by = e -> e.idx)
        sort!(responses, by = r -> r.examinee_idx)
        already_sorted = true
    end
    empty_chain!(item)
    # chain = SharedArrays.SharedArray{Float64}(2,mcmc_iter)
    # # extract `mcmc_iter` samples from the polyagamma and from theta conditional posterior
    # Distributed.@sync Distributed.@distributed for iter in 1:mcmc_iter
    #     W = generate_w(item, examinees)
    #     chain[:,iter] = rand(posterior(item, examinees, responses, W))
    # end
    # item.parameters.chain = [c[:] for c in eachcol(chain)]
    # same as, but without val update (faster)
    for iter = 1:mcmc_iter
        W = generate_w(item, examinees)
        mcmc_iter_pg!(item, examinees, responses, W; sampling = sampling, already_sorted = already_sorted)
    end
    update_estimate!(item; sampling = sampling)
end


