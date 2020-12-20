function calibrate_item!(
    item::AbstractItem,
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{ResponseBinary};
    mcmc_iter = 4_000,
    sampling = true,
)
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
        mcmc_iter!(item, examinees, responses, W; sampling = sampling)
    end
    update_estimate!(item; sampling = sampling)
end

function estimate_ability!(
    examinee::AbstractExaminee,
    items::Vector{<:AbstractItem},
    responses::Vector{ResponseBinary};
    mcmc_iter = 2_000,
    sampling = true,
)
    empty_chain!(examinee)
    # chain = SharedArrays.SharedArray{Float64}(mcmc_iter)
    # # extract `mcmc_iter` samples from the polyagamma and from theta conditional posterior
    # Distributed.@sync Distributed.@distributed for iter in 1:mcmc_iter
    #     W = generate_w(examinee, items)
    #     chain[iter] = rand(posterior(examinee, items, responses, W))
    # end
    # examinee.latent.chain = copy(chain)

    # same as, but without val update (faster)
    for iter = 1:mcmc_iter
        W = generate_w(examinee, items)
        mcmc_iter!(examinee, items, responses, W; sampling = sampling)
    end
    update_estimate!(examinee; sampling = sampling)
end