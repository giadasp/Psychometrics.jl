function calibrate_item!(
    item::AbstractItem,
    responses::Vector{Response},
    examinees::Vector{<:AbstractExaminee};
    mcmc_iter = 4_000,
    sampling = true,
)
    empty_chain!(item)
    for iter = 1:mcmc_iter
        W = generate_w(item, examinees)
        mcmc_iter!(item, examinees, responses, W; sampling = sampling)
    end
    # chain = Vector{Vector{Float64}}(undef, mcmc_iter)
    # for iter = 1:mcmc_iter
    #     W = generate_w(item, examinees)
    #     chain[iter] = rand(posterior(item, examinees, responses, map(y -> y.val, W)))
    #     #mcmc_iter!(examinees_n, items_est[items_idx_n], resp_n, map(w -> w.val, W); sampling = false)
    # end
    #item.parameters.chain = chain
    update_estimate!(item; sampling = sampling)
end
