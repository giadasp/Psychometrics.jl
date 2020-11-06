function calibrate_item!(item::AbstractItem, responses::Vector{Response}, examinees::Vector{<:AbstractExaminee}; mcmc_iter = 4_000)
    for iter in 1:mcmc_iter 
        W = generate_w(item, examinees)
        mcmc_iter!(item, examinees, responses, map( y -> y.val, W); sampling = true)
    end
    update_estimate!(item)
end