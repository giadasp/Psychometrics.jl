mutable struct JointEstimationPolyaGammaModel
    parameters::Vector{<:AbstractParameters}
    latents::Vector{<:AbstractLatent}
    responses_per_item::Vector{Vector{Float64}}
    responses_per_examinee::Vector{Vector{Float64}}
    n_index::Vector{Vector{Int64}}
    i_index::Vector{Vector{Int64}}
    mcmc_iterations::Int64
    max_time::Int64
    item_sampling::Bool
    examinee_sampling::Bool

    JointEstimationPolyaGammaModel(parameters, latents, responses_per_item, responses_per_examinee, n_index, i_index, mcmc_iterations, max_time, item_sampling, examinee_sampling) =
     new(parameters, latents, responses_per_item, responses_per_examinee, n_index, i_index, mcmc_iterations, max_time, item_sampling, examinee_sampling)
end