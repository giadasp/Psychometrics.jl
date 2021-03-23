mutable struct JointEstimationPolyaGammaModel
    parameters::Vector{<:AbstractParameters}
    latents::Vector{<:AbstractLatent}
    responses_per_item::Vector{Vector{Float64}}
    responses_per_examinee::Vector{Vector{Float64}}
    n_index::Vector{Vector{Int64}}
    i_index::Vector{Vector{Int64}}
    item_sampling::Bool
    examinee_sampling::Bool
    ext_opt_settings::Vector{Float64}

    JointEstimationPolyaGammaModel(parameters, latents, responses_per_item, responses_per_examinee, n_index, i_index,item_sampling, examinee_sampling, ext_opt_settings) =
     new(parameters, latents, responses_per_item, responses_per_examinee, n_index, i_index, item_sampling, examinee_sampling, ext_opt_settings)
end