mutable struct JointEstimationMMLEModel
    parameters::Vector{<:AbstractParameters}
    latents::Vector{<:AbstractLatent}
    responses_per_item::Vector{Vector{Float64}}
    responses_per_examinee::Vector{Vector{Float64}}
    n_index::Vector{Vector{Int64}}
    i_index::Vector{Vector{Int64}}
    dist::Distributions.DiscreteUnivariateDistribution
    metric::Vector{Float64}
    rescale_latent::Bool
    ext_opt_settings::Vector{Float64}
    int_opt_settings::Vector{Float64}

    JointEstimationMMLEModel(parameters, latents, responses_per_item, responses_per_examinee, n_index, i_index, dist, metric, rescale_latent, ext_opt_settings, int_opt_settings) =
     new(parameters, latents, responses_per_item, responses_per_examinee, n_index, i_index, dist, metric, rescale_latent, ext_opt_settings, int_opt_settings)
end