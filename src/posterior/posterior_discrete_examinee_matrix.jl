function _posterior(
    prior::Distributions.DiscreteNonParametric,
    items::Vector{<:AbstractItem},
    responses::Vector{Union{Missing, Float64}},
    )   
    return map( (x, w) ->  
            mapreduce( (i, r) -> 
            _likelihood(r, x, i)
            ,
            *,
            items,
            responses,
            )*w,
        prior.support,
        prior.p
        ) 
end

function _update_posterior!(
    latent::Latent1D,
    items::Vector{<:AbstractItem},
    responses::Vector{Union{Missing, Float64}},
    )   
    likelihood = _posterior(latent.prior, items, responses)
    normalizer = sum(likelihood)
    if normalizer > typemin(Float64)
        latent.posterior = Distributions.DiscreteNonParametric(latent.prior.support, likelihood ./ normalizer; check_args = false);
        latent.likelihood = normalizer 
    else
        latent.posterior = Distributions.DiscreteNonParametric(latent.prior.support, likelihood; check_args = false);
    end
    return nothing
end 

function update_posterior!(
    examinee::AbstractExaminee,
    items::Vector{<:AbstractItem},
    responses::Matrix{Union{Missing, Float64}};
    already_sorted::Bool = false,
    kwargs...
    )   
    if !already_sorted 
        responses_filtered = responses[:, examinee.idx]
        items_filtered = items[.!ismissing.(responses_filtered)]
    else
        responses_filtered = responses
        items_filtered = items
    end
    _update_posterior!(examinee.latent, items_filtered, responses_filtered)
    return nothing
end

function update_posterior!(
    examinee::AbstractExaminee,
    items::Vector{<:AbstractItem},
    responses::Vector{Union{Missing, Float64}};
    kwargs...
    )   
    _update_posterior!(examinee.latent, items, responses)
    return nothing
end

function update_posterior!(
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
    responses::Matrix{Union{Missing, Float64}};
    already_sorted::Bool = false,
    kwargs...
)   
    map( e -> update_posterior!(e, items, responses; already_sorted = already_sorted, kwargs...), examinees)
    return nothing
end
