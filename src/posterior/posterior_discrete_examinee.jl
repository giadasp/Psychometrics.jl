"""
```julia 
_posterior(
    prior::Distributions.DiscreteNonParametric,
    responses::Vector{<:AbstractResponse},
    items::Vector{<:AbstractResponse},
    )
```

#Description

It computes the posterior (not normalized) for a discrete non parametric distribution prior (`Distributions.DiscreteNonParametric`), vector of items `items` with vector of answers `responses`.
Items and responses must be sorted by idx (items) and item_idx (responses)
"""
function _posterior(
    prior::Distributions.DiscreteNonParametric,
    items::Vector{<:AbstractItem},
    responses::Vector{<:AbstractResponse},
    )   
    return map( (x, w) ->  
            mapreduce( (i, r) -> 
            _likelihood(r, x, i)
            ,
            *,
            items,
            responses,
            )*w
        ,
        prior.support,
        prior.p
        ) 
end

function _update_posterior!(
    latent::Latent1D,
    items::Vector{<:AbstractItem}, #! must be sorted by idx
    responses::Vector{<:AbstractResponse}, #! must be sorted by item_idx
    )   
    likelihood = _posterior(latent.prior, items, responses)
    normalizer = sum(likelihood)
    if normalizer > typemin(Float64)
        latent.posterior = Distributions.DiscreteNonParametric(latent.prior.support, likelihood ./ normalizer; check_args = false);
        latent.likelihood = normalizer 
    else
        latent.posterior = Distributions.DiscreteNonParametric(latent.prior.support, likelihood; check_args = false);
    end
end 

function update_posterior!(
    examinee::AbstractExaminee,
    items::Vector{<:AbstractItem},
    responses::Vector{<:AbstractResponse};
    already_sorted::Bool = false,
    kwargs...
    )   
    if !already_sorted 
        responses_filtered = filter( r -> r.examinee_idx == examinee.idx, responses)
        items_filtered = items[map(r -> r.item_idx, responses_filtered)]
    else
        responses_filtered = responses
        items_filtered = items
    end
    _update_posterior!(examinee.latent, items_filtered, responses_filtered)
    return nothing
end

function update_posterior!(
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
    responses::Vector{<:AbstractResponse};
    already_sorted::Bool = false,
    kwargs...
)   
    map( e -> update_posterior!(e, items, responses; already_sorted = already_sorted, kwargs...), examinees)
    return nothing
end
