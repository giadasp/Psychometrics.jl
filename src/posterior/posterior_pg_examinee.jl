
function _posterior(
    latent::Latent1D,
    parameters::Parameters2PL,
    response::Response,
    w::PolyaGammaSample,
)
    sigma2 = (parameters.a^2) * w.val
    sigma2 = 1 / (sigma2 + (1 / Distributions.var(latent.prior)))
    mu =
        sigma2 * (
            parameters.a * (parameters.a * parameters.b * w.val + (response.val - 0.5)) +
            (latent.prior.μ / Distributions.var(latent.prior))
        )
    return Distributions.Normal(mu, sqrt(sigma2))
end

function posterior(
    examinee::AbstractExaminee,
    item::AbstractItem,
    response::Response,
    w::PolyaGammaSample
)
    return _posterior(examinee.latent, item.parameters, response, w)
end

function update_posterior!(
    examinee::AbstractExaminee,
    items::Vector{<:AbstractItem},
    responses::Vector{<:AbstractResponse},
    W::Vector{PolyaGammaSample};
    already_sorted = false,
    )   
    if !already_sorted
        sort!(items, by = i -> i.idx)
        sort!(responses, by = r -> r.item_idx)
        already_sorted = true
    end
    examinee.latent.posterior = posterior(examinee, items, responses, W; already_sorted = already_sorted)
end


function _posterior(
    latent::Latent1D,
    parameters::Vector{Parameters2PL},
    responses::Vector{Response},
    W::Vector{PolyaGammaSample},
)
    prior = latent.prior
    sigma2 = mapreduce((i, w) -> (i.a^2) * w.val, +, parameters, W)
    sigma2 = 1 / (sigma2 + (1 / Distributions.var(prior)))
    mu =
        sigma2 * (
            mapreduce(
                (i, w, r) -> i.a * (i.a * i.b * w.val +
                                    #(get_responses_by_item_id(i.id, responses_e)[1].val - 0.5)
                                    (r.val - 0.5)),
                +,
                parameters,
                W,
                responses,
            ) + (prior.μ / Distributions.var(prior))
        )
    return Distributions.Normal(mu, sqrt(sigma2))::Distributions.ContinuousDistribution
end

function posterior(
    examinee::AbstractExaminee,
    items::Vector{<:AbstractItem}, #only items answered by examinee sorted by i.idx
    responses::Vector{<:AbstractResponse}, #only responses of examinee sorted by i.idx
    W::Vector{PolyaGammaSample};
    already_sorted = false,
    )   
    if !already_sorted
        sort!(items, by = i -> i.idx)
        sort!(responses, by = r -> r.item_idx)
        already_sorted = true
    end
    return _posterior(examinee.latent, map(i -> i.parameters, items), responses, W)
end