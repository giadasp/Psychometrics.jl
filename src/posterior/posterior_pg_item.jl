

function _posterior(
    parameters::Parameters2PL,
    latent::Latent1D,
    response::Response,
    w::PolyaGammaSample,
)
    a = parameters.a
    b = parameters.b
    latent_val = latent.val
    r_val = response.val
    w_val = w.val

    sigma2 = [(latent_val - b)^2 * w.val, a^2 * w_val]
    sigma2 = 1 ./ (sigma2 .+ (1 ./ Distributions.var.(parameters.prior.v)))
    mu = [(latent_val - b) * (r_val - 0.5), -a * ((r_val - 0.5) - (a * latent_val * w_val))]
    mu =
        sigma2 .* (
            mu + (
                Distributions.mean.(parameters.prior.v) ./
                Distributions.var.(parameters.prior.v)
            )
        )
    return Distributions.Product([
        Distributions.TruncatedNormal(mu[1], sqrt(sigma2[1]), 0.0, Inf),
        Distributions.Normal(mu[2], sqrt(sigma2[2])),
    ])
end

function posterior(
    item::AbstractItem,
    examinee::AbstractExaminee,
    response::AbstractResponse,
    w::PolyaGammaSample,
)
    return _posterior(item.parameters, examinee.latent, response, w)
end


function _posterior(
    parameters::Parameters2PL,
    latents::Vector{Latent1D}, #must be sorted by e.idx
    responses::Vector{Response}, #only responses of item sorted by e.idx
    W::Vector{PolyaGammaSample}, #sorted by e.idx
)
    prior = parameters.prior.v
    a = parameters.a
    b = parameters.b
    #W = map( e -> Distributions.rand(PolyaGammaDevRoye1Sampler(1.0, item.parameters.a *(e.latent.val - item.parameters.b))), examinees)
    sigma2 = mapreduce((e, w) -> [(e.val - b)^2, a^2] .* w.val, +, latents, W)
    sigma2 = 1 ./ (sigma2 + (1 ./ Distributions.var.(prior)))
    mu = mapreduce(
        (e, w, r) -> [(e.val - b) * (r.val - 0.5), -a * (
            #(get_responses_by_examinee_id(e.id, responses)[1].val - 0.5) -
            (r.val - 0.5) - (a * e.val * w.val)
        )],
        +,
        latents,
        W,
        responses,
    )
    mu = sigma2 .* (mu + (Distributions.mean.(prior) ./ Distributions.var.(prior)))
    return Distributions.Product([
        Distributions.TruncatedNormal(mu[1], sqrt(sigma2[1]), 0.0, Inf),
        Distributions.Normal(mu[2], sqrt(sigma2[2])),
    ])
end

function posterior(
    item::AbstractItem,
    examinees::Vector{<:AbstractExaminee}, 
    responses::Vector{<:AbstractResponse},
    W::Vector{PolyaGammaSample};
    already_sorted = false
)
    if !already_sorted
        sort!(examinees, by = e -> e.idx)
        sort!(responses, by = r -> r.examinee_idx)
        already_sorted = true
    end
    return _posterior(item.parameters, map(e -> e.latent, examinees), responses, W)
end

function update_posterior!(
    item::AbstractItem,
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{<:AbstractResponse},
    W::Vector{PolyaGammaSample};
    already_sorted = false,
)   
    if !already_sorted
        sort!(examinees, by = e -> e.idx)
        sort!(responses, by = r -> r.examinee_idx)
        already_sorted = true
    end
    item.parameters.posterior = posterior(item, examinees, responses, W; already_sorted = already_sorted)
end
