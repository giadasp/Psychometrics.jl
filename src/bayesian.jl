struct PolyaGammaSample
    i_idx::Int64
    e_idx::Int64
    val::Float64
    PolyaGammaSample(i_idx, e_idx, val) = new(i_idx, e_idx, val)
end

"""
    posterior(
        item::AbstractItem,
        examinees::Vector{<:AbstractExaminee}, 
        responses::Vector{<:AbstractResponse},
        W::Vector{PolyaGammaSample};
        already_sorted = false,
    )

"""
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

"""
    update_posterior!(
        item::AbstractItem,
        examinees::Vector{<:AbstractExaminee},
        responses::Vector{<:AbstractResponse},
        W::Vector{PolyaGammaSample};
        already_sorted = false,
    )
"""
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

"""
    posterior(
        examinee::AbstractExaminee,
        items::Vector{<:AbstractItem}, #only items answered by examinee sorted by i.idx
        responses::Vector{<:AbstractResponse}, #only responses of examinee sorted by i.idx
        W::Vector{PolyaGammaSample};
        already_sorted = false,
    )
"""
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

"""
    update_posterior!(
        examinee::AbstractExaminee,
        items::Vector{<:AbstractItem},
        responses::Vector{<:AbstractResponse},
        W::Vector{PolyaGammaSample};
        already_sorted = false,
    )
"""
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


"""
    posterior(
        examinee::AbstractExaminee,
        item::AbstractItem,
        response::Response,
        w::PolyaGammaSample,
        )
"""
function posterior(
    examinee::AbstractExaminee,
    item::AbstractItem,
    response::Response,
    w::PolyaGammaSample
)
    return _posterior(examinee.latent, item.parameters, response, w)
end

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


function generate_w(examinee::AbstractExaminee, items::Vector{<:AbstractItem})
    return map(
        i -> PolyaGammaSample(
            i.idx,
            examinee.idx,
            _generate_w(i.parameters, examinee.latent)
        ),
        items,
    )
end

function _generate_w(parameters::Parameters2PL, latent::Latent1D)
    return Distributions.rand(PolyaGamma(1, parameters.a * (latent.val - parameters.b)))
end

function generate_w(item::AbstractItem, examinees::Vector{<:AbstractExaminee})
    return map(
        e -> PolyaGammaSample(item.idx, e.idx, _generate_w(item.parameters, e.latent)),
        examinees,
    )
end

function generate_w(
    items::Vector{<:AbstractItem},
    examinees_i::Vector{Vector{<:AbstractExaminee}},
)
    return mapreduce(
        i -> map(
            e -> PolyaGammaSample(i.idx, e.idx, _generate_w(i.parameters, e.latent)),
            examinees_i[i.idx],
        ),
        vcat,
        items,
    )
end



#extract a random value from posterior and set it as value
function set_val_from_posterior!(item::AbstractItem; sampling = true)
    vals = _chain_append!(item.parameters; sampling = sampling)
    _set_val!(item.parameters, vals)
end

function set_val_from_posterior!(examinee::AbstractExaminee; sampling = true)
    val = _chain_append!(examinee.latent; sampling = sampling)
    _set_val!(examinee.latent, val)
end

#take the last value of the chain and set it as value
function set_val_from_chain!(item::AbstractItem)
    _set_val_from_chain!(item.parameters)
end


#update the posterior, append sample to chain and set the value as a sample from the posterior
function mcmc_iter!(
    item::AbstractItem,
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{<:AbstractResponse},
    W::Vector{PolyaGammaSample};
    sampling = true,
    already_sorted = false,
)
    if !already_sorted
        sort!(examinees, by = e -> e.idx)
        sort!(responses, by = r -> r.examinee_idx)
        already_sorted = true
    end
    update_posterior!(item, examinees, responses, W; already_sorted = already_sorted)
    #set_val_from_chain!(item)
    set_val_from_posterior!(item; sampling = sampling)
end

function mcmc_iter!(
    examinee::AbstractExaminee,
    items::Vector{<:AbstractItem},
    responses::Vector{<:AbstractResponse},
    W::Vector{PolyaGammaSample};
    sampling = true,
    already_sorted = false,
)
    if !already_sorted
        sort!(items, by = i -> i.idx)
        sort!(responses, by = r -> r.examinee_idx)
        already_sorted = true
    end
    update_posterior!(examinee, items, responses, W; already_sorted = already_sorted)
    #chain_append!(examinee; sampling = sampling)
    set_val_from_posterior!(examinee; sampling = sampling)
    #set_val_from_chain!(examinee)
end

function update_estimate!(item::AbstractItem; sampling = true)
    _update_estimate!(item.parameters; sampling = sampling)
end
