struct PolyaGammaSample
    i_idx::Int64
    e_idx::Int64
    val::Float64
    PolyaGammaSample(i_idx, e_idx, val) = new(i_idx, e_idx, val)
end

function posterior(
    item::AbstractItem,
    examinees::Vector{<:AbstractExaminee}, #must be sorted by e.idx
    responses::Vector{ResponseBinary}, #only responses of item sorted by e.idx
    W::Vector{PolyaGammaSample}, #sorted by e.idx
)
    return posterior(item.parameters,
        map(e -> e.latent, examinees),
        responses,
        W,
    )
end

function update_posterior!(
    item::AbstractItem,
    examinees::Vector{<:AbstractExaminee}, #must be sorted by e.idx
    responses::Vector{<:AbstractResponse}, #only responses of item sorted by e.idx
    W::Vector{PolyaGammaSample}, #sorted by e.idx
)
    item.parameters.posterior = posterior(item, examinees, responses, W)
end

function posterior(
    examinee::AbstractExaminee,
    items::Vector{<:AbstractItem}, #only items answered by examinee sorted by i.idx
    responses::Vector{<:AbstractResponse}, #only responses of examinee sorted by i.idx
    W::Vector{PolyaGammaSample},
)
    return posterior(examinee.latent,
        map(i -> i.parameters, items),
        responses,
        W,
    )
end

function posterior(
    latent::Latent1D,
    parameters::Vector{Parameters2PL},
    responses::Vector{ResponseBinary},
    W::Vector{PolyaGammaSample},
)
    prior = latent.prior
    sigma2 = mapreduce((i, w) -> (i.a^2) * w.val, +, parameters, W)
    sigma2 = 1 / (sigma2 + (1 / Distributions.var(prior)))
    mu =
        sigma2 * (
            mapreduce(
                (i, w, r) ->
                    i.a * (i.a * i.b * w.val +
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

function update_posterior!(
    examinee::AbstractExaminee,
    items::Vector{<:AbstractItem}, #only items answered by examinee sorted by i.idx
    responses::Vector{<:AbstractResponse}, #only responses of examinee sorted by i.idx
    W::Vector{PolyaGammaSample},
)
    examinee.latent.posterior = posterior(examinee, items, responses, W)
end

function posterior(
    latent::Latent1D,
    parameters::Parameters2PL,
    w::PolyaGammaSample,
    r::ResponseBinary
    )
    sigma2 = (parameters.a^2) * w.val
    sigma2 = 1 / (sigma2 + (1 / Distributions.var(latent.prior)))
    mu =
        sigma2 * (
            parameters.a *
            (parameters.a * parameters.b * w.val + (r.val - 0.5)) +
            (latent.prior.μ / Distributions.var(latent.prior))
        )
    return Distributions.Normal(mu, sqrt(sigma2))
end


"""
    posterior(
        examinee::AbstractExaminee,
        item::AbstractItem,
        w::PolyaGammaSample,
        r::ResponseBinary
        )
"""
function posterior(
    examinee::AbstractExaminee,
    item::AbstractItem,
    w::PolyaGammaSample,
    r::ResponseBinary
    )
    return posterior(
        examinee.latent,
        item.parameters,
        w,
        r
        )
end

function posterior(
    parameters::Parameters2PL, 
    latent::Latent1D, 
    w::PolyaGammaSample,
    r::ResponseBinary
    )
    a = parameters.a
    b = parameters.b
    latent_val = latent.val
    r_val = r.val
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
    w::PolyaGammaSample,
    r::ResponseBinary
    )
    return posterior(item.parameters,
    examinee.latent,
    w,
    r,
    )
end


function posterior(
    parameters::Parameters2PL,
    latents::Vector{Latent1D}, #must be sorted by e.idx
    responses::Vector{ResponseBinary}, #only responses of item sorted by e.idx
    W::Vector{PolyaGammaSample}, #sorted by e.idx
)
    prior = parameters.prior.v
    a = parameters.a
    b = parameters.b
    #W = map( e -> Distributions.rand(PolyaGammaDevRoye1Sampler(1.0, item.parameters.a *(e.latent.val - item.parameters.b))), examinees)
    sigma2 = mapreduce((e, w) -> [(e.val - b)^2, a^2] .* w.val, +, latents, W)
    sigma2 = 1 ./ (sigma2 + (1 ./ Distributions.var.(prior)))
    mu = mapreduce(
        (e, w, r) -> [
            (e.val - b) * (r.val - 0.5),
            -a * (
                #(get_responses_by_examinee_id(e.id, responses)[1].val - 0.5) -
                (r.val - 0.5) - (a * e.val * w.val)
            ),
        ],
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


"""
    posterior(
        item::AbstractItem,
        examinees::Vector{<:AbstractExaminee}, #must be sorted by e.idx
        responses::Vector{ResponseBinary}, #only responses of item sorted by e.idx
        W::Vector{PolyaGammaSample}, #sorted by e.idx
    )

"""
function posterior(
    item::AbstractItem,
    examinees::Vector{<:AbstractExaminee}, #must be sorted by e.idx
    responses::Vector{ResponseBinary}, #only responses of item sorted by e.idx
    W::Vector{PolyaGammaSample}, #sorted by e.idx
)
    return posterior(item.parameters,
        map(e -> e.latent, examinees),
        responses,
        W,
    )
end


function generate_w(examinee::AbstractExaminee, items::Vector{<:AbstractItem})
    return map(
        i -> PolyaGammaSample(
            i.idx,
            examinee.idx,
            Distributions.rand(PolyaGamma(
                1,
                i.parameters.a * (examinee.latent.val - i.parameters.b),
            )),
        ),
        items,
    )
end

function generate_w(parameters::Parameters2PL, latent::Latent1D)
    return Distributions.rand(PolyaGamma(
                1,
                parameters.a * (latent.val - parameters.b),
            ))
end

function generate_w(item::AbstractItem, examinees::Vector{<:AbstractExaminee})
    return map(
        e -> PolyaGammaSample(
            item.idx,
            e.idx,
            generate_w(item.parameters, e.latent)
        ),
        examinees,
    )
end

function generate_w(items::Vector{<:AbstractItem}, examinees_i::Vector{Vector{<:AbstractExaminee}})
    return mapreduce(
        i -> map(
            e -> PolyaGammaSample(
                i.idx,
                e.idx,
                generate_w(i.parameters, e.latent)
            ),
            examinees_i[i.idx],
        ),
        vcat,
        items,
    )
end

#extract a value from the posterior and append it to the chain
function chain_append!(parameters::Union{Parameters2PL,Parameters3PL}; sampling = false)
    val = Distributions.rand(parameters.posterior)
    if (sampling && size(parameters.chain, 1) >= 1000)
        parameters.chain[Random.rand(1:1000)] = val
    else
        push!(parameters.chain, val)
    end
    return val::Vector{Float64}
end

function chain_append!(latent::Latent1D; sampling = false)
    val = Distributions.rand(latent.posterior)
    if (sampling && size(latent.chain, 1) >= 1000)
        latent.chain[Random.rand(1:1_000)] = val
    else
        push!(latent.chain, val)
    end
    return val::Float64
end


#extract a random value from posterior and set it as value
function set_val_from_posterior!(item::AbstractItem; sampling = true)
    vals = chain_append!(item.parameters; sampling = sampling)
    set_val!(item.parameters, vals)
end

function set_val_from_posterior!(examinee::AbstractExaminee; sampling = true)
    val = chain_append!(examinee.latent; sampling = sampling)
    set_val!(examinee.latent, val)
end

#take the last value of the chain and set it as value
function set_val_from_chain!(item::AbstractItem)
    set_val_from_chain!(item.parameters)
end

function set_val_from_chain!(examinee::AbstractExaminee)
    set_val_from_chain!(examinee.latent)
end


#update the posterior, append sample to chain and set the value as a sample from the posterior
function mcmc_iter!(
    item::AbstractItem,
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{<:AbstractResponse},
    W::Vector{PolyaGammaSample};
    sampling = true,
)
    update_posterior!(item, examinees, responses, W)
    #set_val_from_chain!(item)
    set_val_from_posterior!(item; sampling = sampling)
end

function mcmc_iter!(
    examinee:: AbstractExaminee,
    items::Vector{<:AbstractItem},
    responses::Vector{<:AbstractResponse},
    W::Vector{PolyaGammaSample};
    sampling = true,
)
    update_posterior!(examinee, items, responses, W)
    #chain_append!(examinee; sampling = sampling)
    set_val_from_posterior!(examinee; sampling = sampling)
    #set_val_from_chain!(examinee)
end

#update the estimate as the mean of the chain values
function update_estimate!(examinee::AbstractExaminee; sampling = true)
    update_estimate!(examinee.latent, sampling = sampling)
end

function update_estimate!(item::AbstractItem; sampling = true)
    update_estimate!(item.parameters; sampling = sampling)
end
