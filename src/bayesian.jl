struct PolyaGammaSample
    i_idx::Int64
    e_idx::Int64
    val::Float64
    PolyaGammaSample(i_idx, e_idx, val) = new(i_idx, e_idx, val)
end

function posterior(
    item::Item2PL,
    examinees::Vector{<:AbstractExaminee}, #must be sorted by e.idx
    responses::Vector{<:AbstractResponse}, #only responses of item sorted by e.idx
    W::Vector{PolyaGammaSample}, #sorted by e.idx
)
    prior = item.parameters.prior.v
    a = item.parameters.a
    b = item.parameters.b
    #W = map( e -> Distributions.rand(PolyaGammaDevRoye1Sampler(1.0, item.parameters.a *(e.latent.val - item.parameters.b))), examinees)
    sigma2 = mapreduce((e, w) -> [(e.latent.val - b)^2, a^2] .* w.val, +, examinees, W)
    sigma2 = 1 ./ (sigma2 + (1 ./ Distributions.var.(prior)))
    mu = mapreduce(
        (e, w, r) -> [
            (e.latent.val - b) * (r.val - 0.5),
            -a * (
                #(get_responses_by_examinee_id(e.id, responses)[1].val - 0.5) -
                (r.val - 0.5) - (a * e.latent.val * w.val)
            ),
        ],
        +,
        examinees,
        W,
        responses,
    )
    mu = sigma2 .* (mu + (Distributions.mean.(prior) ./ Distributions.var.(prior)))
    return Distributions.Product([
        Distributions.TruncatedNormal(mu[1], sqrt(sigma2[1]), 0.0, Inf),
        Distributions.Normal(mu[2], sqrt(sigma2[2])),
    ])
end

function update_posterior!(
    item::Item2PL,
    examinees::Vector{<:AbstractExaminee}, #must be sorted by e.idx
    responses::Vector{<:AbstractResponse}, #only responses of item sorted by e.idx
    W::Vector{PolyaGammaSample}, #sorted by e.idx
)
    item.parameters.posterior = posterior(item, examinees, responses, W)
end

function posterior(
    examinee::Examinee1D,
    items_e::Vector{<:AbstractItem}, #only items answered by examinee sorted by i.idx
    responses_e::Vector{<:AbstractResponse}, #only responses of examinee sorted by i.idx
    W::Vector{PolyaGammaSample},
)
    prior = examinee.latent.prior
    sigma2 = mapreduce((i, w) -> (i.parameters.a^2) * w.val, +, items_e, W)
    sigma2 = 1 / (sigma2 + (1 / Distributions.var(prior)))
    mu =
        sigma2 * (
            mapreduce(
                (i, w, r) ->
                    i.parameters.a * (i.parameters.a * i.parameters.b * w.val +
                     #(get_responses_by_item_id(i.id, responses_e)[1].val - 0.5)
                     (r.val - 0.5)),
                +,
                items_e,
                W,
                responses_e,
            ) + (prior.μ / Distributions.var(prior))
        )
    return Distributions.Normal(mu, sqrt(sigma2))::Distributions.ContinuousDistribution
end
function update_posterior!(
    examinee::Examinee1D,
    items_e::Vector{<:AbstractItem}, #only items answered by examinee sorted by i.idx
    responses_e::Vector{<:AbstractResponse}, #only responses of examinee sorted by i.idx
    W::Vector{PolyaGammaSample},
)
    examinee.latent.posterior = posterior(examinee, items_e, responses_e, W)
end

function posterior(
    item::Item2PL, 
    examinee::Examinee1D, 
    w::PolyaGammaSample,
    r::Response
    )
    a = item.parameters.a
    b = item.parameters.b
    latent_val = examinee.latent.val
    r_val = r.val
    w_val = w.val

    sigma2 = [(latent_val - b)^2 * w.val, a^2 * w_val]
    sigma2 = 1 ./ (sigma2 .+ (1 ./ Distributions.var.(item.parameters.prior.v)))
    mu = [(latent_val - b) * (r_val - 0.5), -a * ((r_val - 0.5) - (a * latent_val * w_val))]
    mu =
        sigma2 .* (
            mu + (
                Distributions.mean.(item.parameters.prior.v) ./
                Distributions.var.(item.parameters.prior.v)
            )
        )
    return Distributions.Product([
        Distributions.TruncatedNormal(mu[1], sqrt(sigma2[1]), 0.0, Inf),
        Distributions.Normal(mu[2], sqrt(sigma2[2])),
    ])
end

function posterior(
    examinee::Examinee1D,
    item::Item2PL,
    w::PolyaGammaSample,
    r::Response
    )
    sigma2 = (item.parameters.a^2) * w.val
    sigma2 = 1 / (sigma2 + (1 / Distributions.var(examinee.latent.prior)))
    mu =
        sigma2 * (
            item.parameters.a *
            (item.parameters.a * item.parameters.b * w.val + (r.val - 0.5)) +
            (examinee.latent.prior.μ / Distributions.var(examinee.latent.prior))
        )
    return Distributions.Normal(mu, sqrt(sigma2))
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
function generate_w(item::AbstractItem, examinees::Vector{<:AbstractExaminee})
    return map(
        e -> PolyaGammaSample(
            item.idx,
            e.idx,
            Distributions.rand(PolyaGamma(
                1,
                item.parameters.a * (e.latent.val - item.parameters.b),
            )),
        ),
        examinees,
    )
end

function generate_w(items::Vector{<:AbstractItem}, examinees_i::Vector{Vector{Examinee1D}})
    return mapreduce(
        i -> map(
            e -> PolyaGammaSample(
                i.idx,
                e.idx,
                Distributions.rand(Psychometrics.PolyaGamma(
                    1,
                    i.parameters.a * (e.latent.val - i.parameters.b),
                )),
            ),
            examinees_i[i.idx],
        ),
        vcat,
        items,
    )
end

#extract a random value from posterior and set it as value
function set_value_from_posterior!(item::Item2PL; sampling = true)
    vals = chain_append!(item; sampling = sampling)
    item.parameters.a = vals[1]
    item.parameters.b = vals[2]
end

function set_value_from_posterior!(examinee::Examinee1D; sampling = true)
    val = chain_append!(examinee; sampling = sampling)
    examinee.latent.val = val
end

#take the last value of the chain and set it as value
function set_value_from_chain!(item::Item2PL)
    item.parameters.a = item.parameters.chain[end][1]
    item.parameters.b = item.parameters.chain[end][2]
end

function set_value_from_chain!(examinee::Examinee1D)
    examinee.latent.val = examinee.latent.chain[end]
end

#extract a value from the posterior and append it to the chain
function chain_append!(item::Union{Item2PL,Item3PL}; sampling = false)
    val = Distributions.rand(item.parameters.posterior)
    if (sampling && size(item.parameters.chain, 1) >= 1000)
        item.parameters.chain[Random.rand(1:1000)] = val
    else
        push!(item.parameters.chain, val)
    end
    return val::Vector{Float64}
end

function chain_append!(examinee::Examinee1D; sampling = false)
    val = Distributions.rand(examinee.latent.posterior)
    if (sampling && size(examinee.latent.chain, 1) >= 1000)
        examinee.latent.chain[Random.rand(1:1_000)] = val
    else
        push!(examinee.latent.chain, val)
    end
    return val::Float64
end


#update the posterior, append sample to chain and set the value as a sample from the posterior
function mcmc_iter!(
    item::Item2PL,
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{<:Response},
    W::Vector{PolyaGammaSample};
    sampling = true,
)
    update_posterior!(item, examinees, responses, W)
    #set_value_from_chain!(item)
    set_value_from_posterior!(item; sampling = sampling)
end

function mcmc_iter!(
    examinee::Examinee1D,
    items::Vector{<:AbstractItem},
    responses::Vector{<:Response},
    W::Vector{PolyaGammaSample};
    sampling = true,
)
    update_posterior!(examinee, items, responses, W)
    #chain_append!(examinee; sampling = sampling)
    set_value_from_posterior!(examinee; sampling = sampling)
    #set_value_from_chain!(examinee)
end

#update the estimate as the mean of the chain values
function update_estimate!(examinee::Examinee1D; sampling = true)
    chain_size = size(examinee.latent.chain, 1)
    if sampling
        examinee.latent.val = sum(examinee.latent.chain[(chain_size - min(999, chain_size - 1)) : end]) / min(1000, chain_size)
    else
        examinee.latent.val = sum(examinee.latent.chain) / chain_size
    end
end

function update_estimate!(item::Item2PL; sampling = true)
    chain_size = size(item.parameters.chain, 1)
    if sampling 
        chain_matrix = hcat(item.parameters.chain[(chain_size - min(999, chain_size - 1)) : end]...)
        vals = [sum(i) / min(1000, chain_size) for i in eachrow(chain_matrix)]
    else
        chain_matrix = hcat(item.parameters.chain...)
        vals = [sum(i) / chain_size for i in eachrow(chain_matrix)]   
    end
    item.parameters.a = vals[1]
    item.parameters.b = vals[2]
end
